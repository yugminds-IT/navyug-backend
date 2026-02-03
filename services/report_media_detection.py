"""
Run face detection on report media (videos/photos) and create DetectedFace records.
Fetches video/images from S3 (or local), runs core VideoProcessor, compares with student
face database, stores DetectedFace in DB and optionally uploads face crops and embeddings to S3.
"""
import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional
from io import BytesIO
from sqlalchemy.orm import Session
from sqlalchemy import or_

import numpy as np
import config
from database.models import User, UserRole, Report, ReportMedia, DetectedFace
from core.video_processor import VideoProcessor
from core.logger import logger
from storage.s3_paths import get_report_detected_face_s3_path, campus_report_embeddings_path


def _student_id_from_person_id(db: Session, person_id: str, college_id: int) -> Optional[int]:
    """
    Resolve face-db person_id (e.g. STUDENT_001) to User.id (student) in the same college.
    Matches login_id, roll_number, or college_student_id.
    """
    student = db.query(User).filter(
        User.role == UserRole.STUDENT,
        User.college_id == college_id,
        or_(
            User.login_id == person_id,
            User.roll_number == person_id,
            User.college_student_id == person_id,
        ),
    ).first()
    return student.id if student else None


def _matched_to_detected_face(
    db: Session,
    report_id: int,
    college_id: int,
    person_id: str,
    name: str,
    confidence: float,
    bbox: Optional[List] = None,
    detected_image_url: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_detected_key: Optional[str] = None,
) -> Optional[DetectedFace]:
    """Create a DetectedFace record if we can resolve person_id to a student."""
    student_id = _student_id_from_person_id(db, person_id, college_id)
    if not student_id:
        logger.debug(f"Report {report_id}: no User (student) for person_id={person_id}, skipping DetectedFace")
        return None

    student = db.query(User).filter(User.id == student_id).first()
    if not student:
        return None

    # confidence from recognizer is 0–1; DB stores 0–100
    confidence_int = min(100, max(0, int(round(confidence * 100))))

    # bounding_box: store as { x, y, width, height } if bbox is [x1,y1,x2,y2]
    bbox_json = None
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        bbox_json = {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}

    return DetectedFace(
        report_id=report_id,
        student_id=student_id,
        name=name,
        department=getattr(student, "branch", None) or getattr(student, "department", None),
        year=getattr(student, "year", None),
        confidence=confidence_int,
        previous_incidents=student.incidents or 0,
        bounding_box=bbox_json,
        reference_image_url=student.face_image_url,
        detected_image_url=detected_image_url,
        s3_bucket=s3_bucket,
        s3_detected_key=s3_detected_key,
    )


def run_report_media_face_detection(
    report_id: int,
    college_id: int,
    media_list: List[Dict[str, Any]],
) -> None:
    """
    Run face detection on report media (videos and images) and persist DetectedFace records.
    Call from a background thread after report and media are saved.

    media_list: [ {"content": bytes, "is_video": bool, "file_ext": str}, ... ]
             or [ {"s3_key": str, "is_video": bool, "file_ext": str}, ... ] when media is in S3 (data taken from S3).
    """
    if not config.face_db or not config.db:
        logger.warning("Report media detection skipped: face_db or db not initialized")
        return

    try:
        with config.db.get_session() as db:
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                logger.warning(f"Report {report_id} not found for face detection")
                return

            college_code = report.college.college_code if report.college else None
            processor = VideoProcessor(config.face_db)
            all_matched: List[Dict[str, Any]] = []
            all_detected_embeddings: List[np.ndarray] = []

            for idx, item in enumerate(media_list):
                s3_key = item.get("s3_key")
                content = item.get("content")
                is_video = item.get("is_video", False)
                file_ext = item.get("file_ext", "")

                # When s3_key is provided, fetch data from S3 (videos and photos stored in S3)
                if s3_key and config.s3_client and college_code:
                    try:
                        if is_video:
                            content = None  # video will be fetched from S3 inside process_video_sync
                        else:
                            buf = config.s3_client.download_fileobj(s3_key, college_code=college_code)
                            content = buf.read()
                    except Exception as dl_err:
                        logger.warning(f"Report {report_id} media {idx}: S3 download failed: {dl_err}")
                        continue

                if is_video:
                    if s3_key and config.s3_client and college_code:
                        # Take video from S3; frames will be stored in S3 by process_video_sync
                        video_id = f"report_{report_id}_{uuid.uuid4().hex[:8]}"
                        if video_id not in config.job_store:
                            config.job_store[video_id] = {}
                        upload_frames = bool(
                            getattr(config, "REPORT_UPLOAD_FRAMES_TO_S3", True)
                            and config.s3_client
                            and college_code
                        )
                        try:
                            processor.process_video_sync(
                                video_id,
                                video_path=None,
                                s3_key=s3_key,
                                college_code=college_code,
                                report_id=report.id,
                                report_id_str=report.report_id,
                                upload_frames_to_s3=upload_frames,
                            )
                        except Exception as e:
                            logger.error(f"Report {report_id} video from S3 failed: {e}", exc_info=True)
                            continue
                        result = config.job_store.get(video_id, {}).get("result", {})
                        matched = result.get("matched_persons", [])
                        for m in matched:
                            all_matched.append({
                                "person_id": m.get("person_id"),
                                "name": m.get("name", m.get("person_id", "")),
                                "confidence": m.get("confidence", 0.0),
                                "bbox": None,
                                "crop_bytes": m.get("crop_bytes"),
                            })
                        emb_lists = result.get("detected_embeddings") or []
                        for e in emb_lists:
                            all_detected_embeddings.append(np.array(e))
                    elif content:
                        # Legacy: content (bytes) provided
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=file_ext or ".mp4"
                        ) as tmp:
                            tmp.write(content)
                            tmp_path = tmp.name
                        video_id = f"report_{report_id}_{uuid.uuid4().hex[:8]}"
                        if video_id not in config.job_store:
                            config.job_store[video_id] = {}
                        upload_frames = bool(
                            getattr(config, "REPORT_UPLOAD_FRAMES_TO_S3", True)
                            and config.s3_client
                            and college_code
                        )
                        try:
                            processor.process_video_sync(
                                video_id,
                                tmp_path,
                                report_id=report.id,
                                college_code=college_code,
                                report_id_str=report.report_id,
                                upload_frames_to_s3=upload_frames,
                            )
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                        result = config.job_store.get(video_id, {}).get("result", {})
                        matched = result.get("matched_persons", [])
                        for m in matched:
                            all_matched.append({
                                "person_id": m.get("person_id"),
                                "name": m.get("name", m.get("person_id", "")),
                                "confidence": m.get("confidence", 0.0),
                                "bbox": None,
                                "crop_bytes": m.get("crop_bytes"),
                            })
                        emb_lists = result.get("detected_embeddings") or []
                        for e in emb_lists:
                            all_detected_embeddings.append(np.array(e))
                else:
                    if not content:
                        continue
                    try:
                        matched = processor.process_image_for_report(content)
                        all_matched.extend(matched)
                        for m in matched:
                            if m.get("embedding") is not None:
                                all_detected_embeddings.append(np.asarray(m["embedding"]))
                    except Exception as e:
                        logger.error(
                            f"Report {report_id} media {idx} face detection failed: {e}",
                            exc_info=True,
                        )

            # Dedupe by person_id, keep max confidence
            by_person: Dict[str, Dict[str, Any]] = {}
            for m in all_matched:
                pid = m.get("person_id")
                if not pid:
                    continue
                if pid not in by_person or m.get("confidence", 0) > by_person[pid].get("confidence", 0):
                    by_person[pid] = m

            created_count = 0
            for m in by_person.values():
                detected_image_url = None
                s3_bucket = None
                s3_detected_key = None
                crop_bytes = m.get("crop_bytes")
                if crop_bytes and config.s3_client and college_code:
                    try:
                        face_id = uuid.uuid4().hex[:12]
                        s3_key = get_report_detected_face_s3_path(
                            college_code,
                            report_id,
                            m["person_id"],
                            face_id,
                            report_id_str=report.report_id if report else None,
                        )
                        detected_image_url = config.s3_client.upload_fileobj(
                            BytesIO(crop_bytes),
                            s3_key,
                            college_code=college_code,
                            content_type="image/jpeg",
                        )
                        s3_bucket = config.s3_client.get_bucket_name(college_code)
                        s3_detected_key = s3_key
                    except Exception as up_err:
                        logger.warning(f"Failed to upload detected face crop to S3: {up_err}")
                df = _matched_to_detected_face(
                    db,
                    report_id,
                    college_id,
                    m["person_id"],
                    m.get("name", m["person_id"]),
                    m.get("confidence", 0.0),
                    m.get("bbox"),
                    detected_image_url=detected_image_url,
                    s3_bucket=s3_bucket,
                    s3_detected_key=s3_detected_key,
                )
                if df:
                    db.add(df)
                    created_count += 1

            # Upload report detected embeddings to S3 (reports/.../embeddings/detected_embeddings.npy)
            if all_detected_embeddings and config.s3_client and college_code:
                try:
                    report_id_str = report.report_id if report else str(report_id)
                    s3_key = campus_report_embeddings_path(college_code, report_id_str, "detected_embeddings.npy")
                    arr = np.vstack(all_detected_embeddings)
                    buf = BytesIO()
                    np.save(buf, arr, allow_pickle=False)
                    buf.seek(0)
                    config.s3_client.upload_fileobj(
                        buf,
                        s3_key,
                        college_code=college_code,
                        content_type="application/octet-stream",
                    )
                    logger.debug(f"Uploaded detected_embeddings.npy to S3 for report {report_id}")
                except Exception as emb_err:
                    logger.warning(f"Failed to upload report embeddings to S3: {emb_err}")

            report.ai_processed = True
            db.commit()
            logger.info(
                f"Report {report_id}: face detection completed, "
                f"{len(by_person)} unique persons matched, {created_count} DetectedFace records created"
            )
    except Exception as e:
        logger.error(f"Report media face detection failed for report {report_id}: {e}", exc_info=True)


def _s3_key_from_file_url(file_url: str) -> Optional[str]:
    """Parse s3://bucket/key to s3 key (path part)."""
    if not file_url or not file_url.startswith("s3://"):
        return None
    parts = file_url.split("/", 3)
    return parts[3] if len(parts) >= 4 else None


def run_report_face_detection_from_storage(report_id: int) -> None:
    """
    Fetch report media from S3 (or local path), run face detection using core VideoProcessor,
    compare with student face database, and store DetectedFace records + upload face crops to S3.

    Use this to (re)process a report when media is already in S3 (e.g. trigger from management UI).
    """
    if not config.face_db or not config.db:
        logger.warning("Report face detection from storage skipped: face_db or db not initialized")
        return

    try:
        with config.db.get_session() as db:
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                logger.warning(f"Report {report_id} not found for face detection from storage")
                return
            college_id = report.college_id
            college_code = report.college.college_code if report.college else None
            media_records = db.query(ReportMedia).filter(ReportMedia.report_id == report_id).all()
            if not media_records:
                logger.warning(f"Report {report_id} has no media to process")
                return

            # Build media_list with s3_key when media is in S3 (detection will fetch from S3); otherwise content from local
            media_list: List[Dict[str, Any]] = []
            for rec in media_records:
                file_url = rec.file_url or ""
                s3_key_stored = getattr(rec, "s3_key", None)
                is_video = rec.media_type and rec.media_type.value == "video"
                file_ext = ".mp4" if is_video else ".jpg"

                if s3_key_stored and config.s3_client and college_code:
                    # Data in S3: pass s3_key so detection fetches from S3 (videos and frames stored in S3)
                    media_list.append({
                        "s3_key": s3_key_stored,
                        "is_video": is_video,
                        "file_ext": file_ext or (".mp4" if is_video else ".jpg"),
                    })
                elif file_url.startswith("s3://"):
                    s3_key = _s3_key_from_file_url(file_url)
                    if s3_key and config.s3_client and college_code:
                        media_list.append({
                            "s3_key": s3_key,
                            "is_video": is_video,
                            "file_ext": file_ext or (".mp4" if is_video else ".jpg"),
                        })
                    else:
                        logger.warning(f"Report {report_id} media: missing s3_client or key")
                else:
                    # Local path: download to content for backward compatibility
                    content = None
                    if os.path.isfile(file_url):
                        with open(file_url, "rb") as f:
                            content = f.read()
                    else:
                        logger.warning(f"Report {report_id} media local path not found: {file_url}")
                    if content:
                        media_list.append({"content": content, "is_video": is_video, "file_ext": file_ext or (".mp4" if is_video else ".jpg")})

            if not media_list:
                logger.warning(f"Report {report_id}: no media content could be loaded")
                return

        # Clear existing DetectedFace for this report so we don't duplicate (optional: keep or replace)
        try:
            with config.db.get_session() as db:
                db.query(DetectedFace).filter(DetectedFace.report_id == report_id).delete()
                report = db.query(Report).filter(Report.id == report_id).first()
                if report:
                    report.ai_processed = False
                db.commit()
        except Exception as e:
            logger.warning(f"Could not clear existing detections for report {report_id}: {e}")

        run_report_media_face_detection(report_id, college_id, media_list)
    except Exception as e:
        logger.error(f"Report face detection from storage failed for report {report_id}: {e}", exc_info=True)
