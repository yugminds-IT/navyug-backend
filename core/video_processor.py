import os
import time
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Any
from collections import defaultdict

from core.frame_extractor import FrameExtractor
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.face_database import FaceDatabase
from core.utils import apply_nms
from core.image_quality import detect_duplicate_images, calculate_image_quality_score
from core.logger import logger
import config

try:
    from storage.s3_paths import campus_report_frames_path, campus_report_detected_faces_path
except ImportError:
    campus_report_frames_path = None
    campus_report_detected_faces_path = None


class VideoProcessor:
    """
    Async video processor that orchestrates the entire pipeline:
    Video → Extract faces → Deduplicate → Keep best quality → Match with database
    """
    
    def __init__(self, face_database: FaceDatabase):
        """
        Initialize video processor.
        
        Args:
            face_database: Loaded face database instance
        """
        self.face_database = face_database
        self.face_detector = FaceDetector(
            ctx_id=config.CTX_ID, 
            det_size=config.DETECTION_SIZE,
            det_thresh=config.DETECTION_THRESHOLD,
            recognition=True
        )
        self.face_recognizer = FaceRecognizer(
            similarity_threshold=config.FACE_MATCH_THRESHOLD,
            use_quality_weighting=config.USE_QUALITY_WEIGHTED_MATCHING,
            use_statistical=config.USE_STATISTICAL_MATCHING
        )
    
    def process_image_for_report(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect and match faces in a single image (for report media).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG/etc.)
            
        Returns:
            List of matched persons: [{"person_id": str, "name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Could not decode image for report face detection")
            return []
        
        db_data = self.face_database.get_embeddings_matrix()
        if len(db_data[1]) == 0:
            return []
        
        if config.ENABLE_MULTI_SCALE_DETECTION:
            faces = self.face_detector.detect_multi_scale(
                frame,
                scales=config.DETECTION_SCALES,
                preprocess=config.ENABLE_IMAGE_PREPROCESSING
            )
        else:
            faces = self.face_detector.detect_and_embed(
                frame,
                preprocess=config.ENABLE_IMAGE_PREPROCESSING,
                align_faces=config.ENABLE_FACE_ALIGNMENT
            )
        
        faces = [f for f in faces if f.get("confidence", 0) >= config.FACE_DETECTION_CONFIDENCE]
        if config.MIN_FACE_SIZE > 0:
            faces = [
                f for f in faces
                if (f["bbox"][2] - f["bbox"][0]) >= config.MIN_FACE_SIZE
                and (f["bbox"][3] - f["bbox"][1]) >= config.MIN_FACE_SIZE
            ]
        if config.ENABLE_NMS and faces:
            faces = apply_nms(faces, iou_threshold=config.NMS_IOU_THRESHOLD)
        
        if not faces:
            return []
        
        query_embeddings = np.vstack([f["embedding"] for f in faces])
        face_qualities_list = []
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            pad = 20
            face_img = frame[
                max(0, y1 - pad):min(frame.shape[0], y2 + pad),
                max(0, x1 - pad):min(frame.shape[1], x2 + pad),
            ]
            face_qualities_list.append(calculate_image_quality_score(face_img, f["bbox"]))
        face_qualities = np.array(face_qualities_list)
        if face_qualities.ndim == 0:
            face_qualities = np.array([float(face_qualities)])
        
        match_results = self.face_recognizer.match_faces_batch(
            query_embeddings,
            db_data,
            face_qualities=face_qualities,
            use_adaptive_threshold=config.USE_ADAPTIVE_THRESHOLD
        )
        
        result_list = []
        for face, (person_id, confidence) in zip(faces, match_results):
            if not person_id or confidence <= 0:
                continue
            p_info = self.face_database.get_person_info(person_id)
            name = p_info["name"] if p_info else person_id
            # Crop face for optional S3 storage (JPEG bytes)
            crop_bytes = None
            try:
                x1, y1, x2, y2 = [int(v) for v in face["bbox"][:4]]
                pad = 20
                face_img = frame[
                    max(0, y1 - pad):min(frame.shape[0], y2 + pad),
                    max(0, x1 - pad):min(frame.shape[1], x2 + pad),
                ]
                _, jpg_bytes = cv2.imencode(".jpg", face_img)
                if jpg_bytes is not None:
                    crop_bytes = jpg_bytes.tobytes()
            except Exception:
                pass
            result_list.append({
                "person_id": person_id,
                "name": name,
                "confidence": round(float(confidence), 3),
                "bbox": face.get("bbox"),
                "crop_bytes": crop_bytes,
                "embedding": face.get("embedding"),
            })
        return result_list
    
    def process_video_sync(
        self,
        video_id: str,
        video_path: Optional[str] = None,
        *,
        s3_key: Optional[str] = None,
        college_code: Optional[str] = None,
        report_id: Optional[int] = None,
        report_id_str: Optional[str] = None,
        upload_frames_to_s3: bool = False,
    ):
        """
        Process video with optimized workflow:
        1. Stream video processing (Video -> Faces)
        2. Incremental deduplication (Keep best quality unique faces)
        3. Vectorized matching against database
        4. Return identification results

        Video source: pass either video_path (local file) or (s3_key + college_code) to fetch from S3.
        When upload_frames_to_s3 is True and college_code + (report_id_str or report_id) are set,
        extracted frames are uploaded to S3 under reports/.../frames/ (every Nth frame per config).
        """
        temp_video_path: Optional[str] = None
        if (not video_path or not video_path.strip()) and s3_key and college_code and config.s3_client:
            try:
                import tempfile
                suffix = ".mp4"
                if "." in s3_key:
                    suffix = "." + s3_key.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    temp_video_path = tmp.name
                config.s3_client.download_file(s3_key, temp_video_path, college_code=college_code)
                video_path = temp_video_path
                logger.info(f"Fetched video from S3: {s3_key}")
            except Exception as e:
                logger.error(f"Failed to fetch video from S3: {e}", exc_info=True)
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                    except OSError:
                        pass
                config.job_store[video_id] = config.job_store.get(video_id, {})
                config.job_store[video_id]["status"] = "failed"
                config.job_store[video_id]["error"] = f"S3 download failed: {e}"
                return
        if not video_path or not video_path.strip():
            logger.error("process_video_sync: need video_path or (s3_key + college_code)")
            config.job_store[video_id] = config.job_store.get(video_id, {})
            config.job_store[video_id]["status"] = "failed"
            config.job_store[video_id]["error"] = "Missing video_path or S3 source"
            return
        try:
            # Update status to processing
            processing_start_time = time.time()
            
            # Update database if available
            try:
                if config.db:
                    from services.video_service import VideoService
                    from database.models import JobStatus
                    
                    with config.db.get_session() as db:
                        VideoService.update_job_status(
                            db=db,
                            video_id=video_id,
                            status=JobStatus.PROCESSING,
                            progress_percentage=0.0,
                            progress_data={
                                "frames_processed": 0,
                                "total_frames": 0,
                                "percentage": 0.0,
                                "stage": "initializing"
                            }
                        )
            except Exception:
                pass
            
            # Update in-memory store for backward compatibility
            if video_id in config.job_store:
                config.job_store[video_id]["status"] = "processing"
                config.job_store[video_id]["started_at"] = processing_start_time
                config.job_store[video_id]["progress"] = {
                    "frames_processed": 0,
                    "total_frames": 0,
                    "percentage": 0.0,
                    "stage": "initializing",
                    "updated_at": processing_start_time
                }
            
            start_time = time.time()
            extraction_start = None
            matching_start = None
            
            # Production: no local debug storage; all detected faces go to S3 only
            
            # Get database embeddings as matrix for vectorized matching
            # This is much faster than the dictionary based approach
            db_data = self.face_database.get_embeddings_matrix()
            db_matrix, db_person_ids = db_data
            
            # Early exit if database is empty
            if len(db_person_ids) == 0:
                logger.warning("Database is empty, skipping face matching")
                result = {
                    "status": "completed",
                    "video_id": video_id,
                    "total_frames_processed": 0,
                    "faces_detected": 0,
                    "unique_faces_after_dedup": 0,
                    "matched_persons": [],
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "warning": "Database is empty, no faces matched"
                }
                config.job_store[video_id]["result"] = result
                config.job_store[video_id]["status"] = "completed"
                return
            
            # ============================================
            # STEP 1 & 2: Extract and Deduplicate Stream
            # ============================================
            extraction_start = time.time()
            logger.info("STEP 1 & 2: Extracting and incrementally deduplicating faces")
            
            unique_faces = [] # List of dicts
            unique_embeddings_list = [] # List of arrays for fast stacking
            face_image_counts = {}  # Track how many images saved per unique face: {face_idx: count}
            
            total_faces_detected = 0
            frames_processed = 0
            duplicates_removed_count = 0
            images_skipped_limit = 0  # Count images skipped due to limit
            
            with FrameExtractor(video_path, frames_per_second=config.FRAME_EXTRACTION_RATE) as extractor:
                video_info = extractor.get_video_info()
                total_frames = video_info["frames_to_extract"]
                config.job_store[video_id]["progress"]["total_frames"] = total_frames
                
                for frame_data in extractor.extract_frames():
                    frame = frame_data["frame"]
                    frame_id = frame_data["frame_id"]
                    timestamp = frame_data["timestamp"]
                    
                    # Optional: upload frame to S3 (reports/.../frames/) for report videos
                    if (
                        upload_frames_to_s3
                        and config.s3_client
                        and college_code
                        and (report_id_str is not None or report_id is not None)
                        and campus_report_frames_path is not None
                        and (frame_id % max(1, getattr(config, "REPORT_FRAME_UPLOAD_INTERVAL", 5)) == 0)
                    ):
                        try:
                            rid = report_id_str if report_id_str is not None else str(report_id)
                            s3_key = campus_report_frames_path(
                                college_code, rid, f"frame_{frame_id:06d}.jpg"
                            )
                            _, jpg_bytes = cv2.imencode(".jpg", frame)
                            if jpg_bytes is not None:
                                config.s3_client.upload_fileobj(
                                    BytesIO(jpg_bytes.tobytes()),
                                    s3_key,
                                    college_code=college_code,
                                    content_type="image/jpeg",
                                )
                        except Exception as up_err:
                            logger.debug("Frame upload to S3 skipped: %s", up_err)
                    
                    # Detect faces
                    if config.ENABLE_MULTI_SCALE_DETECTION:
                        faces = self.face_detector.detect_multi_scale(
                            frame,
                            scales=config.DETECTION_SCALES,
                            preprocess=config.ENABLE_IMAGE_PREPROCESSING
                        )
                    else:
                        faces = self.face_detector.detect_and_embed(
                            frame,
                            preprocess=config.ENABLE_IMAGE_PREPROCESSING,
                            align_faces=config.ENABLE_FACE_ALIGNMENT
                        )
                    
                    # Basic filtering
                    faces = [f for f in faces if f["confidence"] >= config.FACE_DETECTION_CONFIDENCE]
                    if config.MIN_FACE_SIZE > 0:
                        faces = [f for f in faces if (f["bbox"][2] - f["bbox"][0]) >= config.MIN_FACE_SIZE]
                    
                    if config.ENABLE_NMS and faces:
                        faces = apply_nms(faces, iou_threshold=config.NMS_IOU_THRESHOLD)
                    
                    total_faces_detected += len(faces)
                    
                    rid = report_id_str if report_id_str is not None else (str(report_id) if report_id is not None else None)
                    
                    # INCREMENTAL DEDUPLICATION (and save/upload every detected face for debug + S3)
                    for face_idx, face in enumerate(faces):
                        # Prepare face data
                        x1, y1, x2, y2 = face["bbox"]
                        padding = 20
                        x1_p = max(0, x1 - padding)
                        y1_p = max(0, y1 - padding)
                        x2_p = min(frame.shape[1], x2 + padding)
                        y2_p = min(frame.shape[0], y2 + padding)
                        face_img = frame[y1_p:y2_p, x1_p:x2_p]
                        
                        quality_score = calculate_image_quality_score(face_img, face["bbox"])
                        
                        face_data = {
                            'embedding': face["embedding"],
                            'image': face_img,
                            'face_bbox': face["bbox"],
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'detection_confidence': face["confidence"],
                            'quality_score': quality_score
                        }
                        
                        # Check against unique_faces (OPTIMIZED: cache matrix)
                        is_duplicate = False
                        best_match_idx = -1
                        
                        if unique_embeddings_list:
                            # OPTIMIZATION: Only rebuild matrix if it doesn't exist or has changed
                            # Cache the matrix to avoid repeated vstack operations
                            if not hasattr(self, '_cached_unique_matrix') or \
                               len(self._cached_unique_matrix) != len(unique_embeddings_list):
                                self._cached_unique_matrix = np.vstack(unique_embeddings_list)
                            
                            query_matrix = face["embedding"].reshape(1, -1)
                            
                            # (1, N) similarity matrix
                            sims = self.face_recognizer.compute_similarity_matrix(query_matrix, self._cached_unique_matrix)[0]
                            
                            # Find max similarity
                            max_sim_idx = np.argmax(sims)
                            max_sim = sims[max_sim_idx]
                            
                            if max_sim >= config.VIDEO_DUPLICATE_SIMILARITY_THRESHOLD:
                                is_duplicate = True
                                best_match_idx = max_sim_idx
                        
                        # Determine which unique face index this belongs to (for image counting)
                        if is_duplicate:
                            target_face_idx = best_match_idx
                        else:
                            # For new face, index will be len(unique_faces) after append
                            target_face_idx = len(unique_faces)
                        
                        # Upload detected face to S3 only if under limit (max 5 images per unique face)
                        should_save_image = True
                        if target_face_idx in face_image_counts:
                            if face_image_counts[target_face_idx] >= getattr(config, "MAX_IMAGES_PER_FACE", 5):
                                should_save_image = False
                                images_skipped_limit += 1
                        
                        if should_save_image and config.s3_client and college_code and rid and campus_report_detected_faces_path is not None:
                            timestamp_str = f"{timestamp:.2f}".replace(".", "_")
                            s3_fname = f"frame_{frame_id:06d}_face_{face_idx:03d}_t{timestamp_str}.jpg"
                            try:
                                _, jpg_bytes = cv2.imencode(".jpg", face_img)
                                if jpg_bytes is not None:
                                    config.s3_client.upload_fileobj(
                                        BytesIO(jpg_bytes.tobytes()),
                                        campus_report_detected_faces_path(college_code, rid, s3_fname),
                                        college_code=college_code,
                                        content_type="image/jpeg",
                                    )
                                    # Increment count for this unique face
                                    face_image_counts[target_face_idx] = face_image_counts.get(target_face_idx, 0) + 1
                            except Exception as up_err:
                                logger.debug("Detected face upload to S3 skipped: %s", up_err)
                        
                        if is_duplicate:
                            duplicates_removed_count += 1
                            # If new face has better quality, replace the old one
                            if quality_score > unique_faces[best_match_idx]['quality_score']:
                                unique_faces[best_match_idx] = face_data
                                unique_embeddings_list[best_match_idx] = face["embedding"]
                                # Invalidate cache since matrix changed
                                if hasattr(self, '_cached_unique_matrix'):
                                    del self._cached_unique_matrix
                        else:
                            # New unique face - append first, then track index
                            unique_faces.append(face_data)
                            unique_embeddings_list.append(face["embedding"])
                            # Initialize image count for new unique face (if not already set by upload above)
                            if target_face_idx not in face_image_counts:
                                face_image_counts[target_face_idx] = 0
                            # Invalidate cache since matrix changed
                            if hasattr(self, '_cached_unique_matrix'):
                                del self._cached_unique_matrix
                    
                    frames_processed += 1
                    
            # Update progress less frequently for better performance (every 10 frames instead of 5)
            if frames_processed % 10 == 0:
                percentage = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                progress_data = {
                    "frames_processed": frames_processed,
                    "total_frames": total_frames,
                    "percentage": round(percentage, 2),
                    "stage": "processing_stream",
                    "faces_collected": total_faces_detected,
                    "unique_faces": len(unique_faces),
                    "updated_at": time.time()
                }
                
                # Update database if available
                try:
                    if config.db:
                        from services.video_service import VideoService
                        from database.models import JobStatus
                        
                        with config.db.get_session() as db:
                            VideoService.update_job_status(
                                db=db,
                                video_id=video_id,
                                status=JobStatus.PROCESSING,
                                progress_percentage=round(percentage, 2),
                                progress_data=progress_data
                            )
                except Exception:
                    pass
                
                # Update in-memory store
                if video_id in config.job_store:
                    config.job_store[video_id]["progress"].update(progress_data)
            
            extraction_time = time.time() - extraction_start
            logger.info(f"Processed {frames_processed} frames in {extraction_time:.2f}s")
            logger.info(f"Detected {total_faces_detected} faces")
            logger.info(f"Unique faces retained: {len(unique_faces)} (removed {duplicates_removed_count} duplicates)")
            if images_skipped_limit > 0:
                logger.info(f"Skipped {images_skipped_limit} face images due to limit ({getattr(config, 'MAX_IMAGES_PER_FACE', 5)} per unique face)")
            
            if not unique_faces:
                result = {
                    "status": "completed",
                    "video_id": video_id,
                    "total_frames_processed": frames_processed,
                    "faces_detected": 0,
                    "unique_faces_after_dedup": 0,
                    "matched_persons": [],
                    "processing_time_seconds": round(time.time() - start_time, 2)
                }
                config.job_store[video_id]["result"] = result
                config.job_store[video_id]["status"] = "completed"
                return

            # ============================================
            # STEP 3: Match unique faces against database (BATCH OPTIMIZED)
            # ============================================
            matching_start = time.time()
            logger.info(f"STEP 3: Matching {len(unique_faces)} unique faces against database (batch mode)")
            
            # Prepare batch data for vectorized matching
            if len(unique_faces) > 0:
                # Stack all embeddings at once
                query_embeddings = np.vstack([f['embedding'] for f in unique_faces])
                face_qualities = np.array([f['quality_score'] for f in unique_faces])
                
                # Batch match all faces at once (MUCH faster!)
                match_results = self.face_recognizer.match_faces_batch(
                    query_embeddings,
                    db_data,
                    face_qualities=face_qualities,
                    use_adaptive_threshold=config.USE_ADAPTIVE_THRESHOLD
                )
            else:
                match_results = []
            
            # Process batch results
            matched_persons_map = defaultdict(lambda: {
                "confidences": [],
                "count": 0,
                "best_quality": 0.0,
                "best_face_image": None,
                "frames": []
            })
            
            matched_count = 0
            
            for face_data, (person_id, confidence) in zip(unique_faces, match_results):
                if person_id:
                    matched_count += 1
                    pm = matched_persons_map[person_id]
                    pm["confidences"].append(confidence)
                    pm["count"] += 1
                    pm["frames"].append(face_data["frame_id"])
                    if face_data["quality_score"] > pm["best_quality"]:
                        pm["best_quality"] = face_data["quality_score"]
                        pm["best_face_image"] = face_data.get("image")
                    else:
                        pm["best_quality"] = max(pm["best_quality"], face_data["quality_score"])

            # ============================================
            # STEP 4: Compile Results
            # ============================================
            matching_time = time.time() - matching_start
            logger.info(f"Matched {matched_count} faces to {len(matched_persons_map)} persons in {matching_time:.2f}s")
            
            final_results = []
            for pid, data in matched_persons_map.items():
                p_info = self.face_database.get_person_info(pid)
                name = p_info["name"] if p_info else pid
                
                avg_conf = sum(data["confidences"]) / len(data["confidences"])
                
                crop_bytes = None
                best_img = data.get("best_face_image")
                if best_img is not None:
                    try:
                        _, jpg_bytes = cv2.imencode(".jpg", best_img)
                        if jpg_bytes is not None:
                            crop_bytes = jpg_bytes.tobytes()
                    except Exception:
                        pass
                
                final_results.append({
                    "person_id": pid,
                    "name": name,
                    "confidence": round(avg_conf, 3),
                    "total_appearances": data["count"],
                    "best_face_quality": round(data["best_quality"], 3),
                    "frames_seen": sorted(data["frames"]),
                    "crop_bytes": crop_bytes,
                })
            
            final_results.sort(key=lambda x: -x["confidence"])
            
            total_time = time.time() - start_time
            # Serializable list of embeddings for report flow (S3 detected_embeddings.npy)
            detected_embeddings_serializable = [f["embedding"].tolist() for f in unique_faces]
            result = {
                "status": "completed",
                "video_id": video_id,
                "total_frames_processed": frames_processed,
                "faces_detected": total_faces_detected,
                "unique_faces_after_dedup": len(unique_faces),
                "duplicates_removed": duplicates_removed_count,
                "matched_persons": final_results,
                "detected_embeddings": detected_embeddings_serializable,
                "processing_time_seconds": round(total_time, 2),
                "processing_stages": {
                    "stream_processing": "completed",
                    "vector_matching": "completed"
                },
                "performance_metrics": {
                    "extraction_time_seconds": round(extraction_time, 2) if extraction_start else None,
                    "matching_time_seconds": round(matching_time, 2) if matching_start else None,
                    "frames_per_second": round(frames_processed / total_time, 2) if total_time > 0 else 0
                }
            }
            
            # Save to database if available
            try:
                if config.db:
                    from database.connection import Database
                    from services.video_service import VideoService
                    from database.models import JobStatus
                    
                    with config.db.get_session() as db:
                        # Prepare matched persons data
                        matched_persons_data = []
                        for person_result in final_results:
                            matched_persons_data.append({
                                "person_id": person_result["person_id"],
                                "confidence": person_result["confidence"],
                                "total_appearances": person_result["total_appearances"],
                                "frames_seen": person_result["frames_seen"],
                                "best_face_quality": person_result["best_face_quality"]
                            })
                        
                        # Save results to database
                        VideoService.save_results(
                            db=db,
                            video_id=video_id,
                            result=result,
                            matched_persons=matched_persons_data
                        )
                        logger.info(f"Results saved to database for video: {video_id}")
            except Exception as db_error:
                logger.error(f"Failed to save results to database: {db_error}", exc_info=True)
                # Fallback to in-memory store
                if video_id in config.job_store:
                    config.job_store[video_id]["result"] = result
                    config.job_store[video_id]["status"] = "completed"
            else:
                # Update in-memory store for backward compatibility
                if video_id in config.job_store:
                    config.job_store[video_id]["result"] = result
                    config.job_store[video_id]["status"] = "completed"
            
        except Exception as e:
            # Save error to database if available
            try:
                if config.db:
                    from services.video_service import VideoService
                    from database.models import JobStatus
                    
                    with config.db.get_session() as db:
                        VideoService.update_job_status(
                            db=db,
                            video_id=video_id,
                            status=JobStatus.FAILED,
                            error_message=str(e)
                        )
            except Exception:
                pass
            
            # Fallback to in-memory store
            if video_id in config.job_store:
                config.job_store[video_id]["status"] = "failed"
                config.job_store[video_id]["error"] = str(e)
                config.job_store[video_id]["failed_at"] = time.time()
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass