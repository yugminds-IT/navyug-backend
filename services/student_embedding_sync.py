"""
Auto-sync student face embeddings when passport or 360° photos are uploaded.
Runs in background so face_embedding.npy is created/updated without app restart.
"""
import numpy as np
import cv2
from typing import List

from core.logger import logger
from core.face_detector import FaceDetector
from services.person_service import PersonService

import config


def _ensure_face_detector():
    """Ensure config.face_db has a face_detector (lazy init for embedding extraction)."""
    if not getattr(config, "face_db", None):
        return None
    if config.face_db.face_detector is None:
        try:
            config.face_db.face_detector = FaceDetector(
                ctx_id=config.CTX_ID,
                det_size=getattr(config, "DETECTION_SIZE", (640, 640)),
                det_thresh=getattr(config, "DETECTION_THRESHOLD", 0.5),
                recognition=True,
            )
        except Exception as e:
            logger.warning(f"Could not create FaceDetector for student embedding sync: {e}")
            return None
    return config.face_db.face_detector


def run_student_embedding_sync(
    person_id: str,
    name: str,
    college_code: str,
    image_bytes_list: List[bytes],
) -> bool:
    """
    Extract face embeddings from uploaded image(s), update in-memory face DB,
    and upload embeddings/face_embedding.npy to S3. Call from a background thread
    after passport or face-gallery upload so matching works without app restart.

    Args:
        person_id: Student identifier (roll_number, college_student_id, or STU{id})
        name: Display name
        college_code: College code for S3 path
        image_bytes_list: List of image bytes (e.g. from passport + face-gallery)

    Returns:
        True if at least one embedding was extracted and synced to S3
    """
    if not image_bytes_list or not college_code or not person_id:
        logger.debug("student_embedding_sync: missing person_id, college_code, or images")
        return False
    face_db = getattr(config, "face_db", None)
    if not face_db:
        logger.warning("student_embedding_sync: face_db not initialized")
        return False
    detector = _ensure_face_detector()
    if not detector:
        logger.warning("student_embedding_sync: face_detector not available")
        return False

    all_embeddings: List[np.ndarray] = []
    for idx, raw in enumerate(image_bytes_list):
        if not raw:
            continue
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.debug(f"student_embedding_sync: could not decode image {idx + 1}")
            continue
        try:
            faces = detector.detect_and_embed(
                img,
                preprocess=getattr(config, "ENABLE_IMAGE_PREPROCESSING", True),
                align_faces=getattr(config, "ENABLE_FACE_ALIGNMENT", True),
            )
        except Exception as e:
            logger.warning(f"student_embedding_sync: detect_and_embed failed for image {idx + 1}: {e}")
            continue
        faces = [f for f in faces if f.get("confidence", 0) >= getattr(config, "FACE_DETECTION_CONFIDENCE", 0.5)]
        if getattr(config, "MIN_FACE_SIZE", 0) > 0:
            faces = [
                f for f in faces
                if (f["bbox"][2] - f["bbox"][0]) >= config.MIN_FACE_SIZE
                and (f["bbox"][3] - f["bbox"][1]) >= config.MIN_FACE_SIZE
            ]
        for f in faces:
            emb = f.get("embedding")
            if emb is not None:
                all_embeddings.append(np.asarray(emb))

    if not all_embeddings:
        logger.info(f"student_embedding_sync: no faces extracted for person_id={person_id}")
        return False

    try:
        count = face_db.add_embeddings_to_person(person_id, name, all_embeddings)
        logger.info(f"student_embedding_sync: added {len(all_embeddings)} embedding(s) for {person_id}, total={count}")
    except Exception as e:
        logger.error(f"student_embedding_sync: add_embeddings_to_person failed: {e}", exc_info=True)
        return False

    # Get merged list for S3 (same as get_all_embeddings)
    merged = face_db._get_embeddings_list(face_db.persons.get(person_id, {}))
    if not merged:
        return False
    ok = PersonService.upload_embeddings_to_s3(person_id, merged, college_code)
    if ok:
        logger.info(f"student_embedding_sync: uploaded face_embedding.npy for {person_id}")
    else:
        logger.warning(f"student_embedding_sync: failed to upload face_embedding.npy for {person_id}")
    return ok
