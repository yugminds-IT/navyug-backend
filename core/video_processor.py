import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from core.frame_extractor import FrameExtractor
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.face_database import FaceDatabase
from core.utils import apply_nms
from core.image_quality import detect_duplicate_images, calculate_image_quality_score
from core.logger import logger
import config


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
    
    def process_video_sync(self, video_id: str, video_path: str):
        """
        Process video with optimized workflow:
        1. Stream video processing (Video -> Faces)
        2. Incremental deduplication (Keep best quality unique faces)
        3. Vectorized matching against database
        4. Return identification results
        """
        try:
            # Update status to processing
            processing_start_time = time.time()
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
            
            # Create debug directory for this video
            if config.SAVE_DETECTED_FACES:
                debug_video_dir = config.DEBUG_FACES_DIR / video_id
                debug_video_dir.mkdir(exist_ok=True)
            
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
            
            total_faces_detected = 0
            frames_processed = 0
            duplicates_removed_count = 0
            
            with FrameExtractor(video_path, frames_per_second=config.FRAME_EXTRACTION_RATE) as extractor:
                video_info = extractor.get_video_info()
                total_frames = video_info["frames_to_extract"]
                config.job_store[video_id]["progress"]["total_frames"] = total_frames
                
                for frame_data in extractor.extract_frames():
                    frame = frame_data["frame"]
                    frame_id = frame_data["frame_id"]
                    timestamp = frame_data["timestamp"]
                    
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
                    
                    # INCREMENTAL DEDUPLICATION
                    for face in faces:
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
                            # New unique face
                            unique_faces.append(face_data)
                            unique_embeddings_list.append(face["embedding"])
                            # Invalidate cache since matrix changed
                            if hasattr(self, '_cached_unique_matrix'):
                                del self._cached_unique_matrix
                    
                    frames_processed += 1
                    
            # Update progress less frequently for better performance (every 10 frames instead of 5)
            if frames_processed % 10 == 0:
                percentage = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                config.job_store[video_id]["progress"].update({
                    "frames_processed": frames_processed,
                    "total_frames": total_frames,
                    "percentage": round(percentage, 2),
                    "stage": "processing_stream",
                    "faces_collected": total_faces_detected,
                    "unique_faces": len(unique_faces),
                    "updated_at": time.time()
                })
            
            extraction_time = time.time() - extraction_start
            logger.info(f"Processed {frames_processed} frames in {extraction_time:.2f}s")
            logger.info(f"Detected {total_faces_detected} faces")
            logger.info(f"Unique faces retained: {len(unique_faces)} (removed {duplicates_removed_count} duplicates)")
            
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

            # Save debug images (final unique set) - OPTIMIZED: batch write
            if config.SAVE_DETECTED_FACES:
                # Limit number of saved faces for performance
                faces_to_save = unique_faces[:config.MAX_DEBUG_FACES_PER_VIDEO] if hasattr(config, 'MAX_DEBUG_FACES_PER_VIDEO') else unique_faces
                
                for idx, face_data in enumerate(faces_to_save):
                    timestamp_str = f"{face_data['timestamp']:.2f}".replace('.', '_')
                    fname = f"face_{idx:03d}_frame{face_data['frame_id']}_t{timestamp_str}.jpg"
                    fpath = debug_video_dir / fname
                    # Use lower quality JPEG for faster writes (quality=85 instead of default 95)
                    cv2.imwrite(str(fpath), face_data['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    face_data['debug_file_path'] = fpath

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
                    pm["best_quality"] = max(pm["best_quality"], face_data["quality_score"])
                    
                    # Rename debug file if exists
                    if 'debug_file_path' in face_data:
                        old_p = face_data['debug_file_path']
                        if old_p.exists():
                            new_name = f"{person_id}_{confidence:.2f}_{old_p.name}"
                            try:
                                old_p.rename(old_p.parent / new_name)
                                logger.debug(f"Renamed debug file: {old_p.name} -> {new_name}")
                            except OSError as e:
                                logger.warning(f"Could not rename debug file {old_p.name}: {e}")
                            except Exception as e:
                                logger.warning(f"Unexpected error renaming debug file {old_p.name}: {e}", exc_info=True)

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
                
                final_results.append({
                    "person_id": pid,
                    "name": name,
                    "confidence": round(avg_conf, 3),
                    "total_appearances": data["count"],
                    "best_face_quality": round(data["best_quality"], 3),
                    # We only have aggregated frame info now because of deduplication
                    "frames_seen": sorted(data["frames"])
                })
            
            final_results.sort(key=lambda x: -x["confidence"])
            
            total_time = time.time() - start_time
            result = {
                "status": "completed",
                "video_id": video_id,
                "total_frames_processed": frames_processed,
                "faces_detected": total_faces_detected,
                "unique_faces_after_dedup": len(unique_faces),
                "duplicates_removed": duplicates_removed_count,
                "matched_persons": final_results,
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
            
            config.job_store[video_id]["result"] = result
            config.job_store[video_id]["status"] = "completed"
            
        except Exception as e:
            config.job_store[video_id]["status"] = "failed"
            config.job_store[video_id]["error"] = str(e)
            config.job_store[video_id]["failed_at"] = time.time()
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
