import asyncio
import uuid
import shutil
from pathlib import Path
from typing import Optional
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

import config
from core.face_database import FaceDatabase
from core.video_processor import VideoProcessor
from core.logger import logger
from core.validators import (
    sanitize_filename,
    validate_video_file,
    validate_video_extension,
    validate_file_size,
    validate_uuid
)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app.
    Load face database on startup.
    """
    logger.info("=" * 60)
    logger.info("Starting Face Recognition System...")
    logger.info("=" * 60)
    
    # Initialize and load face database
    config.face_db = FaceDatabase(similarity_threshold=config.FACE_MATCH_THRESHOLD)
    
    if config.COLLEGE_FACES_DIR.exists():
        logger.info(f"Loading college faces from: {config.COLLEGE_FACES_DIR}")
        try:
            loaded = config.face_db.load_from_directory(
                str(config.COLLEGE_FACES_DIR),
                ctx_id=config.CTX_ID
            )
            logger.info(f"Successfully loaded {loaded} persons into database")
        except Exception as e:
            logger.error(f"Error loading face database: {e}", exc_info=True)
            logger.warning("The system will start but won't be able to match faces.")
    else:
        logger.warning(f"College faces directory not found: {config.COLLEGE_FACES_DIR}")
        logger.warning("Create the directory and add person folders to enable face matching.")
    
    logger.info("=" * 60)
    logger.info("Server ready!")
    logger.info(f"API Docs: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Incident Video Face Recognition API",
    description="Upload incident videos and match faces against college database",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Incident Video Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /incident/upload-video",
            "status": "GET /incident/status/{video_id}",
            "results": "GET /incident/results/{video_id}"
        },
        "docs": "/docs",
        "database_size": config.face_db.get_database_size() if config.face_db else 0
    }


@app.post("/incident/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """
    Upload incident video for face recognition processing.
    
    Returns immediately with video_id. Processing happens in background.
    Use /incident/status/{video_id} to check progress.
    
    Args:
        video: Video file (MP4, AVI, MOV, MKV, FLV)
            - Max size: 500MB
            - Must contain valid video data
            
    Returns:
        JSON with video_id for status tracking
        
    Raises:
        400: Invalid file type, size, or format
        503: Database not loaded
        500: File save error
    """
    logger.info(f"Received video upload request: {video.filename}")
    
    # Validate filename
    if not video.filename:
        logger.warning("Upload attempt with no filename")
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Sanitize filename to prevent path traversal
    try:
        safe_filename = sanitize_filename(video.filename)
        logger.debug(f"Sanitized filename: {video.filename} -> {safe_filename}")
    except ValueError as e:
        logger.warning(f"Invalid filename: {video.filename} - {e}")
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")
    
    # Validate file extension
    if not validate_video_extension(safe_filename, config.ALLOWED_VIDEO_EXTENSIONS):
        file_ext = Path(safe_filename).suffix.lower()
        logger.warning(f"Invalid file extension: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(config.ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check file size
    video.file.seek(0, 2)  # Seek to end
    file_size = video.file.tell()
    video.file.seek(0)  # Reset to beginning
    
    max_size_bytes = config.MAX_VIDEO_SIZE_MB * 1024 * 1024
    is_valid_size, size_error = validate_file_size(file_size, max_size_bytes)
    if not is_valid_size:
        logger.warning(f"File size validation failed: {size_error}")
        raise HTTPException(status_code=400, detail=size_error)
    
    # Check if face database is loaded
    if not config.face_db or config.face_db.get_database_size() == 0:
        logger.error("Video upload attempted but face database is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Face database not loaded. Please add college faces to the database."
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    file_ext = Path(safe_filename).suffix.lower()
    
    # Save video file
    video_path = config.UPLOADS_DIR / f"{video_id}{file_ext}"
    
    try:
        logger.info(f"Saving video file: {video_path} ({file_size / (1024*1024):.2f} MB)")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"Video file saved successfully: {video_id}")
    except Exception as e:
        logger.error(f"Failed to save video file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Validate the saved video file
    is_valid, validation_error = validate_video_file(video_path)
    if not is_valid:
        logger.error(f"Video file validation failed: {validation_error}")
        # Clean up invalid file
        try:
            video_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Invalid video file: {validation_error}")
    
    # Initialize job status with comprehensive tracking
    config.job_store[video_id] = {
        "status": "pending",
        "video_path": str(video_path),
        "filename": safe_filename,
        "progress": None,
        "result": None,
        "error": None,
        "uploaded_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "failed_at": None
    }
    
    # Start async processing
    # Move initialization inside the thread to avoid blocking the event loop with model loading
    def run_processing(vid_id, vid_path):
        proc = VideoProcessor(config.face_db)
        proc.process_video_sync(vid_id, vid_path)
        
    asyncio.create_task(asyncio.to_thread(run_processing, video_id, str(video_path)))
    logger.info(f"Video processing started for: {video_id}")
    
    return {
        "status": "success",
        "video_id": video_id,
        "message": "Video uploaded successfully. Processing started.",
        "filename": safe_filename,
        "size_mb": round(file_size / (1024 * 1024), 2)
    }


@app.get("/incident/status/{video_id}")
async def get_status(video_id: str):
    """
    Get processing status for a video with detailed progress information.
    
    Status values:
    - pending: Waiting to start processing
    - processing: Currently processing video
    - completed: Processing finished successfully
    - failed: Processing failed with error
    
    Args:
        video_id: UUID of the video job
        
    Returns:
        JSON with comprehensive status and progress information including:
        - Current status
        - Progress percentage
        - Stage information
        - Elapsed time
        - Estimated time remaining
        - Processing statistics
    """
    # Validate UUID format
    if not validate_uuid(video_id):
        logger.warning(f"Invalid UUID format: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    # Check if job exists
    if video_id not in config.job_store:
        logger.warning(f"Video ID not found: {video_id}")
        raise HTTPException(
            status_code=404, 
            detail="Video ID not found. The job may not exist or may have been deleted."
        )
    
    job = config.job_store[video_id]
    current_time = time.time()
    
    # Build base response
    response = {
        "video_id": video_id,
        "status": job["status"],
        "filename": job.get("filename"),
        "timestamp": current_time,
        "uploaded_at": job.get("uploaded_at", current_time)
    }
    
    # Calculate elapsed time
    elapsed_time = current_time - job.get("uploaded_at", current_time)
    response["elapsed_time_seconds"] = round(elapsed_time, 2)
    
    # Handle different statuses
    if job["status"] == "pending":
        response["message"] = "Video uploaded and waiting to start processing"
        response["estimated_wait_time_seconds"] = 5  # Usually starts within 5 seconds
        
    elif job["status"] == "processing":
        progress = job.get("progress")
        if progress:
            # Calculate percentage
            frames_processed = progress.get("frames_processed", 0)
            total_frames = progress.get("total_frames", 0)
            
            if total_frames > 0:
                percentage = (frames_processed / total_frames) * 100
            else:
                percentage = 0.0
            
            # Enhanced progress information
            response["progress"] = {
                "percentage": round(percentage, 2),
                "frames_processed": frames_processed,
                "total_frames": total_frames,
                "stage": progress.get("stage", "processing"),
                "faces_collected": progress.get("faces_collected", 0),
                "unique_faces": progress.get("unique_faces", 0)
            }
            
            # Calculate estimated time remaining
            if frames_processed > 0 and elapsed_time > 0:
                # Calculate processing speed (frames per second)
                processing_speed = frames_processed / elapsed_time
                
                if processing_speed > 0 and total_frames > frames_processed:
                    remaining_frames = total_frames - frames_processed
                    estimated_remaining = remaining_frames / processing_speed
                    response["progress"]["estimated_time_remaining_seconds"] = round(estimated_remaining, 2)
                    response["progress"]["processing_speed_fps"] = round(processing_speed, 2)
                else:
                    response["progress"]["estimated_time_remaining_seconds"] = None
                    response["progress"]["processing_speed_fps"] = round(processing_speed, 2) if processing_speed > 0 else None
            else:
                response["progress"]["estimated_time_remaining_seconds"] = None
                response["progress"]["processing_speed_fps"] = None
            
            # Add stage-specific messages
            stage = progress.get("stage", "processing")
            stage_messages = {
                "initializing": "Initializing video processing...",
                "processing_stream": f"Processing frames: {frames_processed}/{total_frames}",
                "matching": "Matching faces against database...",
                "finalizing": "Finalizing results..."
            }
            response["message"] = stage_messages.get(stage, "Processing video...")
        else:
            response["progress"] = None
            response["message"] = "Processing started, progress information not yet available"
        
        # Add elapsed processing time
        response["processing_time_seconds"] = round(elapsed_time, 2)
        
    elif job["status"] == "completed":
        result = job.get("result")
        if result:
            # Include summary statistics
            response["summary"] = {
                "total_frames_processed": result.get("total_frames_processed", 0),
                "faces_detected": result.get("faces_detected", 0),
                "unique_faces_after_dedup": result.get("unique_faces_after_dedup", 0),
                "duplicates_removed": result.get("duplicates_removed", 0),
                "matched_persons_count": len(result.get("matched_persons", [])),
                "processing_time_seconds": result.get("processing_time_seconds", 0)
            }
            
            # Add performance metrics if available
            if "performance_metrics" in result:
                response["performance_metrics"] = result["performance_metrics"]
            
            response["message"] = "Processing completed successfully"
        else:
            response["message"] = "Processing completed (results available)"
            response["summary"] = None
        
        # Calculate total time
        if "uploaded_at" in job:
            total_time = current_time - job["uploaded_at"]
            response["total_time_seconds"] = round(total_time, 2)
        
    elif job["status"] == "failed":
        error = job.get("error", "Unknown error")
        response["error"] = error
        response["message"] = f"Processing failed: {error}"
        
        # Add failure timestamp if available
        if "failed_at" in job:
            response["failed_at"] = job["failed_at"]
        else:
            response["failed_at"] = current_time
        
        # Log the error for debugging
        logger.error(f"Job {video_id} failed: {error}")
    else:
        # Unknown status
        logger.warning(f"Unknown status '{job['status']}' for job {video_id}")
        response["message"] = f"Unknown status: {job['status']}"
        response["status"] = "unknown"
    
    return response


@app.get("/incident/results/{video_id}")
async def get_results(video_id: str):
    """
    Get face recognition results for a completed video.
    
    Returns matched persons with confidence scores.
    
    Args:
        video_id: UUID of the video job
        
    Returns:
        JSON with face recognition results
    """
    # Validate UUID format
    if not validate_uuid(video_id):
        logger.warning(f"Invalid UUID format: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    if video_id not in config.job_store:
        logger.warning(f"Video ID not found: {video_id}")
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    job = config.job_store[video_id]
    
    if job["status"] == "pending":
        raise HTTPException(
            status_code=202,
            detail="Video is pending processing. Check status endpoint."
        )
    
    if job["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail="Video is still processing. Check status endpoint for progress."
        )
    
    if job["status"] == "failed":
        error_msg = job.get('error', 'Unknown error')
        logger.warning(f"Results requested for failed job {video_id}: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {error_msg}"
        )
    
    logger.info(f"Returning results for video: {video_id}")
    # Return results
    return job["result"]


@app.delete("/incident/{video_id}")
async def delete_video(video_id: str):
    """
    Delete video and its processing results.
    
    Args:
        video_id: UUID of the video job to delete
    """
    # Validate UUID format
    if not validate_uuid(video_id):
        logger.warning(f"Invalid UUID format: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    if video_id not in config.job_store:
        logger.warning(f"Video ID not found for deletion: {video_id}")
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    job = config.job_store[video_id]
    
    # Delete video file
    video_path = Path(job["video_path"])
    if video_path.exists():
        try:
            video_path.unlink()
            logger.info(f"Deleted video file: {video_path}")
        except Exception as e:
            logger.error(f"Failed to delete video file {video_path}: {e}", exc_info=True)
            # Continue with job deletion even if file deletion fails
    
    # Remove from job store
    del config.job_store[video_id]
    logger.info(f"Deleted job: {video_id}")
    
    return {
        "status": "success",
        "message": f"Video {video_id} deleted successfully"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring system status.
    
    Returns:
        JSON with system health information
    """
    import shutil
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Check database
    db_loaded = config.face_db is not None and config.face_db.get_database_size() > 0
    health_status["checks"]["database"] = {
        "loaded": db_loaded,
        "size": config.face_db.get_database_size() if config.face_db else 0
    }
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage(config.UPLOADS_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        health_status["checks"]["disk"] = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "percent_free": round((disk_usage.free / disk_usage.total) * 100, 2)
        }
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        health_status["checks"]["disk"] = {"error": str(e)}
    
    # Check active jobs
    active_jobs = len([j for j in config.job_store.values() if j["status"] == "processing"])
    total_jobs = len(config.job_store)
    health_status["checks"]["jobs"] = {
        "active": active_jobs,
        "total": total_jobs
    }
    
    # Overall status
    if not db_loaded:
        health_status["status"] = "degraded"
        health_status["message"] = "Database not loaded"
    elif health_status["checks"].get("disk", {}).get("percent_free", 100) < 10:
        health_status["status"] = "degraded"
        health_status["message"] = "Low disk space"
    else:
        health_status["message"] = "All systems operational"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
