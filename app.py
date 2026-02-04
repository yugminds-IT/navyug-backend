"""
Production-ready Face Recognition API with PostgreSQL, S3, and security features.
"""
import asyncio
import uuid
import shutil
from pathlib import Path
from typing import Optional
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi_mail import FastMail, ConnectionConfig

import config
from database.connection import Database
from database.models import User, UserRole
from sqlalchemy import or_, text
from storage.s3_client import S3Client
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
from auth.dependencies import get_current_user_or_api_key, require_user
from database.connection import Database
from middleware.security import (
    RateLimitMiddleware, SecurityHeadersMiddleware,
    setup_cors, setup_trusted_hosts
)
from middleware.auth_middleware import AuthRequiredMiddleware
from services.video_service import VideoService
from services.person_service import PersonService
from services.audit_service import AuditService
from routers.auth import router as auth_router
from routers.users import router as users_router
from routers.profile import router as profile_router
from routers.reports import router as reports_router
from routers.colleges import router as colleges_router
from routers.students import router as students_router
from routers.faculty import router as faculty_router
from routers.media import router as media_router
from routers.logs import router as logs_router
from routers.dashboards import router as dashboards_router


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app.
    Initialize database, S3, and load face database on startup.
    """
    logger.info("=" * 60)
    logger.info("Starting Face Recognition System (Production Mode)...")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        config.db = Database(
            database_url=config.DATABASE_URL,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW
        )
        # Create tables if they don't exist
        config.db.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise
    
    # Initialize S3 client if enabled
    if config.USE_S3:
        try:
            default_bucket = (
                config.CAMPUS_SECURITY_BUCKET_NAME
                if getattr(config, "USE_CAMPUS_SECURITY_BUCKET", False)
                else config.S3_BUCKET_NAME
            )
            config.s3_client = S3Client(
                default_bucket_name=default_bucket,
                aws_access_key_id=config.S3_ACCESS_KEY_ID,
                aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
                region_name=config.S3_REGION,
                endpoint_url=config.S3_ENDPOINT_URL,
                auto_create_buckets=True,
            )
            if getattr(config, "USE_CAMPUS_SECURITY_BUCKET", False):
                logger.info(f"S3 client initialized (campus-security bucket: {default_bucket})")
            else:
                logger.info("S3 client initialized successfully (college-specific buckets enabled)")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}", exc_info=True)
            logger.warning("Continuing without S3 - files will be stored locally")
            config.s3_client = None
    else:
        logger.info("S3 storage disabled - using local storage")
        config.s3_client = None
    
    # Initialize FastAPI-Mail for OTP and notifications
    if config.SMTP_USER and config.SMTP_PASSWORD:
        try:
            mail_conf = ConnectionConfig(
                MAIL_USERNAME=config.SMTP_USER,
                MAIL_PASSWORD=config.SMTP_PASSWORD,
                MAIL_FROM=config.SMTP_FROM_EMAIL or config.SMTP_USER,
                MAIL_FROM_NAME=config.SMTP_FROM_NAME,
                MAIL_PORT=config.SMTP_PORT,
                MAIL_SERVER=config.SMTP_HOST,
                MAIL_STARTTLS=config.SMTP_USE_TLS,
                MAIL_SSL_TLS=config.SMTP_USE_SSL,
                USE_CREDENTIALS=True,
                VALIDATE_CERTS=True,
            )
            app.state.mail = FastMail(mail_conf)
            logger.info("FastAPI-Mail initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FastAPI-Mail: {e}", exc_info=True)
            app.state.mail = None
    else:
        logger.warning("SMTP credentials not set (SMTP_USER/SMTP_PASSWORD). OTP emails will not be sent.")
        app.state.mail = None
    
    # Initialize face database (empty); load in background so app starts fast for Coolify health check
    config.face_db = FaceDatabase(similarity_threshold=config.FACE_MATCH_THRESHOLD)

    def _load_face_db_sync():
        """Load face DB from S3/DB/directory (runs in thread so startup is fast)."""
        try:
            with config.db.get_session() as db:
                if getattr(config, "USE_S3_ONLY", True) and config.s3_client:
                    try:
                        from database.models import College
                        colleges = db.query(College).all()
                        for college in colleges:
                            if college.college_code:
                                loaded = config.face_db.load_from_s3(
                                    college.college_code,
                                    s3_client=config.s3_client,
                                    ctx_id=config.CTX_ID,
                                )
                                if loaded:
                                    logger.info(f"Loaded {loaded} persons from S3 for college {college.college_code}")
                        if config.face_db.get_database_size() > 0:
                            logger.info("Face DB populated from S3")
                    except Exception as s3_err:
                        logger.warning(f"load_from_s3 failed: {s3_err}")
                if config.face_db.get_database_size() == 0:
                    persons = PersonService.get_all_persons(db, active_only=True)
                    if persons:
                        for person in persons:
                            embeddings = PersonService.get_embeddings(db, person.person_id)
                            if embeddings:
                                for emb in embeddings:
                                    config.face_db.add_person(person.person_id, person.name, emb)
                        logger.info(f"Loaded {len(persons)} persons from database")
                    elif not getattr(config, "USE_S3_ONLY", True) and config.COLLEGE_FACES_DIR.exists():
                        loaded = config.face_db.load_from_directory(
                            str(config.COLLEGE_FACES_DIR), ctx_id=config.CTX_ID
                        )
                        logger.info(f"Loaded {loaded} persons from directory")
                if config.s3_client and config.face_db.get_database_size() > 0:
                    all_emb = config.face_db.get_all_embeddings()
                    for person_id, emb_list in all_emb.items():
                        if not emb_list:
                            continue
                        student = db.query(User).filter(
                            User.role == UserRole.STUDENT,
                            or_(
                                User.roll_number == person_id,
                                User.login_id == person_id,
                                User.college_student_id == person_id,
                            ),
                        ).first()
                        if student and student.college:
                            PersonService.upload_embeddings_to_s3(
                                person_id, emb_list, student.college.college_code
                            )
        except Exception as e:
            logger.error(f"Error loading face database: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("Server ready! (face DB loading in background)")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"API Docs: http://localhost:8000/docs")
    logger.info("=" * 60)

    # Start face DB load in background so Coolify health check passes quickly
    asyncio.create_task(asyncio.to_thread(_load_face_db_sync))

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    if config.db:
        config.db.engine.dispose()
        logger.info("Database connections closed")


# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description="Production-ready face recognition API with PostgreSQL, S3, and security features",
    version=config.APP_VERSION,
    lifespan=lifespan
)

# Setup security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=config.RATE_LIMIT_PER_MINUTE,
    requests_per_hour=config.RATE_LIMIT_PER_HOUR
)
# Add authentication middleware (enforces auth on all routes except public ones)
app.add_middleware(AuthRequiredMiddleware)
setup_cors(app, config.CORS_ORIGINS)
if config.ENVIRONMENT == "production":
    setup_trusted_hosts(app, config.TRUSTED_HOSTS)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(profile_router)
app.include_router(reports_router)
app.include_router(colleges_router)
app.include_router(students_router)
app.include_router(faculty_router)
app.include_router(media_router)
app.include_router(logs_router)
app.include_router(dashboards_router)


@app.get("/")
async def root():
    """Root endpoint with API information. Public endpoint."""
    db_size = config.face_db.get_database_size() if config.face_db else 0
    return {
        "message": config.APP_NAME,
        "version": config.APP_VERSION,
        "environment": config.ENVIRONMENT,
        "endpoints": {
            "upload": "POST /incident/upload-video",
            "status": "GET /incident/status/{video_id}",
            "results": "GET /incident/results/{video_id}"
        },
        "docs": "/docs",
        "database_size": db_size,
        "s3_enabled": config.USE_S3 and config.s3_client is not None
    }


@app.post("/incident/upload-video")
def get_db():
    """Database dependency."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


async def upload_video(
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_user_or_api_key),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Upload incident video for face recognition processing.
    
    Requires authentication. Returns immediately with video_id. Processing happens in background.
    Use /incident/status/{video_id} to check progress.
    """
    logger.info(f"Received video upload request from user {current_user.id}: {video.filename}")
    
    # Validate filename
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Sanitize filename
    try:
        safe_filename = sanitize_filename(video.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")
    
    # Validate file extension
    if not validate_video_extension(safe_filename, config.ALLOWED_VIDEO_EXTENSIONS):
        file_ext = Path(safe_filename).suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(config.ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check file size
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)
    
    max_size_bytes = config.MAX_VIDEO_SIZE_MB * 1024 * 1024
    is_valid_size, size_error = validate_file_size(file_size, max_size_bytes)
    if not is_valid_size:
        raise HTTPException(status_code=400, detail=size_error)
    
    # Check if face database is loaded
    if not config.face_db or config.face_db.get_database_size() == 0:
        raise HTTPException(
            status_code=503,
            detail="Face database not loaded. Please add college faces to the database."
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    file_ext = Path(safe_filename).suffix.lower()
    
    # Save video file locally first
    local_video_path = config.UPLOADS_DIR / f"{video_id}{file_ext}"
    s3_video_path = None
    
    try:
        logger.info(f"Saving video file: {local_video_path} ({file_size / (1024*1024):.2f} MB)")
        with open(local_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Upload to S3 if enabled
        if config.s3_client:
            try:
                from storage.s3_paths import get_college_code_from_user, get_video_s3_path
                college_code = get_college_code_from_user(current_user)
                s3_key = get_video_s3_path(college_code, video_id, file_ext)
                # Store full S3 path including bucket for reference
                s3_full_path = config.s3_client.upload_file(
                    str(local_video_path),
                    s3_key,
                    college_code=college_code,
                    content_type=video.content_type or "video/mp4"
                )
                s3_video_path = s3_full_path  # Full S3 URL: s3://bucket/key
                logger.info(f"Video uploaded to S3: {s3_full_path}")
                # Optionally delete local file after S3 upload
                # local_video_path.unlink()
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}", exc_info=True)
                # Continue with local storage
        
        logger.info(f"Video file saved successfully: {video_id}")
    except Exception as e:
        logger.error(f"Failed to save video file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Validate the saved video file
    is_valid, validation_error = validate_video_file(local_video_path)
    if not is_valid:
        logger.error(f"Video file validation failed: {validation_error}")
        try:
            local_video_path.unlink()
            if s3_video_path and config.s3_client:
                config.s3_client.delete_file(s3_video_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Invalid video file: {validation_error}")
    
    # Create database job
    try:
        job = VideoService.create_job(
            db=db,
            video_id=video_id,
            filename=safe_filename,
            user_id=current_user.id,
            s3_video_path=s3_video_path,
            local_video_path=str(local_video_path),
            file_size_mb=file_size / (1024 * 1024)
        )
        
        # Audit log
        AuditService.log_from_request(
            db=db,
            request=request,
            action="video_upload",
            user_id=current_user.id,
            resource_type="video_job",
            resource_id=video_id
        )
    except Exception as e:
        logger.error(f"Failed to create database job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create job")
    
    # Start async processing
    def run_processing(vid_id, vid_path):
        proc = VideoProcessor(config.face_db)
        proc.process_video_sync(vid_id, vid_path)
    
    asyncio.create_task(asyncio.to_thread(run_processing, video_id, str(local_video_path)))
    logger.info(f"Video processing started for: {video_id}")
    
    return {
        "status": "success",
        "video_id": video_id,
        "message": "Video uploaded successfully. Processing started.",
        "filename": safe_filename,
        "size_mb": round(file_size / (1024 * 1024), 2)
    }


@app.get("/incident/status/{video_id}")
async def get_status(
    video_id: str,
    current_user: User = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db)
):
    """Get processing status for a video."""
    if not validate_uuid(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    job = VideoService.get_job(db, video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    # Check authorization (users can only see their own jobs, admins can see all)
    if current_user.role.value != "admin" and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    current_time = time.time()
    response = {
        "video_id": video_id,
        "status": job.status.value,
        "filename": job.filename,
        "timestamp": current_time,
        "uploaded_at": job.uploaded_at.timestamp() if job.uploaded_at else current_time
    }
    
    if job.progress_data:
        response["progress"] = job.progress_data
        response["progress"]["percentage"] = job.progress_percentage
    
    if job.status.value == "completed" and job.result:
        response["summary"] = {
            "total_frames_processed": job.result.get("total_frames_processed", 0),
            "faces_detected": job.result.get("faces_detected", 0),
            "matched_persons_count": len(job.result.get("matched_persons", []))
        }
    
    if job.error_message:
        response["error"] = job.error_message
    
    return response


@app.get("/incident/results/{video_id}")
async def get_results(
    video_id: str,
    current_user: User = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db)
):
    """Get face recognition results for a completed video."""
    if not validate_uuid(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    job = VideoService.get_job(db, video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    # Check authorization
    if current_user.role.value != "admin" and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status.value == "pending":
        raise HTTPException(status_code=202, detail="Video is pending processing")
    
    if job.status.value == "processing":
        raise HTTPException(status_code=202, detail="Video is still processing")
    
    if job.status.value == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {job.error_message}")
    
    if not job.result:
        raise HTTPException(status_code=404, detail="Results not available")
    
    return job.result


@app.delete("/incident/{video_id}")
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user_or_api_key),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Delete video and its processing results."""
    if not validate_uuid(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    job = VideoService.get_job(db, video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    # Check authorization
    if current_user.role.value != "admin" and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete video file
    if job.local_video_path and Path(job.local_video_path).exists():
        try:
            Path(job.local_video_path).unlink()
        except Exception as e:
            logger.error(f"Failed to delete local video: {e}")
    
    if job.s3_video_path and config.s3_client:
        try:
            from storage.s3_paths import get_college_code_from_user
            college_code = get_college_code_from_user(current_user)
            # Extract s3_key from full path (s3://bucket/key -> key)
            if job.s3_video_path.startswith("s3://"):
                s3_key = job.s3_video_path.split("/", 3)[-1]  # Get key part after bucket
            else:
                s3_key = job.s3_video_path
            config.s3_client.delete_file(s3_key, college_code=college_code)
        except Exception as e:
            logger.error(f"Failed to delete S3 video: {e}")
    
    # Delete from database
    VideoService.delete_job(db, video_id)
    
    # Audit log
    AuditService.log_from_request(
        db=db,
        request=request,
        action="video_delete",
        user_id=current_user.id,
        resource_type="video_job",
        resource_id=video_id
    )
    
    return {"status": "success", "message": f"Video {video_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring. Public endpoint."""
    import shutil
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Check database
    try:
        if config.db is None:
            health_status["checks"]["database"] = {"status": "error", "error": "not initialized"}
            health_status["status"] = "degraded"
        else:
            with config.db.get_session() as db:
                db.execute(text("SELECT 1"))
            health_status["checks"]["database"] = {"status": "ok"}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check S3 (check default bucket) — uploads go here when USE_S3=true
    if config.s3_client:
        try:
            default_bucket = config.s3_client.default_bucket_name or "default"
            config.s3_client.s3_client.head_bucket(Bucket=default_bucket)
            health_status["checks"]["s3"] = {
                "status": "ok",
                "enabled": True,
                "bucket": default_bucket,
                "note": "Uploads (passport, face-gallery) go to this bucket when USE_S3=true",
            }
        except Exception as e:
            health_status["checks"]["s3"] = {"status": "error", "enabled": True, "bucket": getattr(config.s3_client, "default_bucket_name", None), "error": str(e)}
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["s3"] = {
            "status": "disabled",
            "enabled": False,
            "bucket": None,
            "message": "USE_S3 is false or S3 client failed to init — uploads are stored locally; bucket will stay empty.",
        }
    
    # Check face database
    try:
        db_loaded = config.face_db is not None and config.face_db.get_database_size() > 0
        health_status["checks"]["face_database"] = {
            "loaded": db_loaded,
            "size": config.face_db.get_database_size() if config.face_db else 0
        }
    except Exception:
        health_status["checks"]["face_database"] = {"loaded": False, "size": 0}
    if not db_loaded:
        health_status["status"] = "degraded"
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage(config.UPLOADS_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        health_status["checks"]["disk"] = {
            "free_gb": round(free_gb, 2),
            "percent_free": round((disk_usage.free / disk_usage.total) * 100, 2)
        }
        if free_gb < 1:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["disk"] = {"error": str(e)}
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
