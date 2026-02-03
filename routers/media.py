"""
Media Upload APIs.
"""
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import uuid
import os
import tempfile

from database.connection import Database
from database.models import User, UserRole, Report, ReportMedia
from auth.dependencies import get_current_user
from services.audit_service import AuditService
from storage.s3_client import S3Client
from storage.s3_paths import get_college_code_from_user, get_report_media_s3_path
from core.logger import logger
import config


router = APIRouter(prefix="/api", tags=["media"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


class UploadResponse(BaseModel):
    """Upload response."""
    url: str


class MediaResponse(BaseModel):
    """Media response."""
    media: List[dict]


@router.post("/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db_session)
):
    """
    Upload report media (image/video).
    Faculty and Student only.
    """
    if current_user.role not in [UserRole.FACULTY, UserRole.STUDENT]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only faculty and students can upload media"
        )
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    is_image = file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    if not (is_video or is_image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only images and videos are allowed."
        )
    
    # Validate file size (max 100MB)
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds 100MB limit"
        )
    
    # Upload to S3
    college_code = get_college_code_from_user(current_user)
    media_id = str(uuid.uuid4())
    
    # Use a temporary report ID for standalone uploads
    temp_report_id = 0
    s3_key = get_report_media_s3_path(college_code, temp_report_id, media_id, file_ext)
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Upload to S3
        if config.s3_client:
            file_url = config.s3_client.upload_file(
                tmp_path,
                s3_key,
                college_code=college_code,
                content_type=file.content_type or "application/octet-stream"
            )
        elif getattr(config, "USE_S3_ONLY", True):
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
            )
        else:
            local_dir = config.UPLOADS_DIR / "media" / "standalone"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"{media_id}{file_ext}"
            with open(local_path, "wb") as f:
                f.write(content)
            file_url = str(local_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Log action
        AuditService.log_from_request(
            db=db,
            request=request,
            action="media_upload",
            user_id=current_user.id,
            resource_type="media"
        )
        
        return UploadResponse(url=file_url)
        
    except Exception as e:
        logger.error(f"Failed to upload media file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload media file: {str(e)}"
        )


@router.get("/reports/{report_id}/media", response_model=MediaResponse)
async def get_report_media(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get report media URLs.
    Management, Master, and report owner (faculty/student) can access.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check authorization
    if current_user.role == UserRole.MASTER_ADMIN:
        pass  # Can access all
    elif current_user.role == UserRole.MANAGEMENT:
        if report.college_id != current_user.managed_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    elif current_user.role in [UserRole.FACULTY, UserRole.STUDENT]:
        if report.reporter_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get media records
    media_records = db.query(ReportMedia).filter(ReportMedia.report_id == report_id).all()
    
    media_list = []
    for media in media_records:
        media_list.append({
            "url": media.file_url,
            "type": media.media_type.value
        })
    
    return MediaResponse(media=media_list)
