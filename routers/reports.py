"""
Incident Reports APIs.
"""
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, tuple_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from io import BytesIO
import uuid
import os

from database.connection import Database
from database.models import (
    User, UserRole, Report, ReportMedia, DetectedFace, DisciplinaryAction,
    IncidentType, ReportStatus, ReportReporterType, MediaType, ActionType,
    College, CollegeDepartment,
)
from auth.dependencies import get_current_user, require_master_admin, require_management
from services.audit_service import AuditService
from services.report_media_detection import run_report_media_face_detection, run_report_face_detection_from_storage
from storage.s3_client import S3Client
from storage.s3_paths import get_college_code_from_user, get_report_media_s3_path
from storage.presigned import get_presigned_url
from core.logger import logger
import config


router = APIRouter(prefix="/api/reports", tags=["reports"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


# Request/Response Models
class ReportCreate(BaseModel):
    """Create report request."""
    incidentType: str
    location: str
    occurredAt: str  # ISO datetime string
    description: Optional[str] = None
    witnesses: Optional[str] = None
    anonymous: bool = False


class ReportStatusUpdate(BaseModel):
    """Update report status request."""
    status: str  # pending | investigating | resolved | rejected | fake
    reporterRewardPoints: Optional[int] = None  # When status=resolved: stars/points for reporter; visible to management and reporter


class ReportActionRequest(BaseModel):
    """Take disciplinary action request."""
    actionType: str
    studentIds: List[int]
    notes: Optional[str] = None


class ReportResponse(BaseModel):
    """Report response model."""
    id: int
    reportId: str
    incidentType: str
    location: str
    occurredAt: str
    description: Optional[str]
    witnesses: Optional[str]
    reporterType: str
    reporterId: Optional[int]
    collegeId: int
    status: str
    reporterRewardPoints: Optional[int] = None  # Stars/points when resolved; visible to management and reporter
    hasVideo: bool
    hasPhoto: bool
    mediaUrls: List[dict]
    aiProcessed: bool
    createdAt: str
    updatedAt: str
    reporter: Optional[dict] = None  # When management views and status=fake: student who uploaded (so management can see uploader data)


class ReportListResponse(BaseModel):
    """Report list response."""
    data: List[dict]
    total: int
    page: int
    limit: int


def generate_report_id(college_code: str, reporter_type: str, sequence: int) -> str:
    """Generate report ID like RPT-FAC-013."""
    prefix = "FAC" if reporter_type == "faculty" else "STU"
    return f"RPT-{prefix}-{str(sequence).zfill(3)}"


def get_next_report_sequence(db: Session, reporter_type: str) -> int:
    """
    Get the next globally unique sequence number for report_id (RPT-FAC-xxx or RPT-STU-xxx).
    report_id is unique across all colleges, so we use the max existing sequence for this prefix.
    """
    prefix = "FAC" if reporter_type == "faculty" else "STU"
    pattern = f"RPT-{prefix}-%"
    existing = db.query(Report.report_id).filter(Report.report_id.like(pattern)).all()
    max_seq = 0
    for (rid,) in existing:
        try:
            # report_id format: RPT-STU-001 -> 001
            num_part = rid.split("-")[-1]
            max_seq = max(max_seq, int(num_part))
        except (ValueError, IndexError):
            continue
    return max_seq + 1


def _media_url_for_display(file_url: str, college_code: Optional[str], expiration: int = 3600) -> str:
    """
    Return an HTTP-usable URL for report media.
    If file_url is an S3 URI (s3://bucket/key), return a presigned URL; otherwise return as-is (local).
    """
    if not file_url or not file_url.startswith("s3://"):
        return file_url or ""
    if not config.s3_client or not college_code:
        return file_url
    try:
        parts = file_url.split("/", 3)  # ["s3:", "", "bucket", "key/path"]
        if len(parts) < 4:
            return file_url
        s3_key = parts[3]
        return config.s3_client.get_presigned_url(s3_key, college_code=college_code, expiration=expiration)
    except Exception as e:
        logger.warning(f"Could not generate presigned URL for {file_url}: {e}")
        return file_url


def _parse_anonymous(value: Optional[str]) -> bool:
    """Parse form string to bool. Form data sends everything as strings."""
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_report(
    current_user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db_session),
    incidentType: str = Form(..., description="Incident type: RAGGING, FIGHTING, MISBEHAVIOUR, MONEY, FACULTY, OTHERS"),
    location: str = Form(...),
    occurredAt: str = Form(..., description="ISO 8601 datetime"),
    description: Optional[str] = Form(None),
    witnesses: Optional[str] = Form(None),
    anonymous: Optional[str] = Form("false"),
    media: List[UploadFile] = File(default=[]),
    passportPhoto: Optional[UploadFile] = File(None, alias="passport"),
    degree360Photo: Optional[UploadFile] = File(None),
    degree360Photos: List[UploadFile] = File(default=[]),
):
    """
    Create incident report (faculty / student).
    Use multipart/form-data (not JSON). Field names: incidentType, location, occurredAt,
    description (optional), witnesses (optional), anonymous (optional, send "true" or "false"),
    media (at least one file; use same key "media" for multiple files),
    passportPhoto (optional single file), degree360Photos (optional; use same key for multiple files).
    """
    # If client sent JSON by mistake, fail fast with a clear message
    content_type = request.headers.get("content-type", "") if request else ""
    if "application/json" in content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use multipart/form-data for this endpoint, not JSON. Send incidentType, location, occurredAt, description, witnesses, anonymous, and media (files) as form fields."
        )

    anonymous_bool = _parse_anonymous(anonymous)
    incident_type = incidentType
    occurred_at = occurredAt
    # Validate incident type
    try:
        incident_type_enum = IncidentType[incident_type.upper().replace("-", "_")]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid incident type: {incident_type}"
        )
    
    # Validate role
    if current_user.role not in [UserRole.FACULTY, UserRole.STUDENT]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only faculty and students can create reports"
        )
    
    # Validate media files
    if not media or len(media) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one media file (image/video) is required"
        )
    
    # Parse occurred_at
    try:
        occurred_at_dt = datetime.fromisoformat(occurred_at.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use ISO 8601 format."
        )
    
    # Get college_id
    college_id = current_user.college_id
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must be associated with a college"
        )
    
    # Get college code for S3
    college_code = get_college_code_from_user(current_user)
    
    # Generate report ID (globally unique: RPT-STU-001, RPT-FAC-001, etc.)
    reporter_type = "faculty" if current_user.role == UserRole.FACULTY else "student"
    next_seq = get_next_report_sequence(db, reporter_type)
    report_id = generate_report_id(college_code, reporter_type, next_seq)
    
    # Create report
    report = Report(
        report_id=report_id,
        incident_type=incident_type_enum,
        location=location,
        occurred_at=occurred_at_dt,
        description=description,
        witnesses=witnesses,
        reporter_type=ReportReporterType.ANONYMOUS if anonymous_bool else ReportReporterType.IDENTIFIED,
        reporter_id=None if anonymous_bool else current_user.id,
        owner_id=current_user.id,
        college_id=college_id,
        status=ReportStatus.PENDING,
        has_video=False,
        has_photo=False,
        ai_processed=False
    )
    db.add(report)
    db.flush()  # Get report.id
    
    # Upload media files and collect for AI face detection (run after commit)
    media_urls = []
    has_video = False
    has_photo = False
    media_for_detection: List[dict] = []

    for file in media:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        is_image = file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        if not (is_video or is_image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file_ext}. Only images and videos are allowed."
            )
        
        # Upload to S3 (videos and photos stored in college bucket: reports/{report_id}/{media_id}{ext})
        media_id = str(uuid.uuid4())
        s3_key = get_report_media_s3_path(
            college_code, report.id, media_id, file_ext, report_id_str=getattr(report, "report_id", None)
        )
        
        try:
            content = await file.read()
            
            if config.s3_client:
                file_url = config.s3_client.upload_fileobj(
                    BytesIO(content),
                    s3_key,
                    college_code=college_code,
                    content_type=file.content_type or "application/octet-stream"
                )
            elif getattr(config, "USE_S3_ONLY", True):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
                )
            else:
                local_dir = config.UPLOADS_DIR / "reports" / str(report.id)
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / f"{media_id}{file_ext}"
                with open(local_path, "wb") as f:
                    f.write(content)
                file_url = str(local_path)
            
            # Create media record (link S3 to DB: store bucket + key so we can get data from S3 via DB)
            media_type = MediaType.VIDEO if is_video else MediaType.IMAGE
            s3_bucket = None
            s3_key_stored = None
            if config.s3_client:
                s3_bucket = config.s3_client.get_bucket_name(college_code)
                s3_key_stored = s3_key
            media_record = ReportMedia(
                report_id=report.id,
                media_type=media_type,
                file_url=file_url,
                s3_bucket=s3_bucket,
                s3_key=s3_key_stored,
                file_size_bytes=len(content)
            )
            db.add(media_record)
            
            media_urls.append({
                "url": file_url,
                "type": media_type.value
            })
            # When S3 is used, pass s3_key so detection fetches from S3 (single source); otherwise pass content
            if config.s3_client and s3_key_stored:
                media_for_detection.append({
                    "s3_key": s3_key_stored,
                    "is_video": is_video,
                    "file_ext": file_ext,
                })
            else:
                media_for_detection.append({
                    "content": content,
                    "is_video": is_video,
                    "file_ext": file_ext,
                })

            if is_video:
                has_video = True
            if is_image:
                has_photo = True
                
        except Exception as e:
            logger.error(f"Failed to upload media file: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload media file: {str(e)}"
            )
    
    # Upload optional passport photo (stored as report media IMAGE with key passport.jpg)
    if passportPhoto and passportPhoto.filename:
        ext = os.path.splitext(passportPhoto.filename or "")[1].lower() or ".jpg"
        if ext in [".jpg", ".jpeg", ".png"]:
            try:
                content = await passportPhoto.read()
                media_id_passport = "passport"
                s3_key = get_report_media_s3_path(
                    college_code, report.id, media_id_passport, ext, report_id_str=getattr(report, "report_id", None)
                )
                if config.s3_client:
                    file_url = config.s3_client.upload_fileobj(
                        BytesIO(content), s3_key, college_code=college_code,
                        content_type=passportPhoto.content_type or "image/jpeg"
                    )
                elif getattr(config, "USE_S3_ONLY", True):
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
                    )
                else:
                    local_dir = config.UPLOADS_DIR / "reports" / str(report.id)
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_path = local_dir / f"passport{ext}"
                    with open(local_path, "wb") as f:
                        f.write(content)
                    file_url = str(local_path)
                s3_bucket = config.s3_client.get_bucket_name(college_code) if config.s3_client else None
                s3_key_stored = s3_key if config.s3_client else None
                db.add(ReportMedia(
                    report_id=report.id, media_type=MediaType.IMAGE, file_url=file_url,
                    s3_bucket=s3_bucket, s3_key=s3_key_stored, file_size_bytes=len(content)
                ))
                has_photo = True
                media_urls.append({"url": file_url, "type": "image"})
                # Include in face detection so incident report auto-detects faces in passport
                if config.s3_client and s3_key_stored:
                    media_for_detection.append({"s3_key": s3_key_stored, "is_video": False, "file_ext": ext})
                else:
                    media_for_detection.append({"content": content, "is_video": False, "file_ext": ext})
            except Exception as e:
                logger.warning(f"Failed to upload passport photo: {e}", exc_info=True)
    
    # Upload optional 360-degree / face-gallery photos (stored as report media IMAGE with key face_gallery_NN.ext)
    all_360 = list(degree360Photos or [])
    if degree360Photo and degree360Photo.filename:
        all_360.insert(0, degree360Photo)
    for idx, file in enumerate(all_360):
        if not file or not file.filename:
            continue
        ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            content = await file.read()
            media_id_360 = f"face_gallery_{str(idx + 1).zfill(2)}"
            s3_key = get_report_media_s3_path(
                college_code, report.id, media_id_360, ext, report_id_str=getattr(report, "report_id", None)
            )
            if config.s3_client:
                file_url = config.s3_client.upload_fileobj(
                    BytesIO(content), s3_key, college_code=college_code,
                    content_type=file.content_type or "image/jpeg"
                )
            elif getattr(config, "USE_S3_ONLY", True):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
                )
            else:
                local_dir = config.UPLOADS_DIR / "reports" / str(report.id)
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / f"{media_id_360}{ext}"
                with open(local_path, "wb") as f:
                    f.write(content)
                file_url = str(local_path)
            s3_bucket = config.s3_client.get_bucket_name(college_code) if config.s3_client else None
            s3_key_stored = s3_key if config.s3_client else None
            db.add(ReportMedia(
                report_id=report.id, media_type=MediaType.IMAGE, file_url=file_url,
                s3_bucket=s3_bucket, s3_key=s3_key_stored, file_size_bytes=len(content)
            ))
            has_photo = True
            media_urls.append({"url": file_url, "type": "image"})
            # Include in face detection so incident report auto-detects faces in 360° photos
            if config.s3_client and s3_key_stored:
                media_for_detection.append({"s3_key": s3_key_stored, "is_video": False, "file_ext": ext})
            else:
                media_for_detection.append({"content": content, "is_video": False, "file_ext": ext})
        except Exception as e:
            logger.warning(f"Failed to upload 360-degree photo {idx + 1}: {e}", exc_info=True)
    
    # Update report with media flags
    report.has_video = has_video
    report.has_photo = has_photo
    
    db.commit()
    db.refresh(report)
    
    # Update faculty reports_submitted count
    if current_user.role == UserRole.FACULTY and not anonymous_bool:
        current_user.reports_submitted = (current_user.reports_submitted or 0) + 1
        db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="report_create",
        user_id=current_user.id if not anonymous_bool else None,
        resource_type="report",
        resource_id=str(report.id)
    )

    # Run face detection on uploaded media in background (reuses core VideoProcessor / face DB)
    if media_for_detection and getattr(config, "face_db", None):
        asyncio.create_task(
            asyncio.to_thread(
                run_report_media_face_detection,
                report.id,
                college_id,
                media_for_detection,
            )
        )
        logger.info(f"Report {report.id}: face detection started for {len(media_for_detection)} media file(s)")
    
    return {
        "id": report.id,
        "reportId": report.report_id,
        "message": "Report created successfully"
    }


@router.get("", response_model=ReportListResponse)
async def list_reports(
    college: Optional[int] = Query(None, alias="college"),
    type: Optional[str] = Query(None, alias="type"),
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List reports with role-based filtering.
    """
    query = db.query(Report)
    
    # Role-based filtering
    if current_user.role == UserRole.MASTER_ADMIN:
        # Master can see all reports
        if college:
            query = query.filter(Report.college_id == college)
    elif current_user.role == UserRole.MANAGEMENT:
        # Management can see all reports from their managed college
        managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
        if managed_college_id is not None:
            query = query.filter(Report.college_id == managed_college_id)
        else:
            query = query.filter(False)  # No college assigned
    elif current_user.role == UserRole.FACULTY:
        # Faculty can see all reports from their college
        if current_user.college_id is not None:
            query = query.filter(Report.college_id == current_user.college_id)
        else:
            query = query.filter(False)
    elif current_user.role == UserRole.STUDENT:
        # Students can only see their own submitted reports.
        # Use owner_id (internal owner) when available; fall back to reporter_id for legacy data.
        query = query.filter(
            or_(
                getattr(Report, "owner_id", None) == current_user.id if hasattr(Report, "owner_id") else False,
                Report.reporter_id == current_user.id,
            )
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Filter by incident type
    if type:
        try:
            incident_type_enum = IncidentType[type.upper()]
            query = query.filter(Report.incident_type == incident_type_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid incident type: {type}"
            )
    
    # Filter by status
    if status_filter:
        try:
            status_enum = ReportStatus[status_filter.upper()]
            query = query.filter(Report.status == status_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )
    
    # Search
    if search:
        query = query.filter(
            or_(
                Report.location.ilike(f"%{search}%"),
                Report.description.ilike(f"%{search}%"),
                Report.report_id.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    reports = query.order_by(Report.created_at.desc()).offset(offset).limit(limit).all()
    
    # Build response
    report_list = []
    for report in reports:
        college_code = report.college.college_code if report.college else None
        # Get media URLs (presigned for S3 so clients can view video/photos)
        media_records = db.query(ReportMedia).filter(ReportMedia.report_id == report.id).all()
        media_urls = [
            {"url": _media_url_for_display(m.file_url, college_code), "type": m.media_type.value}
            for m in media_records
        ]
        
        report_dict = {
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "occurredAt": report.occurred_at.isoformat(),
            "description": report.description,
            "witnesses": report.witnesses,
            "reporterType": report.reporter_type.value,
            "reporterId": report.reporter_id,
            "collegeId": report.college_id,
            "status": report.status.value,
            "reporterRewardPoints": getattr(report, "reporter_reward_points", None),
            "hasVideo": report.has_video,
            "hasPhoto": report.has_photo,
            "mediaUrls": media_urls,
            "aiProcessed": report.ai_processed,
            "createdAt": report.created_at.isoformat(),
            "updatedAt": report.updated_at.isoformat()
        }
        # Management: when status is FAKE, include reporter (uploader) data so management can see who uploaded fake video
        if (current_user.role in [UserRole.MASTER_ADMIN, UserRole.MANAGEMENT]
                and report.status == ReportStatus.FAKE and report.reporter_id):
            reporter_user = db.query(User).filter(User.id == report.reporter_id).first()
            if reporter_user:
                report_dict["reporter"] = {
                    "id": reporter_user.id,
                    "fullName": reporter_user.full_name,
                    "loginId": reporter_user.login_id,
                    "email": reporter_user.email,
                    "studentId": getattr(reporter_user, "student_id", None),
                    "rollNumber": reporter_user.roll_number,
                    "branch": reporter_user.branch,
                    "year": reporter_user.year,
                }
        report_list.append(report_dict)
    
    return ReportListResponse(
        data=report_list,
        total=total,
        page=page,
        limit=limit
    )


@router.delete("/{report_id}")
async def delete_report(
    report_id: int,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Delete a report permanently.
    Master admin only.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    db.delete(report)
    db.commit()
    AuditService.log_from_request(
        db=db,
        request=request,
        action="report_delete",
        user_id=current_user.id,
        resource_type="report",
        resource_id=str(report_id)
    )
    return {"success": True, "message": "Report deleted"}


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get report by ID.
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
        managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
        if managed_college_id is None or report.college_id != managed_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    elif current_user.role == UserRole.FACULTY:
        # Faculty can view any report from their college
        if current_user.college_id is None or report.college_id != current_user.college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    elif current_user.role == UserRole.STUDENT:
        # Students can only view their own submitted reports.
        # Check owner_id first (internal owner), then fall back to reporter_id for legacy data.
        owner_id = getattr(report, "owner_id", None)
        if (owner_id or report.reporter_id) != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    
    # Get media URLs (presigned for S3 so clients can view video/photos)
    college_code = report.college.college_code if report.college else None
    media_records = db.query(ReportMedia).filter(ReportMedia.report_id == report.id).all()
    media_urls = [
        {"url": _media_url_for_display(m.file_url, college_code), "type": m.media_type.value}
        for m in media_records
    ]
    
    reporter_info = None
    # Management: when status is FAKE, include reporter (uploader) data so management can see who uploaded fake video
    if (current_user.role in [UserRole.MASTER_ADMIN, UserRole.MANAGEMENT]
            and report.status == ReportStatus.FAKE and report.reporter_id):
        reporter_user = db.query(User).filter(User.id == report.reporter_id).first()
        if reporter_user:
            reporter_info = {
                "id": reporter_user.id,
                "fullName": reporter_user.full_name,
                "loginId": reporter_user.login_id,
                "email": reporter_user.email,
                "studentId": getattr(reporter_user, "student_id", None),
                "rollNumber": reporter_user.roll_number,
                "branch": reporter_user.branch,
                "year": reporter_user.year,
            }
    return ReportResponse(
        id=report.id,
        reportId=report.report_id,
        incidentType=report.incident_type.value,
        location=report.location,
        occurredAt=report.occurred_at.isoformat(),
        description=report.description,
        witnesses=report.witnesses,
        reporterType=report.reporter_type.value,
        reporterId=report.reporter_id,
        collegeId=report.college_id,
        status=report.status.value,
        reporterRewardPoints=getattr(report, "reporter_reward_points", None),
        hasVideo=report.has_video,
        hasPhoto=report.has_photo,
        mediaUrls=media_urls,
        aiProcessed=report.ai_processed,
        createdAt=report.created_at.isoformat(),
        updatedAt=report.updated_at.isoformat(),
        reporter=reporter_info,
    )


@router.post("/{report_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_report_media(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Fetch report media from S3 (or local), run face detection using core pipeline,
    compare with student face database, store DetectedFace records and upload face crops to S3.
    Management and Faculty (for their college) or Master admin can trigger.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    if current_user.role == UserRole.MASTER_ADMIN:
        pass
    elif current_user.role == UserRole.MANAGEMENT:
        managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
        if managed_college_id is None or report.college_id != managed_college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user.role == UserRole.FACULTY:
        if current_user.college_id is None or report.college_id != current_user.college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    asyncio.create_task(asyncio.to_thread(run_report_face_detection_from_storage, report_id))
    return {
        "message": "Report media face detection started. Media will be fetched from S3, faces detected and compared with student photos, results stored in DB and face crops in S3.",
        "reportId": report_id,
    }


@router.patch("/{report_id}/status")
async def update_report_status(
    report_id: int,
    status_data: ReportStatusUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update report status.
    Management and Master only.
    """
    if current_user.role not in [UserRole.MASTER_ADMIN, UserRole.MANAGEMENT]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check authorization for management
    if current_user.role == UserRole.MANAGEMENT:
        managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
        if managed_college_id is None or report.college_id != managed_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    
    # Validate status (pending | investigating | resolved | rejected | fake)
    try:
        status_enum = ReportStatus[status_data.status.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {status_data.status}. Use: pending, investigating, resolved, rejected, fake."
        )
    
    # When marking as fake: management can see reporter data via list/get (reporter included when status=fake).
    # When marking as resolved: optional reporterRewardPoints (stars/points) for the reporter; visible to management and reporter.
    reporter_reward_points = getattr(status_data, "reporterRewardPoints", None)
    if reporter_reward_points is not None:
        if status_enum != ReportStatus.RESOLVED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="reporterRewardPoints can only be set when status is resolved"
            )
        if not (0 <= reporter_reward_points <= 100):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="reporterRewardPoints must be between 0 and 100"
            )
    
    # Update status
    old_status = report.status
    report.status = status_enum
    if status_enum == ReportStatus.RESOLVED and reporter_reward_points is not None:
        report.reporter_reward_points = reporter_reward_points
        if report.reporter_id:
            reporter = db.query(User).filter(User.id == report.reporter_id).first()
            if reporter:
                reporter.points = (reporter.points or 0) + reporter_reward_points
                db.flush()
    db.commit()
    
    # Update faculty reports_resolved count if resolved
    if status_enum == ReportStatus.RESOLVED and report.reporter_id:
        reporter = db.query(User).filter(User.id == report.reporter_id).first()
        if reporter and reporter.role == UserRole.FACULTY:
            reporter.reports_resolved = (reporter.reports_resolved or 0) + 1
            db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="report_status_update",
        user_id=current_user.id,
        resource_type="report",
        resource_id=str(report_id)
    )
    
    return {
        "id": report.id,
        "reportId": report.report_id,
        "status": report.status.value,
        "reporterRewardPoints": getattr(report, "reporter_reward_points", None),
        "message": "Status updated successfully"
    }


@router.get("/{report_id}/ai-detection")
async def get_ai_detection(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get AI detection results (faces) for a report.
    Master admin, Management, and Faculty (for their college) can view.
    """
    if current_user.role not in [UserRole.MASTER_ADMIN, UserRole.MANAGEMENT, UserRole.FACULTY]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check authorization: management by managed college, faculty by college
    if current_user.role == UserRole.MANAGEMENT:
        managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
        if managed_college_id is None or report.college_id != managed_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    elif current_user.role == UserRole.FACULTY:
        if current_user.college_id is None or report.college_id != current_user.college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    
    # Get detected faces with student (User) loaded for full details
    detected_faces = (
        db.query(DetectedFace)
        .filter(DetectedFace.report_id == report_id)
        .all()
    )
    # Load all students in one query for efficiency
    student_ids = [f.student_id for f in detected_faces]
    students = {u.id: u for u in db.query(User).filter(User.id.in_(student_ids)).all()} if student_ids else {}
    # Load colleges for student college names
    college_ids = {u.college_id for u in students.values() if u.college_id}
    colleges = {c.id: c for c in db.query(College).filter(College.id.in_(college_ids)).all()} if college_ids else {}
    # Resolve department: first by student.department_id (department the student is tagged to), then by (college_id, department_name)
    dept_ids = [u.department_id for u in students.values() if u.department_id]
    department_by_id = {d.id: d for d in db.query(CollegeDepartment).filter(CollegeDepartment.id.in_(dept_ids)).all()} if dept_ids else {}
    dept_pairs = [(u.college_id, u.department_name) for u in students.values() if u.college_id and u.department_name]
    department_by_college_name = {}
    if dept_pairs:
        depts = db.query(CollegeDepartment).filter(
            tuple_(CollegeDepartment.college_id, CollegeDepartment.name).in_(dept_pairs)
        ).all()
        department_by_college_name = {(d.college_id, d.name): d for d in depts}

    def _department_payload(dept):
        """Full department details for response."""
        if not dept:
            return None
        return {
            "id": dept.id,
            "name": dept.name,
            "collegeId": dept.college_id,
            "createdAt": dept.created_at.isoformat() if dept.created_at else None,
        }

    face_list = []
    for face in detected_faces:
        student = students.get(face.student_id)
        college = colleges.get(student.college_id) if student and student.college_id else None
        dept = None
        if student:
            if getattr(student, "department_id", None) and student.department_id in department_by_id:
                dept = department_by_id[student.department_id]
            elif student.college_id and student.department_name:
                dept = department_by_college_name.get((student.college_id, student.department_name))
        department_name_resolved = (dept.name if dept else (student.department_name if student else None)) or face.department
        department_details = _department_payload(dept)

        student_payload = None
        if student:
            student_payload = {
                "id": student.id,
                "loginId": student.login_id,
                "email": student.email,
                "name": student.full_name,
                "rollNumber": student.roll_number,
                "studentId": getattr(student, "student_id", None),
                "collegeStudentId": student.college_student_id,
                "departmentId": dept.id if dept else None,
                "department": dept.name if dept else student.department_name,
                "departmentDetails": department_details,
                "year": student.year,
                "branch": student.branch,
                "collegeId": student.college_id,
                "collegeName": college.name if college else None,
                "collegeCode": college.college_code if college else None,
                "status": student.status.value if student.status else None,
                "incidents": student.incidents,
                "lastIncidentAt": student.last_incident_at.isoformat() if student.last_incident_at else None,
                "faceRegistered": student.face_registered,
                "createdAt": student.created_at.isoformat() if student.created_at else None,
                "updatedAt": student.updated_at.isoformat() if student.updated_at else None,
            }
        face_dict = {
            "id": face.id,
            "reportId": face.report_id,
            "studentId": face.student_id,
            "name": face.name,
            "department": department_name_resolved,
            "departmentDetails": department_details,
            "year": face.year,
            "confidence": face.confidence,
            "previousIncidents": face.previous_incidents,
            "boundingBox": face.bounding_box,
            "referenceImageUrl": get_presigned_url(face.reference_image_url) or face.reference_image_url,
            "detectedImageUrl": get_presigned_url(face.detected_image_url) or face.detected_image_url,
            "createdAt": face.created_at.isoformat(),
            "student": student_payload,
        }
        face_list.append(face_dict)

    return {
        "detectedFaces": face_list
    }


@router.post("/{report_id}/action")
async def take_action(
    report_id: int,
    action_data: ReportActionRequest,
    request: Request,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Take disciplinary action on students.
    Management only.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check authorization (management: report must belong to their college)
    managed_college_id = getattr(current_user, "managed_college_id", None) or current_user.college_id
    if managed_college_id is None or report.college_id != managed_college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Validate action type
    try:
        action_type_enum = ActionType[action_data.actionType.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action type: {action_data.actionType}"
        )
    
    # Create disciplinary actions
    created_actions = []
    for student_id in action_data.studentIds:
        # Verify student belongs to management's college
        student = db.query(User).filter(
            User.id == student_id,
            User.college_id == managed_college_id,
            User.role == UserRole.STUDENT
        ).first()
        
        if not student:
            continue  # Skip invalid students
        
        # Create action
        action = DisciplinaryAction(
            report_id=report_id,
            student_id=student_id,
            action_type=action_type_enum,
            notes=action_data.notes,
            created_by=current_user.id
        )
        db.add(action)
        created_actions.append(student_id)
        
        # Update student status based on action
        if action_type_enum in [ActionType.SUSPENSION_1D, ActionType.SUSPENSION_3D, ActionType.SUSPENSION_7D]:
            from database.models import StudentStatus
            student.status = StudentStatus.SUSPENDED
        elif action_type_enum == ActionType.WARNING:
            from database.models import StudentStatus
            student.status = StudentStatus.WARNING
        
        # Update student incidents count
        student.incidents = (student.incidents or 0) + 1
        student.last_incident_at = datetime.utcnow()
    
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="disciplinary_action",
        user_id=current_user.id,
        resource_type="report",
        resource_id=str(report_id)
    )
    
    return {
        "success": True,
        "actionsCreated": len(created_actions),
        "studentIds": created_actions
    }
