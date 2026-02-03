"""
Profile APIs (All Authenticated Users).
"""
import asyncio
import os
import tempfile
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request, File, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from database.connection import Database
from database.models import User, UserRole, StudentMedia, StudentMediaType, CollegeDepartment
from auth.dependencies import get_current_user
from services.audit_service import AuditService
from services.student_embedding_sync import run_student_embedding_sync
from storage.s3_paths import (
    get_college_code_from_user,
    get_face_s3_path,
    campus_student_profile_passport_path,
    campus_student_face_gallery_path,
    _use_campus_bucket,
)
from storage.presigned import get_presigned_url
from core.logger import logger
import config


router = APIRouter(prefix="/api/profile", tags=["profile"])


def _viewable_url_for_media(media: StudentMedia, college_code: Optional[str]) -> Optional[str]:
    """Return a URL the frontend can use to display the file (presigned HTTPS if S3, else file_url)."""
    if not media:
        return None
    if media.s3_bucket and media.s3_key and config.s3_client and college_code:
        try:
            return config.s3_client.get_presigned_url(
                media.s3_key, college_code=college_code, expiration=3600
            )
        except Exception as e:
            logger.warning("Presigned URL failed for student_media %s: %s", media.id, e)
    # If file_url is an S3 path (s3://...), convert to presigned HTTPS so browsers can load it
    if media.file_url and str(media.file_url).strip().startswith("s3://"):
        presigned = get_presigned_url(media.file_url)
        if presigned:
            return presigned
    return media.file_url


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


class ProfileUpdate(BaseModel):
    """Update profile request."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    collegeId: Optional[int] = None
    collegeName: Optional[str] = None
    department: Optional[str] = None
    avatar: Optional[str] = None  # URL or base64


class ProfileResponse(BaseModel):
    """Profile response model."""
    id: int
    studentId: Optional[str] = None  # Generated unique ID (students only)
    rollNumber: Optional[str] = None
    email: Optional[str]
    name: Optional[str]
    role: str
    status: str
    collegeId: Optional[int]
    collegeName: Optional[str]
    department: Optional[str]
    year: Optional[str]
    avatarUrl: Optional[str]
    createdAt: str
    updatedAt: str


def _resolve_department_name(db: Session, user: User) -> Optional[str]:
    """
    Resolve department name for profile display.
    Priority:
    1. Explicit department_name on user
    2. CollegeDepartment.name via department_id
    3. For students, fall back to branch (common frontend mapping)
    """
    # 1) Direct field
    if getattr(user, "department_name", None):
        return user.department_name
    
    # 2) Department via foreign key
    department_id = getattr(user, "department_id", None)
    if department_id:
        dept = (
            db.query(CollegeDepartment)
            .filter(CollegeDepartment.id == department_id)
            .first()
        )
        if dept:
            return dept.name
    
    # 3) For students, use branch as department fallback
    if getattr(user, "role", None) == UserRole.STUDENT and getattr(user, "branch", None):
        return user.branch
    
    return None


@router.get("", response_model=ProfileResponse)
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get own profile.
    All authenticated users.
    """
    college_name = None
    if current_user.college_id:
        from database.models import College
        college = db.query(College).filter(College.id == current_user.college_id).first()
        if college:
            college_name = college.name

    department = _resolve_department_name(db, current_user)

    return ProfileResponse(
        id=current_user.id,
        studentId=getattr(current_user, "student_id", None),
        rollNumber=current_user.roll_number,
        email=current_user.email,
        name=current_user.full_name,
        role=current_user.role.value,
        status=current_user.status.value if current_user.status else "active",
        collegeId=current_user.college_id,
        collegeName=college_name or current_user.college_name,
        department=department,
        year=current_user.year,
        avatarUrl=current_user.avatar_url,
        createdAt=current_user.created_at.isoformat(),
        updatedAt=current_user.updated_at.isoformat()
    )


@router.put("", response_model=ProfileResponse)
@router.patch("", response_model=ProfileResponse)
async def update_profile(
    profile_data: ProfileUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update own profile (PUT or PATCH).
    All authenticated users.
    """
    # Load user from this route's session to avoid "not persistent in this Session" on refresh
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Update fields
    if profile_data.name is not None:
        user.full_name = profile_data.name

    if profile_data.email is not None:
        existing = db.query(User).filter(User.email == profile_data.email, User.id != current_user.id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        user.email = profile_data.email

    if profile_data.collegeId is not None:
        if current_user.role.value not in ["master_admin", "master"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot change college ID"
            )
        user.college_id = profile_data.collegeId

    if profile_data.collegeName is not None:
        user.college_name = profile_data.collegeName

    if profile_data.department is not None:
        user.department_name = profile_data.department

    if profile_data.avatar is not None:
        user.avatar_url = profile_data.avatar

    db.commit()
    db.refresh(user)

    AuditService.log_from_request(
        db=db,
        request=request,
        action="profile_update",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )

    college_name = None
    if user.college_id:
        from database.models import College
        college = db.query(College).filter(College.id == user.college_id).first()
        if college:
            college_name = college.name

    department = _resolve_department_name(db, user)

    return ProfileResponse(
        id=user.id,
        studentId=getattr(user, "student_id", None),
        rollNumber=user.roll_number,
        email=user.email,
        name=user.full_name,
        role=user.role.value,
        status=user.status.value if user.status else "active",
        collegeId=user.college_id,
        collegeName=college_name or user.college_name,
        department=department,
        year=user.year,
        avatarUrl=user.avatar_url,
        createdAt=user.created_at.isoformat(),
        updatedAt=user.updated_at.isoformat()
    )


@router.post("/passport", status_code=status.HTTP_200_OK)
async def upload_own_passport(
    file: UploadFile = File(...),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Upload your own passport-size photo (My Profile > Face Recognition Setup).
    Students and faculty only. Stored in S3 (or local) and linked in student_media.
    """
    if current_user.role not in (UserRole.STUDENT, UserRole.FACULTY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students and faculty can upload profile passport photo.",
        )
    if not current_user.college_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your account is not assigned to a college.",
        )
    file_ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    if file_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPEG/PNG allowed")
    college_code = get_college_code_from_user(current_user)
    if not college_code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="College code not found")
    student_id_str = (
        getattr(current_user, "college_student_id", None)
        or current_user.roll_number
        or f"STU{current_user.id}"
    )
    if _use_campus_bucket():
        s3_key = campus_student_profile_passport_path(college_code, student_id_str)
    else:
        person_id = current_user.roll_number or f"STUDENT_{current_user.id}"
        s3_key = get_face_s3_path(college_code, person_id, f"passport{file_ext}")
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        s3_bucket = None
        if config.s3_client:
            file_url = config.s3_client.upload_file(
                tmp_path, s3_key, college_code=college_code, content_type=file.content_type or "image/jpeg"
            )
            s3_bucket = config.s3_client.get_bucket_name(college_code)
        elif getattr(config, "USE_S3_ONLY", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
            )
        else:
            local_dir = config.COLLEGE_FACES_DIR / student_id_str / "profile"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / "passport.jpg"
            with open(local_path, "wb") as f:
                f.write(content)
            file_url = str(local_path)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        user = db.query(User).filter(User.id == current_user.id).first()
        if user:
            user.face_image_url = file_url
            user.face_registered = True
            user.s3_bucket = s3_bucket
            user.s3_face_key = s3_key if s3_bucket else None
        existing = db.query(StudentMedia).filter(
            StudentMedia.user_id == current_user.id,
            StudentMedia.media_type == StudentMediaType.PASSPORT,
        ).first()
        if existing:
            existing.s3_bucket = s3_bucket
            existing.s3_key = s3_key if s3_bucket else None
            existing.file_url = file_url
            existing.filename = "passport.jpg"
        else:
            db.add(
                StudentMedia(
                    user_id=current_user.id,
                    media_type=StudentMediaType.PASSPORT,
                    s3_bucket=s3_bucket,
                    s3_key=s3_key if s3_bucket else None,
                    file_url=file_url,
                    filename="passport.jpg",
                    display_order=0,
                )
            )
        db.commit()
        AuditService.log_from_request(
            db=db, request=request, action="profile_passport_upload", user_id=current_user.id, resource_type="user", resource_id=str(current_user.id)
        )
        # Auto-create/update embeddings/face_embedding.npy in background (no app restart)
        person_id = student_id_str
        name = getattr(current_user, "name", None) or student_id_str
        if getattr(config, "face_db", None):
            asyncio.create_task(
                asyncio.to_thread(
                    run_student_embedding_sync,
                    person_id,
                    name,
                    college_code,
                    [content],
                )
            )
        return {"success": True, "message": "Passport photo uploaded", "url": file_url}
    except Exception as e:
        logger.error(f"Profile passport upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/passport", status_code=status.HTTP_200_OK)
async def get_own_passport(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Get your own passport photo URL (for My Profile > Face Recognition Setup).
    Students and faculty only.
    """
    if current_user.role not in (UserRole.STUDENT, UserRole.FACULTY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students and faculty have profile passport.",
        )
    college_code = get_college_code_from_user(current_user)
    passport = db.query(StudentMedia).filter(
        StudentMedia.user_id == current_user.id,
        StudentMedia.media_type == StudentMediaType.PASSPORT,
    ).first()
    if not passport:
        return {"hasPassport": False, "url": None, "filename": None}
    url = _viewable_url_for_media(passport, college_code)
    return {"hasPassport": True, "url": url, "filename": passport.filename}


@router.get("/face-gallery", status_code=status.HTTP_200_OK)
async def get_own_face_gallery(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Get your own 360° / face-gallery photo URLs (for My Profile > Face Recognition Setup).
    Students and faculty only.
    """
    if current_user.role not in (UserRole.STUDENT, UserRole.FACULTY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students and faculty have profile face-gallery.",
        )
    college_code = get_college_code_from_user(current_user)
    items = (
        db.query(StudentMedia)
        .filter(
            StudentMedia.user_id == current_user.id,
            StudentMedia.media_type == StudentMediaType.FACE_GALLERY,
        )
        .order_by(StudentMedia.display_order.asc(), StudentMedia.id.asc())
        .all()
    )
    result = []
    for m in items:
        url = _viewable_url_for_media(m, college_code)
        result.append({"id": m.id, "filename": m.filename, "url": url, "displayOrder": m.display_order})
    return {"items": result}


@router.post("/face-gallery", status_code=status.HTTP_200_OK)
async def upload_own_face_gallery(
    files: List[UploadFile] = File(...),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Upload your own 360-degree / face-gallery photos (My Profile > Face Recognition Setup).
    Students and faculty only. Stored in S3 (or local) and linked in student_media.
    """
    if current_user.role not in (UserRole.STUDENT, UserRole.FACULTY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students and faculty can upload profile face-gallery photos.",
        )
    if not current_user.college_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your account is not assigned to a college.",
        )
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file required")
    college_code = get_college_code_from_user(current_user)
    if not college_code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="College code not found")
    student_id_str = (
        getattr(current_user, "college_student_id", None)
        or current_user.roll_number
        or f"STU{current_user.id}"
    )
    uploaded = []
    for idx, file in enumerate(files):
        if not file or not file.filename:
            continue
        ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        display_name = f"img_{str(idx + 1).zfill(2)}{ext}"
        if _use_campus_bucket():
            s3_key = campus_student_face_gallery_path(college_code, student_id_str, display_name)
        else:
            person_id = current_user.roll_number or f"STUDENT_{current_user.id}"
            s3_key = get_face_s3_path(college_code, person_id, f"gallery_{idx + 1}{ext}")
            display_name = s3_key.split("/")[-1]
        try:
            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            s3_bucket = None
            if config.s3_client:
                url = config.s3_client.upload_file(
                    tmp_path, s3_key, college_code=college_code, content_type=file.content_type or "image/jpeg"
                )
                s3_bucket = config.s3_client.get_bucket_name(college_code)
            elif getattr(config, "USE_S3_ONLY", True):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
                )
            else:
                local_dir = config.COLLEGE_FACES_DIR / student_id_str / "face-gallery"
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / display_name
                with open(local_path, "wb") as f:
                    f.write(content)
                url = str(local_path)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            db.add(
                StudentMedia(
                    user_id=current_user.id,
                    media_type=StudentMediaType.FACE_GALLERY,
                    s3_bucket=s3_bucket,
                    s3_key=s3_key if s3_bucket else None,
                    file_url=url,
                    filename=display_name,
                    display_order=idx + 1,
                )
            )
            uploaded.append({"filename": display_name, "url": url})
        except Exception as e:
            logger.warning(f"Profile face-gallery file {idx + 1} upload failed: {e}")
    if not uploaded:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid image could be uploaded")
    db.commit()
    AuditService.log_from_request(
        db=db, request=request, action="profile_face_gallery_upload", user_id=current_user.id, resource_type="user", resource_id=str(current_user.id)
    )
    return {"success": True, "message": f"{len(uploaded)} photo(s) uploaded", "uploaded": uploaded}
