"""
Student Management APIs (Management).
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import asyncio
import csv
import io
import os

from database.connection import Database
from database.models import (
    User, UserRole, UserStatus, StudentStatus, Report, DisciplinaryAction, ActionType,
    StudentMedia, StudentMediaType, CollegeDepartment,
)
from auth.dependencies import get_current_user, require_management, require_master_admin, get_management_college_id
from services.auth_service import AuthService
from services.audit_service import AuditService
from services.student_id import generate_student_id
from services.student_embedding_sync import run_student_embedding_sync
from storage.s3_client import S3Client
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


router = APIRouter(prefix="/api/students", tags=["students"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


# Request/Response Models
class StudentUpdate(BaseModel):
    """Update student request."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    department: Optional[str] = None
    departmentId: Optional[int] = None  # CollegeDepartment id the student is tagged to
    year: Optional[str] = None
    rollNumber: Optional[str] = None
    status: Optional[str] = None


class StudentActionRequest(BaseModel):
    """Issue disciplinary action request."""
    actionType: str
    reportId: Optional[int] = None
    notes: Optional[str] = None


class StudentImportRequest(BaseModel):
    """Bulk import students request."""
    students: List[dict]  # [{ id, name, email, department?, year? }]


class StudentResponse(BaseModel):
    """Student response model."""
    id: int
    studentId: Optional[str] = None  # Generated unique ID (STU-{college_id}-{seq})
    rollNumber: Optional[str] = None
    name: str
    email: Optional[str]
    department: Optional[str]
    departmentId: Optional[int] = None
    year: Optional[str]
    collegeId: int
    status: str
    incidents: int
    lastIncident: Optional[str]
    faceRegistered: bool
    createdAt: str


class StudentListResponse(BaseModel):
    """Student list response."""
    data: List[dict]
    total: int
    page: int
    limit: int


class ImportResponse(BaseModel):
    """Import response."""
    imported: int
    errors: Optional[List[dict]] = None


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


@router.get("", response_model=StudentListResponse)
async def list_students(
    department: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List students. Management: all students in their college (optional department filter).
    Faculty: only students in the same department.
    """
    if current_user.role == UserRole.MANAGEMENT:
        college_id = get_management_college_id(current_user)
        if not college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
            )
        query = db.query(User).filter(
            User.role == UserRole.STUDENT,
            User.college_id == college_id
        )
        if department:
            query = query.filter(User.department_name == department)
    elif current_user.role == UserRole.FACULTY:
        if not current_user.college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college."
            )
        # Faculty sees only students in the same department
        query = db.query(User).filter(
            User.role == UserRole.STUDENT,
            User.college_id == current_user.college_id,
            User.department_name == current_user.department_name
        )
        # department query param is ignored for faculty (they only see their dept)
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Management or faculty only."
        )
    
    # Filter by status (shared)
    
    # Filter by status
    if status_filter:
        try:
            status_enum = StudentStatus[status_filter.upper()]
            query = query.filter(User.status == status_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )
    
    # Search
    if search:
        query = query.filter(
            or_(
                User.full_name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
                User.roll_number.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    students = query.offset(offset).limit(limit).all()
    
    # Resolve department names from CollegeDepartment for students with department_id
    dept_ids = [s.department_id for s in students if getattr(s, "department_id", None)]
    dept_by_id = {d.id: d for d in db.query(CollegeDepartment).filter(CollegeDepartment.id.in_(dept_ids)).all()} if dept_ids else {}
    
    # Build response
    student_list = []
    for student in students:
        dept = dept_by_id.get(student.department_id) if getattr(student, "department_id", None) else None
        student_dict = {
            "id": student.id,
            "studentId": getattr(student, "student_id", None),
            "rollNumber": student.roll_number,
            "name": student.full_name,
            "email": student.email,
            "department": (dept.name if dept else student.department_name),
            "departmentId": getattr(student, "department_id", None),
            "year": student.year,
            "collegeId": student.college_id,
            "status": student.status.value if student.status else "active",
            "incidents": student.incidents or 0,
            "lastIncident": student.last_incident_at.isoformat() if student.last_incident_at else None,
            "faceRegistered": student.face_registered or False,
            "createdAt": student.created_at.isoformat()
        }
        student_list.append(student_dict)
    
    return StudentListResponse(
        data=student_list,
        total=total,
        page=page,
        limit=limit
    )


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(
    student_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get student by ID. Management: any student in their college. Faculty: only students in same department.
    """
    if current_user.role == UserRole.MANAGEMENT:
        college_id = get_management_college_id(current_user)
        if not college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
            )
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == college_id
        ).first()
    elif current_user.role == UserRole.FACULTY:
        if not current_user.college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college."
            )
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == current_user.college_id,
            User.department_name == current_user.department_name
        ).first()
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Management or faculty only."
        )
    
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    dept_name = None
    if getattr(student, "department_id", None):
        dept = db.query(CollegeDepartment).filter(CollegeDepartment.id == student.department_id).first()
        dept_name = dept.name if dept else student.department_name
    return StudentResponse(
        id=student.id,
        studentId=getattr(student, "student_id", None),
        rollNumber=student.roll_number,
        name=student.full_name or "",
        email=student.email,
        department=dept_name or student.department_name,
        departmentId=getattr(student, "department_id", None),
        year=student.year,
        collegeId=student.college_id,
        status=student.status.value if student.status else "active",
        incidents=student.incidents or 0,
        lastIncident=student.last_incident_at.isoformat() if student.last_incident_at else None,
        faceRegistered=student.face_registered or False,
        createdAt=student.created_at.isoformat()
    )


@router.post("/import", response_model=ImportResponse)
async def import_students(
    import_data: StudentImportRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Bulk import students from CSV data.
    Management: can add students to their college (any department).
    Faculty: can add students to their college and department only.
    """
    if current_user.role == UserRole.MANAGEMENT:
        college_id = get_management_college_id(current_user)
        if not college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
            )
        # Management can set any department from student_data
        required_department = None
    elif current_user.role == UserRole.FACULTY:
        if not current_user.college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a college."
            )
        if not current_user.department_name:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is not assigned to a department."
            )
        college_id = current_user.college_id
        # Faculty can only add students to their own department
        required_department = current_user.department_name
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Management or faculty only."
        )
    
    imported_count = 0
    errors = []
    
    for idx, student_data in enumerate(import_data.students):
        try:
            # Validate required fields
            if not student_data.get("name") or not student_data.get("email"):
                errors.append({
                    "row": idx + 1,
                    "message": "Name and email are required"
                })
                continue
            
            # Check if student already exists
            existing = db.query(User).filter(
                User.email == student_data["email"],
                User.college_id == college_id
            ).first()
            
            if existing:
                errors.append({
                    "row": idx + 1,
                    "message": f"Student with email {student_data['email']} already exists"
                })
                continue
            
            # Create student without password; user will set password via OTP flow
            student = AuthService.create_user(
                db=db,
                login_id=student_data.get("email") or student_data.get("roll_number") or student_data.get("id"),
                role=UserRole.STUDENT,
                email=student_data["email"],
                full_name=student_data["name"],
                college_id=college_id,
            )
            
            # Set student-specific fields
            # For faculty: force department to match faculty's department
            # For management: use department from student_data if provided
            student.department_name = required_department if required_department else student_data.get("department")
            if student_data.get("departmentId") is not None:
                dept = db.query(CollegeDepartment).filter(
                    CollegeDepartment.id == student_data["departmentId"],
                    CollegeDepartment.college_id == college_id,
                ).first()
                if dept:
                    student.department_id = dept.id
                    student.department_name = dept.name
            elif student.department_name and college_id:
                dept = db.query(CollegeDepartment).filter(
                    CollegeDepartment.college_id == college_id,
                    CollegeDepartment.name == student.department_name,
                ).first()
                if dept:
                    student.department_id = dept.id
            student.year = student_data.get("year")
            # Roll number from input only (academic roll); separate from generated student_id
            student.roll_number = student_data.get("roll_number") or student_data.get("rollNumber")
            # Generated unique ID for storage/S3; college_student_id kept in sync for paths
            student.student_id = generate_student_id(db, college_id)
            student.college_student_id = student.student_id
            
            db.commit()
            imported_count += 1
            
        except Exception as e:
            errors.append({
                "row": idx + 1,
                "message": str(e)
            })
            logger.error(f"Error importing student row {idx + 1}: {e}", exc_info=True)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="students_import",
        user_id=current_user.id,
        resource_type="user"
    )
    
    return ImportResponse(
        imported=imported_count,
        errors=errors if errors else None
    )


@router.put("/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: int,
    student_data: StudentUpdate,
    request: Request,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Update student.
    Management only.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id
    ).first()
    
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Update fields
    if student_data.name is not None:
        student.full_name = student_data.name
    if student_data.email is not None:
        # Check if email already exists
        existing = db.query(User).filter(User.email == student_data.email, User.id != student_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        student.email = student_data.email
    if student_data.department is not None:
        student.department_name = student_data.department
        if student.college_id:
            dept = db.query(CollegeDepartment).filter(
                CollegeDepartment.college_id == student.college_id,
                CollegeDepartment.name == student_data.department,
            ).first()
            if dept:
                student.department_id = dept.id
    if getattr(student_data, "departmentId", None) is not None:
        dept = db.query(CollegeDepartment).filter(
            CollegeDepartment.id == student_data.departmentId,
            CollegeDepartment.college_id == student.college_id,
        ).first()
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Department not found or does not belong to the student's college.",
            )
        student.department_id = dept.id
        student.department_name = dept.name
    if student_data.year is not None:
        student.year = student_data.year
    if getattr(student_data, "rollNumber", None) is not None:
        student.roll_number = student_data.rollNumber
    if student_data.status is not None:
        try:
            status_enum = StudentStatus[student_data.status.upper()]
            student.status = status_enum
            student.is_active = (status_enum == StudentStatus.ACTIVE)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {student_data.status}"
            )
    
    db.commit()
    db.refresh(student)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="student_update",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(student_id)
    )
    
    dept_name = None
    if getattr(student, "department_id", None):
        dept = db.query(CollegeDepartment).filter(CollegeDepartment.id == student.department_id).first()
        dept_name = dept.name if dept else student.department_name
    return StudentResponse(
        id=student.id,
        studentId=getattr(student, "student_id", None),
        rollNumber=student.roll_number,
        name=student.full_name or "",
        email=student.email,
        department=dept_name or student.department_name,
        departmentId=getattr(student, "department_id", None),
        year=student.year,
        collegeId=student.college_id,
        status=student.status.value if student.status else "active",
        incidents=student.incidents or 0,
        lastIncident=student.last_incident_at.isoformat() if student.last_incident_at else None,
        faceRegistered=student.face_registered or False,
        createdAt=student.created_at.isoformat()
    )


@router.delete("/{student_id}")
async def delete_student(
    student_id: int,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Delete (deactivate) a student.
    Master admin only.
    """
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT
    ).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    student.status = UserStatus.INACTIVE
    student.is_active = False
    db.commit()
    AuditService.log_from_request(
        db=db,
        request=request,
        action="student_delete",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(student_id)
    )
    return {"success": True, "message": "Student deactivated"}


@router.post("/{student_id}/face")
async def register_face(
    student_id: int,
    file: UploadFile = File(...),
    request: Request = None,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Register/update face photo for student.
    Management only.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id
    ).first()
    
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed"
        )
    
    college_code = get_college_code_from_user(current_user)
    if not college_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="College code not found for upload"
        )
    student_id_str = getattr(student, "student_id", None) or student.college_student_id or student.roll_number or f"STU{student_id}"
    person_id = student_id_str
    s3_key = get_face_s3_path(college_code, person_id, f"face{file_ext}")
    
    try:
        content = await file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        s3_bucket = None
        if config.s3_client:
            file_url = config.s3_client.upload_file(
                tmp_path,
                s3_key,
                college_code=college_code,
                content_type=file.content_type or "image/jpeg"
            )
            s3_bucket = config.s3_client.get_bucket_name(college_code)
        elif getattr(config, "USE_S3_ONLY", True):
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 is required for file storage. Configure USE_S3 and S3 credentials (USE_S3_ONLY=true).",
            )
        else:
            local_dir = config.COLLEGE_FACES_DIR / person_id
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"face{file_ext}"
            with open(local_path, "wb") as f:
                f.write(content)
            file_url = str(local_path)
        
        os.unlink(tmp_path)
        student.face_image_url = file_url
        student.face_registered = True
        student.s3_bucket = s3_bucket
        student.s3_face_key = s3_key if s3_bucket else None
        # Link S3 to DB: upsert student_media (passport) so we can get data from S3 via DB
        existing = db.query(StudentMedia).filter(
            StudentMedia.user_id == student_id,
            StudentMedia.media_type == StudentMediaType.PASSPORT,
        ).first()
        if existing:
            existing.s3_bucket = s3_bucket
            existing.s3_key = s3_key if s3_bucket else None
            existing.file_url = file_url
            existing.filename = f"face{file_ext}"
        else:
            db.add(StudentMedia(
                user_id=student_id,
                media_type=StudentMediaType.PASSPORT,
                s3_bucket=s3_bucket,
                s3_key=s3_key if s3_bucket else None,
                file_url=file_url,
                filename=f"face{file_ext}",
                display_order=0,
            ))
        db.commit()
        
        # Log action
        AuditService.log_from_request(
            db=db,
            request=request,
            action="student_face_register",
            user_id=current_user.id,
            resource_type="user",
            resource_id=str(student_id)
        )
        
        return {"success": True, "message": "Face photo registered successfully"}
        
    except Exception as e:
        logger.error(f"Failed to register face photo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register face photo: {str(e)}"
        )


@router.get("/{student_id}/profile/passport", status_code=status.HTTP_200_OK)
async def get_student_passport(
    student_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Get student passport photo (from DB/S3). Returns viewable URL so the UI can show the image.
    """
    if current_user.role == UserRole.MANAGEMENT:
        college_id = get_management_college_id(current_user)
        if not college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not assigned to a college.")
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == college_id,
        ).first()
    elif current_user.role == UserRole.FACULTY:
        if not current_user.college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not assigned to a college.")
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == current_user.college_id,
            User.department_name == current_user.department_name,
        ).first()
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    college_code = student.college.college_code if student.college else None
    passport = db.query(StudentMedia).filter(
        StudentMedia.user_id == student_id,
        StudentMedia.media_type == StudentMediaType.PASSPORT,
    ).first()
    if not passport:
        return {"hasPassport": False, "url": None, "filename": None}
    url = _viewable_url_for_media(passport, college_code)
    return {"hasPassport": True, "url": url, "filename": passport.filename}


@router.post("/{student_id}/profile/passport", status_code=status.HTTP_200_OK)
async def upload_student_passport(
    student_id: int,
    file: UploadFile = File(...),
    request: Request = None,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session),
):
    """
    Upload student passport-size photo.
    Stored at: colleges/college_id=X/students/student_id=Y/profile/passport.jpg (campus-security bucket).
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college."
        )
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id,
    ).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    file_ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    if file_ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPEG/PNG allowed")

    college_code = student.college.college_code if student.college else None
    if not college_code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="College code not found")
    student_id_str = getattr(student, "student_id", None) or student.college_student_id or student.roll_number or f"STU{student_id}"

    if _use_campus_bucket():
        s3_key = campus_student_profile_passport_path(college_code, student_id_str)
    else:
        person_id = student_id_str
        s3_key = get_face_s3_path(college_code, person_id, f"passport{file_ext}")

    try:
        content = await file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        s3_bucket = None
        if config.s3_client:
            logger.info("Uploading passport photo to S3")
            file_url = config.s3_client.upload_file(
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
            logger.info("S3 disabled or not configured - storing passport photo locally")
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
        student.face_image_url = file_url
        student.face_registered = True
        student.s3_bucket = s3_bucket
        student.s3_face_key = s3_key if s3_bucket else None
        # Link to DB: upsert student_media row for passport so we can get data from S3 via DB
        existing = db.query(StudentMedia).filter(
            StudentMedia.user_id == student_id,
            StudentMedia.media_type == StudentMediaType.PASSPORT,
        ).first()
        if existing:
            existing.s3_bucket = s3_bucket
            existing.s3_key = s3_key if s3_bucket else None
            existing.file_url = file_url
            existing.filename = "passport.jpg"
        else:
            db.add(StudentMedia(
                user_id=student_id,
                media_type=StudentMediaType.PASSPORT,
                s3_bucket=s3_bucket,
                s3_key=s3_key if s3_bucket else None,
                file_url=file_url,
                filename="passport.jpg",
                display_order=0,
            ))
        db.commit()
        AuditService.log_from_request(db=db, request=request, action="student_passport_upload", user_id=current_user.id, resource_type="user", resource_id=str(student_id))
        # Auto-create/update embeddings/face_embedding.npy in background (no app restart)
        person_id = student_id_str
        name = getattr(student, "name", None) or student_id_str
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
        logger.error(f"Passport upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{student_id}/face-gallery", status_code=status.HTTP_200_OK)
async def get_student_face_gallery(
    student_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    List student face-gallery images (from DB/S3). Returns viewable URLs so the UI can show images.
    """
    if current_user.role == UserRole.MANAGEMENT:
        college_id = get_management_college_id(current_user)
        if not college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not assigned to a college.")
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == college_id,
        ).first()
    elif current_user.role == UserRole.FACULTY:
        if not current_user.college_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not assigned to a college.")
        student = db.query(User).filter(
            User.id == student_id,
            User.role == UserRole.STUDENT,
            User.college_id == current_user.college_id,
            User.department_name == current_user.department_name,
        ).first()
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    college_code = student.college.college_code if student.college else None
    items = (
        db.query(StudentMedia)
        .filter(
            StudentMedia.user_id == student_id,
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


@router.post("/{student_id}/face-gallery", status_code=status.HTTP_200_OK)
async def upload_student_face_gallery(
    student_id: int,
    files: List[UploadFile] = File(...),
    request: Request = None,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session),
):
    """
    Upload 360-degree / multiple face photos for a student.
    Stored at: colleges/college_id=X/students/student_id=Y/face-gallery/img_01.jpg, img_02.jpg, ... (campus-security bucket).
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account is not assigned to a college.")
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id,
    ).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file required")

    college_code = student.college.college_code if student.college else None
    if not college_code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="College code not found")
    student_id_str = getattr(student, "student_id", None) or student.college_student_id or student.roll_number or f"STU{student_id}"

    uploaded = []
    image_bytes_for_sync = []  # collect for auto face_embedding.npy sync
    for idx, file in enumerate(files):
        ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        display_name = f"img_{str(idx + 1).zfill(2)}{ext}"
        if _use_campus_bucket():
            s3_key = campus_student_face_gallery_path(college_code, student_id_str, display_name)
        else:
            person_id = student_id_str
            s3_key = get_face_s3_path(college_code, person_id, f"gallery_{idx + 1}{ext}")
            display_name = s3_key.split("/")[-1]
        try:
            content = await file.read()
            image_bytes_for_sync.append(content)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            s3_bucket = None
            if config.s3_client:
                logger.debug("Uploading face-gallery image to S3: %s", display_name)
                url = config.s3_client.upload_file(tmp_path, s3_key, college_code=college_code, content_type=file.content_type or "image/jpeg")
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
                logger.info("S3 disabled or not configured - storing face-gallery image locally")
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
            db.add(StudentMedia(
                user_id=student_id,
                media_type=StudentMediaType.FACE_GALLERY,
                s3_bucket=s3_bucket,
                s3_key=s3_key if s3_bucket else None,
                file_url=url,
                filename=display_name,
                display_order=idx + 1,
            ))
            uploaded.append({"filename": display_name, "url": url})
        except Exception as e:
            logger.warning(f"Face gallery file {idx + 1} upload failed: {e}")
    if not uploaded:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid image could be uploaded")
    db.commit()
    AuditService.log_from_request(db=db, request=request, action="student_face_gallery_upload", user_id=current_user.id, resource_type="user", resource_id=str(student_id))
    # Auto-create/update embeddings/face_embedding.npy in background (no app restart)
    person_id = student_id_str
    name = getattr(student, "name", None) or student_id_str
    if image_bytes_for_sync and getattr(config, "face_db", None):
        asyncio.create_task(
            asyncio.to_thread(
                run_student_embedding_sync,
                person_id,
                name,
                college_code,
                image_bytes_for_sync,
            )
        )
    return {"success": True, "message": f"{len(uploaded)} photo(s) uploaded", "uploaded": uploaded}


@router.post("/{student_id}/sync-embeddings", status_code=status.HTTP_202_ACCEPTED)
async def sync_student_embeddings(
    student_id: int,
    request: Request = None,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session),
):
    """
    Manually re-run face embedding extraction from stored passport + face-gallery
    and update embeddings/face_embedding.npy in S3. Use when auto-sync was skipped
    or to refresh after editing photos. Runs in background.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account is not assigned to a college.")
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id,
    ).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    college_code = student.college.college_code if student.college else None
    if not college_code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="College code not found")
    student_id_str = getattr(student, "student_id", None) or student.college_student_id or student.roll_number or f"STU{student_id}"
    person_id = student_id_str
    name = getattr(student, "name", None) or student_id_str

    media_list = (
        db.query(StudentMedia)
        .filter(
            StudentMedia.user_id == student_id,
            StudentMedia.media_type.in_([StudentMediaType.PASSPORT, StudentMediaType.FACE_GALLERY]),
        )
        .order_by(StudentMedia.media_type.asc(), StudentMedia.display_order.asc())
        .all()
    )
    image_bytes_list = []
    for m in media_list:
        try:
            if getattr(m, "s3_key", None) and config.s3_client and college_code:
                buf = config.s3_client.download_fileobj(m.s3_key, college_code=college_code)
                image_bytes_list.append(buf.read())
            elif getattr(m, "file_url", None) and not (m.file_url or "").startswith("s3://"):
                if os.path.isfile(m.file_url):
                    with open(m.file_url, "rb") as f:
                        image_bytes_list.append(f.read())
        except Exception as e:
            logger.warning(f"Could not load media for sync-embeddings: {e}")
    if not image_bytes_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No passport or face-gallery images found for this student. Upload photos first.",
        )
    if not getattr(config, "face_db", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Face database not initialized.")

    asyncio.create_task(
        asyncio.to_thread(
            run_student_embedding_sync,
            person_id,
            name,
            college_code,
            image_bytes_list,
        )
    )
    AuditService.log_from_request(
        db=db, request=request, action="student_sync_embeddings", user_id=current_user.id,
        resource_type="user", resource_id=str(student_id),
    )
    return {
        "message": "Embedding sync started. Face embeddings will be extracted from passport and face-gallery and uploaded to S3 (embeddings/face_embedding.npy).",
        "studentId": student_id,
    }


@router.post("/{student_id}/action")
async def issue_action(
    student_id: int,
    action_data: StudentActionRequest,
    request: Request,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Issue warning / suspend student.
    Management only.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id
    ).first()
    
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Validate action type
    try:
        action_type_enum = ActionType[action_data.actionType.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action type: {action_data.actionType}"
        )
    
    # Create disciplinary action
    action = DisciplinaryAction(
        report_id=action_data.reportId,
        student_id=student_id,
        action_type=action_type_enum,
        notes=action_data.notes,
        created_by=current_user.id
    )
    db.add(action)
    
    # Update student status based on action
    if action_type_enum in [ActionType.SUSPENSION_1D, ActionType.SUSPENSION_3D, ActionType.SUSPENSION_7D]:
        student.status = StudentStatus.SUSPENDED
    elif action_type_enum == ActionType.WARNING:
        student.status = StudentStatus.WARNING
    
    # Update student incidents count
    student.incidents = (student.incidents or 0) + 1
    student.last_incident_at = datetime.utcnow()
    
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="student_action",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(student_id)
    )
    
    return {"success": True, "message": "Action issued successfully"}


@router.get("/{student_id}/incidents")
async def get_student_incidents(
    student_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    List incidents (reports) involving student.
    Management only.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    student = db.query(User).filter(
        User.id == student_id,
        User.role == UserRole.STUDENT,
        User.college_id == college_id
    ).first()
    
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Get reports where student was detected
    from database.models import DetectedFace
    detected_faces = db.query(DetectedFace).filter(DetectedFace.student_id == student_id).all()
    report_ids = [df.report_id for df in detected_faces]
    
    # Get reports
    query = db.query(Report).filter(Report.id.in_(report_ids))
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    reports = query.order_by(Report.created_at.desc()).offset(offset).limit(limit).all()
    
    # Build response
    report_list = []
    for report in reports:
        report_dict = {
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "occurredAt": report.occurred_at.isoformat(),
            "status": report.status.value,
            "createdAt": report.created_at.isoformat()
        }
        report_list.append(report_dict)
    
    return {
        "data": report_list,
        "total": total,
        "page": page,
        "limit": limit
    }
