"""
Faculty Management APIs (Management).
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr
from typing import Optional, List

from database.connection import Database
from database.models import User, UserRole, UserStatus, FacultyStatus
from auth.dependencies import get_current_user, require_management, require_master_admin, get_management_college_id
from services.auth_service import AuthService
from services.audit_service import AuditService
from core.logger import logger
import config


router = APIRouter(prefix="/api/faculty", tags=["faculty"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


# Request/Response Models
class FacultyUpdate(BaseModel):
    """Update faculty request."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    department: Optional[str] = None
    status: Optional[str] = None


class FacultyImportRequest(BaseModel):
    """Bulk import faculty request."""
    faculty: List[dict]  # [{ id, name, email, department? }]


class FacultyResponse(BaseModel):
    """Faculty response model."""
    id: int
    name: str
    email: Optional[str]
    department: Optional[str]
    collegeId: int
    reportsSubmitted: int
    reportsResolved: int
    status: str
    createdAt: str


class FacultyListResponse(BaseModel):
    """Faculty list response."""
    data: List[dict]
    total: int
    page: int
    limit: int


class ImportResponse(BaseModel):
    """Import response."""
    imported: int
    errors: Optional[List[dict]] = None


@router.get("", response_model=FacultyListResponse)
async def list_faculty(
    department: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    List faculty (management's college).
    Management only.
    """
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    query = db.query(User).filter(
        User.role == UserRole.FACULTY,
        User.college_id == college_id
    )
    
    # Filter by department
    if department:
        query = query.filter(User.department_name == department)
    
    # Filter by status
    if status_filter:
        try:
            status_enum = FacultyStatus[status_filter.upper()]
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
                User.faculty_id.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    faculty_list = query.offset(offset).limit(limit).all()
    
    # Build response
    faculty_data = []
    for faculty in faculty_list:
        faculty_dict = {
            "id": faculty.id,
            "name": faculty.full_name,
            "email": faculty.email,
            "department": faculty.department_name,
            "collegeId": faculty.college_id,
            "reportsSubmitted": faculty.reports_submitted or 0,
            "reportsResolved": faculty.reports_resolved or 0,
            "status": faculty.status.value if faculty.status else "active",
            "createdAt": faculty.created_at.isoformat()
        }
        faculty_data.append(faculty_dict)
    
    return FacultyListResponse(
        data=faculty_data,
        total=total,
        page=page,
        limit=limit
    )


@router.get("/{faculty_id}", response_model=FacultyResponse)
async def get_faculty(
    faculty_id: int,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Get faculty by ID.
    Management only.
    """
    if not get_management_college_id(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    faculty = db.query(User).filter(
        User.id == faculty_id,
        User.role == UserRole.FACULTY,
        User.college_id == get_management_college_id(current_user)
    ).first()
    
    if not faculty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Faculty not found"
        )
    
    return FacultyResponse(
        id=faculty.id,
        name=faculty.full_name or "",
        email=faculty.email,
        department=faculty.department_name,
        collegeId=faculty.college_id,
        reportsSubmitted=faculty.reports_submitted or 0,
        reportsResolved=faculty.reports_resolved or 0,
        status=faculty.status.value if faculty.status else "active",
        createdAt=faculty.created_at.isoformat()
    )


@router.post("/import", response_model=ImportResponse)
async def import_faculty(
    import_data: FacultyImportRequest,
    request: Request,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Bulk import faculty from CSV data.
    Management only.
    """
    if not get_management_college_id(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    imported_count = 0
    errors = []
    
    for idx, faculty_data in enumerate(import_data.faculty):
        try:
            # Validate required fields
            if not faculty_data.get("name") or not faculty_data.get("email"):
                errors.append({
                    "row": idx + 1,
                    "message": "Name and email are required"
                })
                continue
            
            # Check if faculty already exists
            existing = db.query(User).filter(
                User.email == faculty_data["email"],
                User.college_id == get_management_college_id(current_user)
            ).first()
            
            if existing:
                errors.append({
                    "row": idx + 1,
                    "message": f"Faculty with email {faculty_data['email']} already exists"
                })
                continue
            
            # Create faculty without password; user will set password via OTP flow
            faculty = AuthService.create_user(
                db=db,
                login_id=faculty_data.get("id") or faculty_data["email"],
                role=UserRole.FACULTY,
                email=faculty_data["email"],
                full_name=faculty_data["name"],
                college_id=get_management_college_id(current_user),
            )
            
            # Set faculty-specific fields
            faculty.department_name = faculty_data.get("department")
            faculty.faculty_id = faculty_data.get("id")
            
            db.commit()
            imported_count += 1
            
        except Exception as e:
            errors.append({
                "row": idx + 1,
                "message": str(e)
            })
            logger.error(f"Error importing faculty row {idx + 1}: {e}", exc_info=True)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="faculty_import",
        user_id=current_user.id,
        resource_type="user"
    )
    
    return ImportResponse(
        imported=imported_count,
        errors=errors if errors else None
    )


@router.put("/{faculty_id}", response_model=FacultyResponse)
async def update_faculty(
    faculty_id: int,
    faculty_data: FacultyUpdate,
    request: Request,
    current_user: User = Depends(require_management),
    db: Session = Depends(get_db_session)
):
    """
    Update faculty.
    Management only.
    """
    if not get_management_college_id(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    faculty = db.query(User).filter(
        User.id == faculty_id,
        User.role == UserRole.FACULTY,
        User.college_id == get_management_college_id(current_user)
    ).first()
    
    if not faculty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Faculty not found"
        )
    
    # Update fields
    if faculty_data.name is not None:
        faculty.full_name = faculty_data.name
    if faculty_data.email is not None:
        # Check if email already exists
        existing = db.query(User).filter(User.email == faculty_data.email, User.id != faculty_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        faculty.email = faculty_data.email
    if faculty_data.department is not None:
        faculty.department_name = faculty_data.department
    if faculty_data.status is not None:
        try:
            status_enum = FacultyStatus[faculty_data.status.upper()]
            faculty.status = status_enum
            faculty.is_active = (status_enum == FacultyStatus.ACTIVE)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {faculty_data.status}"
            )
    
    db.commit()
    db.refresh(faculty)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="faculty_update",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(faculty_id)
    )
    
    return FacultyResponse(
        id=faculty.id,
        name=faculty.full_name or "",
        email=faculty.email,
        department=faculty.department_name,
        collegeId=faculty.college_id,
        reportsSubmitted=faculty.reports_submitted or 0,
        reportsResolved=faculty.reports_resolved or 0,
        status=faculty.status.value if faculty.status else "active",
        createdAt=faculty.created_at.isoformat()
    )


@router.delete("/{faculty_id}")
async def delete_faculty(
    faculty_id: int,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Delete (deactivate) a faculty member.
    Master admin only.
    """
    faculty = db.query(User).filter(
        User.id == faculty_id,
        User.role == UserRole.FACULTY
    ).first()
    if not faculty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Faculty not found"
        )
    faculty.status = UserStatus.INACTIVE
    faculty.is_active = False
    db.commit()
    AuditService.log_from_request(
        db=db,
        request=request,
        action="faculty_delete",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(faculty_id)
    )
    return {"success": True, "message": "Faculty deactivated"}
