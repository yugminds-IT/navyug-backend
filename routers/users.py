"""
User Management APIs (Master Admin).
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

from database.connection import Database
from database.models import User, UserRole, UserStatus
from auth.dependencies import get_current_user, require_master_admin
from services.auth_service import AuthService
from services.audit_service import AuditService
from services.student_id import generate_student_id
from core.logger import logger
import config


router = APIRouter(prefix="/api/users", tags=["users"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


# Request/Response Models
class UserCreate(BaseModel):
    """Create user request."""
    name: str
    email: EmailStr
    role: str
    collegeId: Optional[int] = None
    department: Optional[str] = None
    year: Optional[str] = None
    rollNumber: Optional[str] = None


class UserUpdate(BaseModel):
    """Update user request."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    collegeId: Optional[int] = None
    managedCollegeId: Optional[int] = None  # For management users: college they manage
    department: Optional[str] = None
    year: Optional[str] = None
    rollNumber: Optional[str] = None
    status: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    """Reset password request."""
    newPassword: str


class UserResponse(BaseModel):
    """User response model."""
    id: int
    email: Optional[str]
    name: Optional[str]
    role: str
    status: str
    collegeId: Optional[int]
    department: Optional[str]
    year: Optional[str]
    rollNumber: Optional[str]
    createdAt: str
    updatedAt: str


class UserListResponse(BaseModel):
    """User list response."""
    data: List[dict]
    total: int
    page: int
    limit: int


class UserStatsResponse(BaseModel):
    """User stats response."""
    management: int
    faculty: int
    students: int


@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Get user counts by role.
    Master admin only.
    """
    management_count = db.query(func.count(User.id)).filter(User.role == UserRole.MANAGEMENT).scalar() or 0
    faculty_count = db.query(func.count(User.id)).filter(User.role == UserRole.FACULTY).scalar() or 0
    students_count = db.query(func.count(User.id)).filter(User.role == UserRole.STUDENT).scalar() or 0
    
    return UserStatsResponse(
        management=management_count,
        faculty=faculty_count,
        students=students_count
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    List users by role (paginated, filterable).
    Master admin only.
    """
    query = db.query(User)
    
    # Filter by role
    if role:
        try:
            role_enum = UserRole[role.upper()]
            query = query.filter(User.role == role_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role}"
            )
    
    # Filter by status
    if status_filter:
        try:
            status_enum = UserStatus[status_filter.upper()]
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
    users = query.offset(offset).limit(limit).all()
    
    # Build response
    user_list = []
    for user in users:
        user_dict = {
            "id": user.id,
            "email": user.email,
            "name": user.full_name,
            "role": user.role.value,
            "status": user.status.value if user.status else "active",
            "collegeId": user.college_id,
            "department": user.department_name,
            "year": user.year,
            "rollNumber": user.roll_number,
            "createdAt": user.created_at.isoformat(),
            "updatedAt": user.updated_at.isoformat()
        }
        user_list.append(user_dict)
    
    return UserListResponse(
        data=user_list,
        total=total,
        page=page,
        limit=limit
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Get user by ID.
    Master admin only.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.full_name,
        role=user.role.value,
        status=user.status.value if user.status else "active",
        collegeId=user.college_id,
        department=user.department_name,
        year=user.year,
        rollNumber=user.roll_number,
        createdAt=user.created_at.isoformat(),
        updatedAt=user.updated_at.isoformat()
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Create user (management / faculty / student).
    Master admin only.
    """
    # Validate role
    try:
        role_enum = UserRole[user_data.role.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {user_data.role}"
        )
    
    # Check if email already exists
    if user_data.email:
        existing = db.query(User).filter(User.email == user_data.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
    
    # Check if roll_number already exists (for students)
    if user_data.rollNumber:
        existing = db.query(User).filter(User.roll_number == user_data.rollNumber).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this roll number already exists"
            )
    
    # Create user without password; user will set password via OTP flow (send-otp → verify-otp → setup-password)
    try:
        # For students: generate unique student_id before create (roll_number is separate)
        generated_student_id = None
        if role_enum == UserRole.STUDENT and user_data.collegeId:
            generated_student_id = generate_student_id(db, user_data.collegeId)

        user = AuthService.create_user(
            db=db,
            login_id=user_data.email or user_data.rollNumber or f"user_{datetime.utcnow().timestamp()}",
            role=role_enum,
            email=user_data.email,
            full_name=user_data.name,
            college_id=user_data.collegeId,
            college_student_id=generated_student_id if role_enum == UserRole.STUDENT else None,
        )
        
        # Set role-specific fields
        if role_enum == UserRole.STUDENT:
            user.roll_number = user_data.rollNumber  # Academic roll number (input only)
            user.year = user_data.year
            if generated_student_id:
                user.student_id = generated_student_id
                user.college_student_id = generated_student_id
        elif role_enum == UserRole.FACULTY:
            user.department_name = user_data.department
        
        db.commit()
        db.refresh(user)
        
        # Log action
        AuditService.log_from_request(
            db=db,
            request=request,
            action="user_create",
            user_id=current_user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return UserResponse(
            id=user.id,
            email=user.email,
            name=user.full_name,
            role=user.role.value,
            status=user.status.value if user.status else "active",
            collegeId=user.college_id,
            department=user.department_name,
            year=user.year,
            rollNumber=user.roll_number,
            createdAt=user.created_at.isoformat(),
            updatedAt=user.updated_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Update user.
    Master admin only.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    if user_data.name is not None:
        user.full_name = user_data.name
    if user_data.email is not None:
        # Check if email already exists
        existing = db.query(User).filter(User.email == user_data.email, User.id != user_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        user.email = user_data.email
    if user_data.role is not None:
        try:
            role_enum = UserRole[user_data.role.upper()]
            user.role = role_enum
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {user_data.role}"
            )
    if user_data.collegeId is not None:
        user.college_id = user_data.collegeId
    if user_data.managedCollegeId is not None:
        # Master admin can assign which college a management user manages
        user.managed_college_id = user_data.managedCollegeId
    if user_data.department is not None:
        user.department_name = user_data.department
    if user_data.year is not None:
        user.year = user_data.year
    if user_data.rollNumber is not None:
        # Check if roll_number already exists
        existing = db.query(User).filter(User.roll_number == user_data.rollNumber, User.id != user_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this roll number already exists"
            )
        user.roll_number = user_data.rollNumber
    if user_data.status is not None:
        try:
            status_enum = UserStatus[user_data.status.upper()]
            user.status = status_enum
            user.is_active = (status_enum == UserStatus.ACTIVE)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {user_data.status}"
            )
    
    db.commit()
    db.refresh(user)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="user_update",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id)
    )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.full_name,
        role=user.role.value,
        status=user.status.value if user.status else "active",
        collegeId=user.college_id,
        department=user.department_name,
        year=user.year,
        rollNumber=user.roll_number,
        createdAt=user.created_at.isoformat(),
        updatedAt=user.updated_at.isoformat()
    )


@router.delete("/management/{user_id}")
async def delete_management(
    user_id: int,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Soft-deactivate a management user.
    Master admin only.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if user.role != UserRole.MANAGEMENT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a management user. Use DELETE /api/users/{user_id} for other roles."
        )
    user.status = UserStatus.INACTIVE
    user.is_active = False
    db.commit()
    AuditService.log_from_request(
        db=db,
        request=request,
        action="management_delete",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id)
    )
    return {"success": True, "message": "Management user deactivated"}


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Soft-deactivate user (any role: student, faculty, management).
    Master admin only.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Soft delete: set status to inactive
    user.status = UserStatus.INACTIVE
    user.is_active = False
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="user_deactivate",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id)
    )
    
    return {"success": True}


@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    password_data: ResetPasswordRequest,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Admin reset user password.
    Master admin only.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate password
    from auth.security import validate_password
    is_valid, error_message = validate_password(password_data.newPassword)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Reset password
    from auth.security import get_password_hash
    user.hashed_password = get_password_hash(password_data.newPassword)
    user.password_changed_at = datetime.utcnow()
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="admin_reset_password",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id)
    )
    
    return {"success": True}
