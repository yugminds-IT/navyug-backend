"""
College Management APIs.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, EmailStr
from typing import Optional, List

from database.connection import Database
from database.models import User, UserRole, College, CollegeStatus, CollegeDepartment, Report
from auth.dependencies import get_current_user, require_master_admin, get_management_college_id
from services.audit_service import AuditService
from core.logger import logger
import config


router = APIRouter(prefix="/api/colleges", tags=["colleges"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


def _get_college_live_counts(db: Session, college_id: int) -> tuple:
    """Return (total_students, total_faculty, total_reports) for a college from live data."""
    total_students = db.query(User).filter(
        User.role == UserRole.STUDENT,
        User.college_id == college_id,
        User.is_active == True
    ).count()
    total_faculty = db.query(User).filter(
        User.role == UserRole.FACULTY,
        User.college_id == college_id,
        User.is_active == True
    ).count()
    total_reports = db.query(Report).filter(Report.college_id == college_id).count()
    return total_students, total_faculty, total_reports


# Request/Response Models
class CollegeCreate(BaseModel):
    """Create college request."""
    name: str
    code: str
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    departments: Optional[List[dict]] = None


class CollegeUpdate(BaseModel):
    """Update college request."""
    name: Optional[str] = None
    code: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    departments: Optional[List[dict]] = None


class CollegeStatusUpdate(BaseModel):
    """Update college status request."""
    status: str


class CollegeResponse(BaseModel):
    """College response model."""
    id: int
    name: str
    code: str
    address: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    website: Optional[str]
    status: str
    departments: List[dict]
    totalStudents: int
    totalFaculty: int
    totalReports: int
    createdAt: str
    updatedAt: str


class CollegeListResponse(BaseModel):
    """College list response."""
    data: List[dict]
    total: int


class DepartmentCreate(BaseModel):
    """Create department request."""
    name: str


class DepartmentUpdate(BaseModel):
    """Update department request."""
    name: str


class DepartmentCollegeRef(BaseModel):
    """College reference in department response."""
    id: int
    name: str
    code: str


class DepartmentResponse(BaseModel):
    """Department response model with college."""
    id: int
    name: str
    college: DepartmentCollegeRef


@router.get("", response_model=CollegeListResponse)
async def list_colleges(
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List colleges.
    Master: all colleges. Management/Faculty/Student: own college only.
    """
    query = db.query(College)
    
    # Role-based filtering
    if current_user.role == UserRole.MASTER_ADMIN:
        pass  # Can see all
    elif current_user.role == UserRole.MANAGEMENT:
        my_college_id = get_management_college_id(current_user)
        if not my_college_id:
            query = query.filter(College.id == -1)  # No college assigned: empty list
        else:
            query = query.filter(College.id == my_college_id)
    elif current_user.role in (UserRole.FACULTY, UserRole.STUDENT):
        if current_user.college_id:
            query = query.filter(College.id == current_user.college_id)
        else:
            query = query.filter(College.id == -1)  # No college: empty list
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Filter by status
    if status_filter:
        try:
            status_enum = CollegeStatus[status_filter.upper()]
            query = query.filter(College.status == status_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )
    
    # Search
    if search:
        query = query.filter(
            College.name.ilike(f"%{search}%") | College.college_code.ilike(f"%{search}%")
        )
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    colleges = query.offset(offset).limit(limit).all()
    
    # Build response (totals from live counts so they match faculty/students/reports)
    college_list = []
    for college in colleges:
        # Get departments
        departments = db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college.id).all()
        dept_list = [{"id": d.id, "name": d.name} for d in departments]
        total_students, total_faculty, total_reports = _get_college_live_counts(db, college.id)
        college_dict = {
            "id": college.id,
            "name": college.name,
            "code": college.college_code,
            "address": college.address,
            "phone": college.contact_phone,
            "email": college.contact_email,
            "website": college.website,
            "status": college.status.value if college.status else "active",
            "departments": dept_list,
            "totalStudents": total_students,
            "totalFaculty": total_faculty,
            "totalReports": total_reports,
            "createdAt": college.created_at.isoformat(),
            "updatedAt": college.updated_at.isoformat()
        }
        college_list.append(college_dict)
    
    return CollegeListResponse(
        data=college_list,
        total=total
    )


@router.get("/{college_id}", response_model=CollegeResponse)
async def get_college(
    college_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get college by ID.
    Master and Management (own college only).
    """
    college = db.query(College).filter(College.id == college_id).first()
    if not college:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="College not found"
        )
    
    # Check authorization: master sees all; management/faculty/student only their own college
    if current_user.role == UserRole.MASTER_ADMIN:
        pass  # Can access all
    elif current_user.role == UserRole.MANAGEMENT:
        my_college_id = get_management_college_id(current_user)
        if my_college_id is None or college.id != my_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only access your own college."
            )
    elif current_user.role in (UserRole.FACULTY, UserRole.STUDENT):
        if current_user.college_id != college.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only access your own college."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get departments and live counts
    departments = db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college.id).all()
    dept_list = [{"id": d.id, "name": d.name} for d in departments]
    total_students, total_faculty, total_reports = _get_college_live_counts(db, college.id)
    
    return CollegeResponse(
        id=college.id,
        name=college.name,
        code=college.college_code,
        address=college.address,
        phone=college.contact_phone,
        email=college.contact_email,
        website=college.website,
        status=college.status.value if college.status else "active",
        departments=dept_list,
        totalStudents=total_students,
        totalFaculty=total_faculty,
        totalReports=total_reports,
        createdAt=college.created_at.isoformat(),
        updatedAt=college.updated_at.isoformat()
    )


@router.post("", response_model=CollegeResponse, status_code=status.HTTP_201_CREATED)
async def create_college(
    college_data: CollegeCreate,
    request: Request,
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Create college.
    Master admin only.
    """
    # Check if code already exists
    existing = db.query(College).filter(College.college_code == college_data.code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="College with this code already exists"
        )
    
    # Create college
    college = College(
        college_code=college_data.code,
        name=college_data.name,
        address=college_data.address,
        contact_phone=college_data.phone,
        contact_email=college_data.email,
        website=college_data.website,
        status=CollegeStatus.ACTIVE,
        created_by=current_user.id
    )
    db.add(college)
    db.flush()  # Get college.id
    
    # Create departments
    if college_data.departments:
        for dept_data in college_data.departments:
            dept = CollegeDepartment(
                college_id=college.id,
                name=dept_data.get("name", "")
            )
            db.add(dept)
    
    db.commit()
    db.refresh(college)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="college_create",
        user_id=current_user.id,
        resource_type="college",
        resource_id=str(college.id)
    )
    
    # Get departments
    departments = db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college.id).all()
    dept_list = [{"id": d.id, "name": d.name} for d in departments]
    
    return CollegeResponse(
        id=college.id,
        name=college.name,
        code=college.college_code,
        address=college.address,
        phone=college.contact_phone,
        email=college.contact_email,
        website=college.website,
        status=college.status.value,
        departments=dept_list,
        totalStudents=0,
        totalFaculty=0,
        totalReports=0,
        createdAt=college.created_at.isoformat(),
        updatedAt=college.updated_at.isoformat()
    )


@router.put("/{college_id}", response_model=CollegeResponse)
async def update_college(
    college_id: int,
    college_data: CollegeUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update college.
    Master admin: any college. Management: own college only.
    """
    college = db.query(College).filter(College.id == college_id).first()
    if not college:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="College not found"
        )
    
    # Authorization: master can update any; management only their own college
    if current_user.role == UserRole.MASTER_ADMIN:
        pass
    elif current_user.role == UserRole.MANAGEMENT:
        my_college_id = get_management_college_id(current_user)
        if my_college_id is None or college.id != my_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only edit your own college."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update fields
    if college_data.name is not None:
        college.name = college_data.name
    if college_data.code is not None:
        # Check if code already exists
        existing = db.query(College).filter(College.college_code == college_data.code, College.id != college_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="College with this code already exists"
            )
        college.college_code = college_data.code
    if college_data.address is not None:
        college.address = college_data.address
    if college_data.phone is not None:
        college.contact_phone = college_data.phone
    if college_data.email is not None:
        college.contact_email = college_data.email
    if college_data.website is not None:
        college.website = college_data.website
    
    # Update departments
    if college_data.departments is not None:
        # Delete existing departments
        db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college_id).delete()
        
        # Create new departments
        for dept_data in college_data.departments:
            dept = CollegeDepartment(
                college_id=college_id,
                name=dept_data.get("name", "")
            )
            db.add(dept)
    
    db.commit()
    db.refresh(college)
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="college_update",
        user_id=current_user.id,
        resource_type="college",
        resource_id=str(college_id)
    )
    
    # Get departments and live counts
    departments = db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college.id).all()
    dept_list = [{"id": d.id, "name": d.name} for d in departments]
    total_students, total_faculty, total_reports = _get_college_live_counts(db, college.id)
    
    return CollegeResponse(
        id=college.id,
        name=college.name,
        code=college.college_code,
        address=college.address,
        phone=college.contact_phone,
        email=college.contact_email,
        website=college.website,
        status=college.status.value,
        departments=dept_list,
        totalStudents=total_students,
        totalFaculty=total_faculty,
        totalReports=total_reports,
        createdAt=college.created_at.isoformat(),
        updatedAt=college.updated_at.isoformat()
    )


@router.patch("/{college_id}/status")
async def update_college_status(
    college_id: int,
    status_data: CollegeStatusUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Activate/deactivate college.
    Master admin: any college. Management: own college only.
    """
    college = db.query(College).filter(College.id == college_id).first()
    if not college:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="College not found"
        )
    
    # Authorization: master can update any; management only their own college
    if current_user.role == UserRole.MASTER_ADMIN:
        pass
    elif current_user.role == UserRole.MANAGEMENT:
        my_college_id = get_management_college_id(current_user)
        if my_college_id is None or college.id != my_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only update your own college status."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Validate status
    try:
        status_enum = CollegeStatus[status_data.status.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {status_data.status}"
        )
    
    # Update status
    college.status = status_enum
    college.is_active = (status_enum == CollegeStatus.ACTIVE)
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="college_status_update",
        user_id=current_user.id,
        resource_type="college",
        resource_id=str(college_id)
    )
    
    # Get departments and live counts
    departments = db.query(CollegeDepartment).filter(CollegeDepartment.college_id == college.id).all()
    dept_list = [{"id": d.id, "name": d.name} for d in departments]
    total_students, total_faculty, total_reports = _get_college_live_counts(db, college.id)
    
    return CollegeResponse(
        id=college.id,
        name=college.name,
        code=college.college_code,
        address=college.address,
        phone=college.contact_phone,
        email=college.contact_email,
        website=college.website,
        status=college.status.value,
        departments=dept_list,
        totalStudents=total_students,
        totalFaculty=total_faculty,
        totalReports=total_reports,
        createdAt=college.created_at.isoformat(),
        updatedAt=college.updated_at.isoformat()
    )


def _check_college_access(current_user: User, college_id: int, db: Session) -> College:
    """Verify college exists and current user has access. Returns college or raises HTTPException."""
    college = db.query(College).filter(College.id == college_id).first()
    if not college:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="College not found"
        )
    if current_user.role == UserRole.MASTER_ADMIN:
        return college
    if current_user.role == UserRole.MANAGEMENT:
        my_college_id = get_management_college_id(current_user)
        if my_college_id is None or college.id != my_college_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this college"
            )
        return college
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied"
    )


@router.post("/{college_id}/departments", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED)
async def create_department(
    college_id: int,
    body: DepartmentCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Add a department to a college.
    Body: { "name": "Department Name" }
    Master admin: any college. Management: can add departments to their own college only.
    """
    _check_college_access(current_user, college_id, db)
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Department name is required"
        )
    existing = db.query(CollegeDepartment).filter(
        CollegeDepartment.college_id == college_id,
        CollegeDepartment.name == name
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A department with this name already exists for this college"
        )
    dept = CollegeDepartment(college_id=college_id, name=name)
    db.add(dept)
    db.commit()
    db.refresh(dept)
    AuditService.log_from_request(
        db=db,
        request=request,
        action="department_create",
        user_id=current_user.id,
        resource_type="college_department",
        resource_id=str(dept.id)
    )
    college = db.query(College).filter(College.id == college_id).first()
    return DepartmentResponse(
        id=dept.id,
        name=dept.name,
        college=DepartmentCollegeRef(id=college.id, name=college.name, code=college.college_code)
    )


@router.put("/{college_id}/departments/{department_id}", response_model=DepartmentResponse)
async def update_department(
    college_id: int,
    department_id: int,
    body: DepartmentUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update a department.
    Body: { "name": string }
    Master admin: any college. Management: own college only.
    """
    _check_college_access(current_user, college_id, db)
    dept = db.query(CollegeDepartment).filter(
        CollegeDepartment.id == department_id,
        CollegeDepartment.college_id == college_id
    ).first()
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Department not found"
        )
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Department name is required"
        )
    existing = db.query(CollegeDepartment).filter(
        CollegeDepartment.college_id == college_id,
        CollegeDepartment.name == name,
        CollegeDepartment.id != department_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A department with this name already exists for this college"
        )
    dept.name = name
    db.commit()
    db.refresh(dept)
    AuditService.log_from_request(
        db=db,
        request=request,
        action="department_update",
        user_id=current_user.id,
        resource_type="college_department",
        resource_id=str(dept.id)
    )
    college = db.query(College).filter(College.id == college_id).first()
    return DepartmentResponse(
        id=dept.id,
        name=dept.name,
        college=DepartmentCollegeRef(id=college.id, name=college.name, code=college.college_code)
    )


@router.delete("/{college_id}/departments/{department_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_department(
    college_id: int,
    department_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete a department.
    Master admin: any college. Management: own college only.
    """
    _check_college_access(current_user, college_id, db)
    dept = db.query(CollegeDepartment).filter(
        CollegeDepartment.id == department_id,
        CollegeDepartment.college_id == college_id
    ).first()
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Department not found"
        )
    db.delete(dept)
    db.commit()
    AuditService.log_from_request(
        db=db,
        request=request,
        action="department_delete",
        user_id=current_user.id,
        resource_type="college_department",
        resource_id=str(department_id)
    )
    return None
