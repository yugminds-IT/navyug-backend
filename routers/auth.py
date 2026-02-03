"""
Enhanced authentication endpoints with 4 user roles and advanced security.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

from database.connection import Database
from database.models import User, UserRole, College, OtpStore, UserStatus
from auth.dependencies import get_current_user, require_master_admin, require_management
from auth.security import (
    security, api_key_header, decode_access_token, decode_refresh_token,
    verify_refresh_token, verify_session_key, generate_refresh_token_hash,
    get_password_hash
)
from services.auth_service import AuthService
from services.audit_service import AuditService
from services.student_id import generate_student_id
from services.email_service import EmailService
from core.logger import logger
from datetime import datetime, timedelta
import config


router = APIRouter(prefix="/api/auth", tags=["authentication"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


# Request Models
class MasterAdminLogin(BaseModel):
    """Master admin login request."""
    login_id: str
    password: str


class ManagementLogin(BaseModel):
    """Management login request. User enters their email (mail id) and password."""
    login_id: Optional[str] = None  # Email or login ID (management users use email)
    email: Optional[EmailStr] = None  # Alternative: send email for login
    password: str

    def get_login_id(self) -> str:
        """Return identifier for login (email or login_id)."""
        return (self.email or self.login_id or "").strip()


class FacultyLogin(BaseModel):
    """Faculty login request. User enters email or faculty ID; frontend should call check-user first - if exists and hasPassword false, redirect to OTP to create password."""
    login: Optional[str] = None  # Faculty ID or email
    email: Optional[EmailStr] = None  # Alternative: send email for login
    password: str

    def get_identifier(self) -> str:
        """Return identifier for login (email or login)."""
        return (self.email or self.login or "").strip()


class StudentLogin(BaseModel):
    """Student login request. User enters email or roll number / college ID; frontend should call check-user first - if exists and hasPassword false, redirect to OTP to create password."""
    college_id: Optional[str] = None  # Roll number or college student ID
    email: Optional[EmailStr] = None  # Alternative: send email for login
    password: str

    def get_identifier(self) -> str:
        """Return identifier for login (email or college_id/roll number)."""
        return (self.email or self.college_id or "").strip()


# Sign-up Models
class MasterAdminSignUp(BaseModel):
    """Master admin sign-up request."""
    email: EmailStr
    password: str


class ManagementSignUp(BaseModel):
    """Management sign-up request."""
    college_name: str
    college_code: str  # College code to link management to
    email: EmailStr
    phone: str
    password: str
    address: Optional[str] = None
    username: str


class FacultySignUp(BaseModel):
    """Faculty sign-up request."""
    name: str
    faculty_id: Optional[str] = None
    email: EmailStr
    password: str
    department_name: str
    college_code: Optional[str] = None  # Optional: if provided, associate with college


class StudentSignUp(BaseModel):
    """Student sign-up request."""
    name: str
    roll_number: Optional[str] = None
    branch: Optional[str] = None
    year: Optional[str] = None
    email: EmailStr
    password: str
    college_code: Optional[str] = None  # Optional: if provided, associate with college


class CollegeCreate(BaseModel):
    """College creation request (master admin only)."""
    college_code: str
    name: str
    address: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    contact_phone: Optional[str] = None


class ManagementCreate(BaseModel):
    """Management account creation (master admin only). Add email (mail id); user will use it on login page. Password optional; if omitted, user sets password via OTP."""
    email: EmailStr  # Required: mail id for the management user (they enter this on login page)
    full_name: Optional[str] = None
    college_code: str  # College to manage
    college_name: Optional[str] = None  # College name (optional, for display/update)
    password: Optional[str] = None  # Optional; if omitted, user sets password via OTP flow


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    session_key: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: dict


class APIKeyCreate(BaseModel):
    """API key creation request model."""
    name: Optional[str] = None
    expires_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key response model."""
    access_key: str
    key_id: int
    key_prefix: str
    name: Optional[str]
    expires_at: Optional[str]


class CheckUserRequest(BaseModel):
    """Check user request. Frontend can send emailOrRoll or email."""
    emailOrRoll: Optional[str] = None
    email: Optional[str] = None

    def get_identifier(self) -> Optional[str]:
        """Return normalized identifier (email or roll) for lookup."""
        raw = self.emailOrRoll or self.email
        return raw.strip() if raw else None


class SendOtpRequest(BaseModel):
    """Send OTP request."""
    email: EmailStr


class VerifyOtpRequest(BaseModel):
    """Verify OTP request."""
    email: EmailStr
    otp: str


class SetupPasswordRequest(BaseModel):
    """Setup password request."""
    email: EmailStr
    password: str


class ResetPasswordRequest(BaseModel):
    """Reset password request."""
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class LogoutResponse(BaseModel):
    """Logout response."""
    success: bool


# New Auth Endpoints per END_TO_END_APIS_PROMPT.md

class CheckUserResponse(BaseModel):
    """Check user response. Frontend: hasPassword true -> route to enter password; false -> route to OTP page."""
    exists: bool
    email: Optional[str] = None
    hasPassword: bool


@router.post("/check-user", response_model=CheckUserResponse)
async def check_user(
    request_data: CheckUserRequest,
    db: Session = Depends(get_db_session)
):
    """
    Check if user exists by email or roll number.
    Returns user existence, email, and whether password is set.
    Frontend: if hasPassword is true -> route to enter password; if false -> route to OTP page.
    """
    identifier = request_data.get_identifier()
    if not identifier:
        return CheckUserResponse(exists=False, email=None, hasPassword=False)

    identifier = identifier.strip()

    # Look up by email (case-insensitive), roll_number, or login_id
    user = db.query(User).filter(
        or_(
            func.lower(User.email) == identifier.lower(),
            User.roll_number == identifier,
            User.login_id == identifier,
        )
    ).first()

    if not user:
        return CheckUserResponse(exists=False, email=None, hasPassword=False)

    # Password is "available" only if hashed_password is non-empty and not just placeholder/whitespace
    raw_pw = user.hashed_password
    has_password = bool(raw_pw is not None and str(raw_pw).strip() != "")

    return CheckUserResponse(
        exists=True,
        email=user.email,
        hasPassword=has_password,
    )


@router.post("/send-otp")
async def send_otp(
    request_data: SendOtpRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """
    Send OTP to email address.
    Stores OTP in database with expiry (10 minutes).
    """
    email = request_data.email
    
    # Generate OTP
    otp = EmailService.generate_otp(length=6)
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    
    # Store or update OTP in database
    otp_record = db.query(OtpStore).filter(OtpStore.email == email).first()
    if otp_record:
        otp_record.otp = otp
        otp_record.expires_at = expires_at
        otp_record.created_at = datetime.utcnow()
    else:
        otp_record = OtpStore(
            email=email,
            otp=otp,
            expires_at=expires_at
        )
        db.add(otp_record)
    
    db.commit()
    
    # Send email via fastapi-mail
    fm = getattr(request.app.state, "mail", None)
    if not fm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service not configured. Set SMTP_USER and SMTP_PASSWORD."
        )
    success = await EmailService.send_otp_email(email, otp, fm)
    
    if not success:
        logger.error(f"Failed to send OTP email to {email}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email"
        )
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="send_otp",
        resource_type="otp"
    )
    
    return {"success": True}


@router.post("/verify-otp")
async def verify_otp(
    request_data: VerifyOtpRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """
    Verify OTP code.
    Returns success if OTP is valid and not expired.
    """
    email = request_data.email
    otp = request_data.otp
    
    # Get OTP record
    otp_record = db.query(OtpStore).filter(OtpStore.email == email).first()
    
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OTP not found. Please request a new OTP."
        )
    
    # Check if expired
    if otp_record.expires_at < datetime.utcnow():
        db.delete(otp_record)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new OTP."
        )
    
    # Verify OTP
    if otp_record.otp != otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP"
        )
    
    # Delete OTP after successful verification
    db.delete(otp_record)
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="verify_otp",
        resource_type="otp"
    )
    
    return {"success": True}


@router.post("/setup-password")
async def setup_password(
    request_data: SetupPasswordRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """
    Set password for first-time setup (post-OTP verification).
    User should verify OTP first before calling this endpoint.
    """
    email = request_data.email
    password = request_data.password
    
    # Find user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate password
    from auth.security import validate_password
    is_valid, error_message = validate_password(password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Set password
    user.hashed_password = get_password_hash(password)
    user.password_changed_at = datetime.utcnow()
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="setup_password",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    return {"success": True}


@router.post("/reset-password")
async def reset_password(
    request_data: ResetPasswordRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """
    Reset password (forgot password flow).
    User should verify OTP first before calling this endpoint.
    """
    email = request_data.email
    password = request_data.password
    
    # Find user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate password
    from auth.security import validate_password
    is_valid, error_message = validate_password(password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Reset password
    user.hashed_password = get_password_hash(password)
    user.password_changed_at = datetime.utcnow()
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="reset_password",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    return {"success": True}


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """
    Login with email + password.
    Returns JWT tokens and user info.
    """
    user = AuthService.authenticate_user(
        db=db,
        login_id=credentials.email,
        password=credentials.password,
        ip_address=request.client.host if request.client else None
    )
    
    if not user:
        AuditService.log_from_request(
            db=db,
            request=request,
            action="login_failed",
            resource_type="user"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens and session
    access_token, refresh_token = AuthService.create_tokens(user)
    AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    session_key, _ = AuthService.create_session(
        db=db,
        user_id=user.id,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    # Audit log
    AuditService.log_from_request(
        db=db,
        request=request,
        action="login",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    # Build user info
    user_info = {
        "id": user.id,
        "login_id": user.login_id,
        "email": user.email,
        "role": user.role.value,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }
    
    # Add role-specific fields
    if user.role == UserRole.MANAGEMENT:
        user_info.update({
            "username": user.username,
            "college_name": user.college_name,
            "phone": user.phone,
            "address": user.address,
            "managed_college_id": user.managed_college_id
        })
    elif user.role == UserRole.FACULTY:
        user_info.update({
            "name": user.full_name,
            "faculty_id": user.faculty_id,
            "department_name": user.department_name,
            "college_id": user.college_id,
            "management_id": user.management_id
        })
    elif user.role == UserRole.STUDENT:
        user_info.update({
            "name": user.full_name,
            "roll_number": user.roll_number,
            "branch": user.branch,
            "year": user.year,
            "college_id": user.college_id,
            "college_student_id": user.college_student_id,
            "management_id": user.management_id
        })
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        session_key=session_key,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_info
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Logout (invalidate session/token).
    """
    # Get session hash from request headers or token
    # For now, we'll revoke all sessions for the user
    from database.models import Session as DBSession
    sessions = db.query(DBSession).filter(
        DBSession.user_id == current_user.id,
        DBSession.is_active == True
    ).all()
    
    for session in sessions:
        session.is_active = False
    
    db.commit()
    
    # Log action
    AuditService.log_from_request(
        db=db,
        request=request,
        action="logout",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(current_user.id)
    )
    
    return {"success": True}


# Login Endpoints
@router.post("/login/master-admin", response_model=TokenResponse)
async def login_master_admin(
    credentials: MasterAdminLogin,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Master admin login."""
    user = AuthService.authenticate_user(
        db=db,
        login_id=credentials.login_id,
        password=credentials.password,
        ip_address=request.client.host if request.client else None
    )
    
    if not user:
        AuditService.log_from_request(
            db=db,
            request=request,
            action="master_admin_login_failed",
            resource_type="user"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login ID or password"
        )
    
    if user.role != UserRole.MASTER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Master admin access required."
        )
    
    # Create tokens and session
    access_token, refresh_token = AuthService.create_tokens(user)
    refresh_token_obj = AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    session_key, session = AuthService.create_session(
        db=db,
        user_id=user.id,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    # Audit log
    AuditService.log_from_request(
        db=db,
        request=request,
        action="master_admin_login",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    # Build complete user info
    user_info = {
        "id": user.id,
        "login_id": user.login_id,
        "email": user.email,
        "role": user.role.value,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }
    
    # Add role-specific fields
    if user.role == UserRole.MASTER_ADMIN:
        pass  # Master admin only has email
    elif user.role == UserRole.MANAGEMENT:
        user_info.update({
            "username": user.username,
            "college_name": user.college_name,
            "phone": user.phone,
            "address": user.address,
            "managed_college_id": user.managed_college_id
        })
    elif user.role == UserRole.FACULTY:
        user_info.update({
            "name": user.full_name,
            "faculty_id": user.faculty_id,
            "email": user.email,
            "department_name": user.department_name,
            "college_id": user.college_id,
            "management_id": user.management_id
        })
    elif user.role == UserRole.STUDENT:
        user_info.update({
            "name": user.full_name,
            "roll_number": user.roll_number,
            "branch": user.branch,
            "year": user.year,
            "email": user.email,
            "college_id": user.college_id,
            "college_student_id": user.college_student_id,
            "management_id": user.management_id
        })
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        session_key=session_key,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_info
    )


@router.post("/login/management", response_model=TokenResponse)
async def login_management(
    credentials: ManagementLogin,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Management login. User enters their email (mail id) and password. If no password set yet, use check-user then OTP flow."""
    login_id = credentials.get_login_id()
    if not login_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or login ID required"
        )
    user = AuthService.authenticate_user(
        db=db,
        login_id=login_id,
        password=credentials.password,
        ip_address=request.client.host if request.client else None
    )
    
    if not user:
        AuditService.log_from_request(
            db=db,
            request=request,
            action="management_login_failed",
            resource_type="user"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login ID or password"
        )
    
    if user.role != UserRole.MANAGEMENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Management access required."
        )
    
    # Create tokens and session
    access_token, refresh_token = AuthService.create_tokens(user)
    AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    session_key, _ = AuthService.create_session(
        db=db,
        user_id=user.id,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    AuditService.log_from_request(
        db=db,
        request=request,
        action="management_login",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    # Load college details (college the management user manages)
    college_details = None
    if user.managed_college_id:
        college = db.query(College).filter(College.id == user.managed_college_id).first()
        if college:
            college_details = {
                "id": college.id,
                "college_code": college.college_code,
                "name": college.name,
                "address": college.address,
                "contact_email": college.contact_email,
                "contact_phone": college.contact_phone,
                "website": college.website,
                "status": college.status.value if hasattr(college.status, "value") else str(college.status),
                "is_active": college.is_active,
            }
    
    # Build complete user info with college details
    user_info = {
        "id": user.id,
        "login_id": user.login_id,
        "email": user.email,
        "role": user.role.value,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "username": user.username,
        "college_name": user.college_name,
        "phone": user.phone,
        "address": user.address,
        "managed_college_id": user.managed_college_id,
        "college": college_details,
    }
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        session_key=session_key,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_info
    )


@router.post("/login/faculty", response_model=TokenResponse)
async def login_faculty(
    credentials: FacultyLogin,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Faculty login. User enters email or faculty ID and password. If no password set yet, use check-user then OTP flow."""
    identifier = credentials.get_identifier()
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or faculty ID required"
        )
    user = AuthService.authenticate_user(
        db=db,
        login_id=identifier,
        password=credentials.password,
        ip_address=request.client.host if request.client else None
    )
    
    if not user:
        AuditService.log_from_request(
            db=db,
            request=request,
            action="faculty_login_failed",
            resource_type="user"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login or password"
        )
    
    if user.role != UserRole.FACULTY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Faculty access required."
        )
    
    # Create tokens and session
    access_token, refresh_token = AuthService.create_tokens(user)
    AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    session_key, _ = AuthService.create_session(
        db=db,
        user_id=user.id,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    AuditService.log_from_request(
        db=db,
        request=request,
        action="faculty_login",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    # Build complete user info
    user_info = {
        "id": user.id,
        "login_id": user.login_id,
        "email": user.email,
        "role": user.role.value,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "name": user.full_name,
        "faculty_id": user.faculty_id,
        "department_name": user.department_name,
        "college_id": user.college_id,
        "management_id": user.management_id
    }
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        session_key=session_key,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_info
    )


@router.post("/login/student", response_model=TokenResponse)
async def login_student(
    credentials: StudentLogin,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Student login. User enters email or roll number / college ID and password. If no password set yet, use check-user then OTP flow."""
    identifier = credentials.get_identifier()
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or roll number / college ID required"
        )
    user = AuthService.authenticate_user(
        db=db,
        login_id=identifier,
        password=credentials.password,
        ip_address=request.client.host if request.client else None
    )
    
    if not user:
        AuditService.log_from_request(
            db=db,
            request=request,
            action="student_login_failed",
            resource_type="user"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or roll number or password"
        )
    
    if user.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Student access required."
        )
    
    # Create tokens and session
    access_token, refresh_token = AuthService.create_tokens(user)
    AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    session_key, _ = AuthService.create_session(
        db=db,
        user_id=user.id,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    AuditService.log_from_request(
        db=db,
        request=request,
        action="student_login",
        user_id=user.id,
        resource_type="user",
        resource_id=str(user.id)
    )
    
    # Build complete user info
    user_info = {
        "id": user.id,
        "login_id": user.login_id,
        "email": user.email,
        "role": user.role.value,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "name": user.full_name,
        "roll_number": user.roll_number,
        "branch": user.branch,
        "year": user.year,
        "college_id": user.college_id,
        "college_student_id": user.college_student_id,
        "management_id": user.management_id
    }
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        session_key=session_key,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_info
    )


# Refresh Token Endpoint
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Refresh access token using refresh token."""
    refresh_token = token_data.refresh_token
    
    # Decode refresh token
    payload = decode_refresh_token(refresh_token, config.SECRET_KEY)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Verify token in database
    from database.models import RefreshToken
    token_hash = generate_refresh_token_hash(refresh_token)
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.is_revoked == False,
        RefreshToken.expires_at > datetime.utcnow()
    ).first()
    
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or expired"
        )
    
    # Get user
    user = AuthService.get_user_by_id(db, payload.get("user_id"))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token, new_refresh_token = AuthService.create_tokens(user)
    
    # Revoke old refresh token and save new one
    AuthService.revoke_refresh_token(db, token_hash)
    AuthService.save_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=new_refresh_token,
        device_info=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "login_id": user.login_id,
            "role": user.role.value
        }
    )


# Master Admin Endpoints
@router.post("/master-admin/colleges", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_college(
    college_data: CollegeCreate,
    current_user: User = Depends(require_master_admin),
    request: Request = None,
    db: Session = Depends(get_db_session)
):
    """Create a new college (master admin only)."""
    try:
        college = AuthService.create_college(
            db=db,
            college_code=college_data.college_code,
            name=college_data.name,
            created_by=current_user.id,
            address=college_data.address,
            contact_email=college_data.contact_email,
            contact_phone=college_data.contact_phone
        )
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="college_create",
            user_id=current_user.id,
            resource_type="college",
            resource_id=str(college.id)
        )
        
        return {
            "message": "College created successfully",
            "college_id": college.id,
            "college_code": college.college_code,
            "name": college.name
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/master-admin/management", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_management(
    management_data: ManagementCreate,
    current_user: User = Depends(require_master_admin),
    request: Request = None,
    db: Session = Depends(get_db_session)
):
    """Create a management account for a college (master admin only). Add the management user's email (mail id); they will use it on the login page and set their password via OTP if not provided."""
    # Get college
    college = db.query(College).filter(College.college_code == management_data.college_code).first()
    if not college:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="College not found"
        )
    
    # Check email not already used
    existing = db.query(User).filter(User.email == management_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists"
        )
    
    # Update college name if provided
    if management_data.college_name:
        college.name = management_data.college_name
        db.commit()
    
    try:
        # Use email as login_id so management user can enter their mail id on login page
        user = AuthService.create_user(
            db=db,
            login_id=management_data.email,
            role=UserRole.MANAGEMENT,
            email=management_data.email,
            full_name=management_data.full_name or management_data.college_name or college.name,
            managed_college_id=college.id,
            created_by=current_user.id,
            password=management_data.password,  # Optional; if None, user sets via OTP
        )
        
        # Set management-specific fields
        user.college_name = management_data.college_name or college.name
        db.commit()
        db.refresh(user)
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="management_create",
            user_id=current_user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return {
            "message": "Management account created successfully. User can log in with this email; if no password was set, they will set it via OTP.",
            "user_id": user.id,
            "email": user.email,
            "login_id": user.login_id,
            "college_code": college.college_code,
            "college_name": college.name,
            "hasPassword": bool(user.hashed_password and str(user.hashed_password).strip() != ""),
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# User Info Endpoint
@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Get current user information with all details."""
    from auth.dependencies import get_management_college_id

    # Base user info
    user_info = {
        "id": current_user.id,
        "login_id": current_user.login_id,
        "email": current_user.email,
        "role": current_user.role.value,
        "full_name": current_user.full_name,
        "college_id": current_user.college_id,
        "managed_college_id": current_user.managed_college_id,
        "college_student_id": current_user.college_student_id,
        "management_id": current_user.management_id,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }
    
    # Add role-specific fields
    if current_user.role == UserRole.MASTER_ADMIN:
        # Master admin has email and password (password not returned)
        pass
    
    elif current_user.role == UserRole.MANAGEMENT:
        user_info.update({
            "username": current_user.username,
            "college_name": current_user.college_name,
            "phone": current_user.phone,
            "address": current_user.address
        })
        # Include college details (college they manage; use managed_college_id or college_id)
        effective_college_id = get_management_college_id(current_user)
        if effective_college_id:
            college = db.query(College).filter(College.id == effective_college_id).first()
            if college:
                user_info["college"] = {
                    "id": college.id,
                    "college_code": college.college_code,
                    "name": college.name,
                    "address": college.address,
                    "contact_email": college.contact_email,
                    "contact_phone": college.contact_phone,
                    "website": college.website,
                    "status": college.status.value if hasattr(college.status, "value") else str(college.status),
                    "is_active": college.is_active,
                }
            else:
                user_info["college"] = None
        else:
            user_info["college"] = None
    
    elif current_user.role == UserRole.FACULTY:
        user_info.update({
            "name": current_user.full_name,
            "faculty_id": current_user.faculty_id,
            "department_name": current_user.department_name,
            "management_id": current_user.management_id
        })
    
    elif current_user.role == UserRole.STUDENT:
        user_info.update({
            "name": current_user.full_name,
            "roll_number": current_user.roll_number,
            "branch": current_user.branch,
            "year": current_user.year,
            "management_id": current_user.management_id
        })
    
    return user_info


# API Key Endpoints
@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db_session)
):
    """Create a new API key for the current user."""
    access_key, api_key = AuthService.create_api_key(
        db=db,
        user_id=current_user.id,
        name=key_data.name,
        expires_days=key_data.expires_days
    )
    
    AuditService.log_from_request(
        db=db,
        request=request,
        action="api_key_create",
        user_id=current_user.id,
        resource_type="api_key",
        resource_id=str(api_key.id)
    )
    
    return APIKeyResponse(
        access_key=access_key,  # Only returned once!
        key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None
    )


@router.get("/api-keys", response_model=list[dict])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """List user's API keys."""
    from database.models import APIKey
    api_keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id
    ).all()
    
    return [
        {
            "id": key.id,
            "key_prefix": key.key_prefix,
            "name": key.name,
            "is_active": key.is_active,
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            "created_at": key.created_at.isoformat()
        }
        for key in api_keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db_session)
):
    """Revoke an API key."""
    from database.models import APIKey
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    AuthService.revoke_api_key(db, key_id)
    
    AuditService.log_from_request(
        db=db,
        request=request,
        action="api_key_revoke",
        user_id=current_user.id,
        resource_type="api_key",
        resource_id=str(key_id)
    )
    
    return None


# Sign-up Endpoints
@router.post("/signup/master-admin", response_model=dict, status_code=status.HTTP_201_CREATED)
async def signup_master_admin(
    signup_data: MasterAdminSignUp,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Master admin sign-up."""
    try:
        # Use email as login_id for master admin
        user = AuthService.create_user(
            db=db,
            login_id=signup_data.email,
            password=signup_data.password,
            role=UserRole.MASTER_ADMIN,
            email=signup_data.email
        )
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="master_admin_signup",
            user_id=user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return {
            "message": "Master admin account created successfully",
            "user": {
                "id": user.id,
                "login_id": user.login_id,
                "email": user.email,
                "role": user.role.value,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/signup/management", response_model=dict, status_code=status.HTTP_201_CREATED)
async def signup_management(
    signup_data: ManagementSignUp,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Management sign-up."""
    try:
        # Get or create college based on college_code
        college = db.query(College).filter(College.college_code == signup_data.college_code).first()
        if not college:
            # College doesn't exist, create it
            college = College(
                college_code=signup_data.college_code,
                name=signup_data.college_name,
                address=signup_data.address,
                is_active=True
            )
            db.add(college)
            db.commit()
            db.refresh(college)
        
        # Use username as login_id for management
        user = AuthService.create_user(
            db=db,
            login_id=signup_data.username,
            password=signup_data.password,
            role=UserRole.MANAGEMENT,
            email=signup_data.email,
            full_name=signup_data.college_name,
            managed_college_id=college.id
        )
        
        # Update management-specific fields
        user.college_name = signup_data.college_name
        user.phone = signup_data.phone
        user.address = signup_data.address
        user.username = signup_data.username
        db.commit()
        db.refresh(user)
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="management_signup",
            user_id=user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return {
            "message": "Management account created successfully",
            "user": {
                "id": user.id,
                "login_id": user.login_id,
                "username": user.username,
                "email": user.email,
                "phone": user.phone,
                "college_name": user.college_name,
                "college_code": signup_data.college_code,
                "managed_college_id": user.managed_college_id,
                "address": user.address,
                "role": user.role.value,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/signup/faculty", response_model=dict, status_code=status.HTTP_201_CREATED)
async def signup_faculty(
    signup_data: FacultySignUp,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Faculty sign-up."""
    try:
        # Use email or faculty_id as login_id
        login_id = signup_data.faculty_id if signup_data.faculty_id else signup_data.email
        
        # Get college if college_code provided
        college_id = None
        if signup_data.college_code:
            college = db.query(College).filter(College.college_code == signup_data.college_code).first()
            if college:
                college_id = college.id
        
        user = AuthService.create_user(
            db=db,
            login_id=login_id,
            password=signup_data.password,
            role=UserRole.FACULTY,
            email=signup_data.email,
            full_name=signup_data.name,
            college_id=college_id
        )
        
        # Update faculty-specific fields
        user.faculty_id = signup_data.faculty_id
        user.department_name = signup_data.department_name
        db.commit()
        db.refresh(user)
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="faculty_signup",
            user_id=user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return {
            "message": "Faculty account created successfully",
            "user": {
                "id": user.id,
                "login_id": user.login_id,
                "name": user.full_name,
                "faculty_id": user.faculty_id,
                "email": user.email,
                "department_name": user.department_name,
                "college_id": user.college_id,
                "management_id": user.management_id,
                "role": user.role.value,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/signup/student", response_model=dict, status_code=status.HTTP_201_CREATED)
async def signup_student(
    signup_data: StudentSignUp,
    request: Request,
    db: Session = Depends(get_db_session)
):
    """Student sign-up."""
    try:
        # Use roll_number or email as login_id
        login_id = signup_data.roll_number if signup_data.roll_number else signup_data.email
        
        # Get college if college_code provided
        college_id = None
        if signup_data.college_code:
            college = db.query(College).filter(College.college_code == signup_data.college_code).first()
            if college:
                college_id = college.id
        
        # Generate unique student_id when college is set (separate from roll_number)
        generated_student_id = generate_student_id(db, college_id) if college_id else None
        
        user = AuthService.create_user(
            db=db,
            login_id=login_id,
            password=signup_data.password,
            role=UserRole.STUDENT,
            email=signup_data.email,
            full_name=signup_data.name,
            college_id=college_id,
            college_student_id=generated_student_id if generated_student_id else signup_data.roll_number,
        )
        
        # Update student-specific fields
        user.roll_number = signup_data.roll_number
        user.branch = signup_data.branch
        # Keep department_name in sync for students so profile & lists can show it
        if signup_data.branch:
            user.department_name = signup_data.branch
        user.year = signup_data.year
        if generated_student_id:
            user.student_id = generated_student_id
            user.college_student_id = generated_student_id
        db.commit()
        db.refresh(user)
        
        AuditService.log_from_request(
            db=db,
            request=request,
            action="student_signup",
            user_id=user.id,
            resource_type="user",
            resource_id=str(user.id)
        )
        
        return {
            "message": "Student account created successfully",
            "user": {
                "id": user.id,
                "student_id": getattr(user, "student_id", None),
                "login_id": user.login_id,
                "name": user.full_name,
                "roll_number": user.roll_number,
                "branch": user.branch,
                "year": user.year,
                "email": user.email,
                "college_id": user.college_id,
                "college_student_id": user.college_student_id,
                "management_id": user.management_id,
                "role": user.role.value,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat()
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
