"""
Enhanced authentication service with multi-role support, refresh tokens, and sessions.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from database.models import User, APIKey, UserRole, RefreshToken, Session as DBSession, College
from auth.security import (
    verify_password, get_password_hash, validate_password, create_access_token, create_refresh_token,
    generate_api_key, hash_api_key, generate_session_key, hash_session_key,
    generate_refresh_token_hash, encrypt_data, decrypt_data
)
from core.logger import logger
import config


class AuthService:
    """Service for authentication operations with advanced security."""
    
    @staticmethod
    def create_user(
        db: Session,
        login_id: str,
        role: UserRole,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        college_id: Optional[int] = None,
        managed_college_id: Optional[int] = None,
        college_student_id: Optional[str] = None,
        created_by: Optional[int] = None,
        management_id: Optional[int] = None,
        password: Optional[str] = None,
    ) -> User:
        """
        Create a new user. Password is optional; if omitted, user sets it via OTP flow.

        Args:
            db: Database session
            login_id: Login ID (username for most, college_id for students)
            role: User role
            email: Email address (optional for students)
            full_name: Full name
            college_id: College ID for faculty/students
            managed_college_id: College ID for management
            college_student_id: Student's college ID number
            created_by: User ID who created this user (for management accounts)
            management_id: Management user ID (for students/faculty, auto-detected if not provided)
            password: Plain text password (optional; if None/empty, user will set via OTP)

        Returns:
            Created User
        """
        if password is not None and password.strip() != "":
            is_valid, error_message = validate_password(password)
            if not is_valid:
                raise ValueError(error_message)
            hashed_password = get_password_hash(password)
            password_changed_at = datetime.utcnow()
        else:
            hashed_password = ""
            password_changed_at = None

        # Check if user exists
        existing = db.query(User).filter(User.login_id == login_id).first()

        if existing:
            raise ValueError("User with this login ID already exists")

        # Auto-detect management_id for students and faculty if not provided
        if management_id is None and college_id is not None and role in [UserRole.STUDENT, UserRole.FACULTY]:
            # Find management user who manages this college
            management_user = db.query(User).filter(
                User.role == UserRole.MANAGEMENT,
                User.managed_college_id == college_id,
                User.is_active == True
            ).first()

            if management_user:
                management_id = management_user.id
                logger.info(f"Auto-linked {role.value} to management user {management_id} for college {college_id}")
            else:
                logger.warning(f"No active management user found for college {college_id}, management_id will be NULL")

        user = User(
            login_id=login_id,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
            college_id=college_id,
            managed_college_id=managed_college_id,
            college_student_id=college_student_id,
            created_by=created_by,
            management_id=management_id,
            is_active=True,
            is_verified=False,
            password_changed_at=password_changed_at,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created user: {login_id} (role: {role.value})")
        return user
    
    @staticmethod
    def authenticate_user(
        db: Session,
        login_id: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate a user with account lockout protection.
        
        Args:
            db: Database session
            login_id: Login ID (username or college_id for students)
            password: Plain text password
            ip_address: IP address for logging
            
        Returns:
            User if authenticated, None otherwise
        """
        user = db.query(User).filter(User.login_id == login_id).first()
        # Allow login by email (e.g. management/faculty/student enter email on login page)
        if not user and "@" in login_id:
            user = db.query(User).filter(func.lower(User.email) == login_id.lower()).first()
        # Allow login by roll number (e.g. student enters roll number / college ID)
        if not user:
            user = db.query(User).filter(User.roll_number == login_id).first()
        
        if not user:
            return None
        
        # User has no password set (must set via OTP flow)
        if not user.hashed_password or not str(user.hashed_password).strip():
            return None
        
        # Check if account is locked
        if user.is_locked:
            if user.locked_until and user.locked_until > datetime.utcnow():
                logger.warning(f"Login attempt for locked account: {login_id}")
                return None
            else:
                # Lockout expired, unlock account
                user.is_locked = False
                user.locked_until = None
                user.failed_login_attempts = 0
                db.commit()
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if max attempts reached
            if user.failed_login_attempts >= config.MAX_LOGIN_ATTEMPTS:
                user.is_locked = True
                user.locked_until = datetime.utcnow() + timedelta(minutes=config.LOCKOUT_DURATION_MINUTES)
                logger.warning(f"Account locked due to too many failed attempts: {login_id}")
            
            db.commit()
            return None
        
        if not user.is_active:
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.is_locked = False
        user.locked_until = None
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    @staticmethod
    def create_tokens(user: User) -> Tuple[str, str]:
        """
        Create access and refresh tokens for user.
        
        Args:
            user: User object
            
        Returns:
            Tuple of (access_token, refresh_token)
        """
        data = {
            "sub": user.login_id,
            "email": user.email,
            "role": user.role.value,
            "user_id": user.id,
            "college_id": user.college_id,
            "managed_college_id": user.managed_college_id
        }
        
        access_token = create_access_token(data, config.SECRET_KEY)
        refresh_token = create_refresh_token(data, config.SECRET_KEY)
        
        return access_token, refresh_token
    
    @staticmethod
    def save_refresh_token(
        db: Session,
        user_id: int,
        refresh_token: str,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> RefreshToken:
        """
        Save refresh token to database.
        
        Args:
            db: Database session
            user_id: User ID
            refresh_token: Refresh token string
            device_info: Device/browser info
            ip_address: IP address
            
        Returns:
            Created RefreshToken
        """
        token_hash = generate_refresh_token_hash(refresh_token)
        expires_at = datetime.utcnow() + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
        
        refresh_token_obj = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            device_info=device_info,
            ip_address=ip_address,
            expires_at=expires_at
        )
        db.add(refresh_token_obj)
        db.commit()
        db.refresh(refresh_token_obj)
        return refresh_token_obj
    
    @staticmethod
    def revoke_refresh_token(db: Session, token_hash: str) -> bool:
        """Revoke a refresh token."""
        token = db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False
        ).first()
        
        if not token:
            return False
        
        token.is_revoked = True
        token.revoked_at = datetime.utcnow()
        db.commit()
        return True
    
    @staticmethod
    def create_session(
        db: Session,
        user_id: int,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[str, DBSession]:
        """
        Create a new session for user.
        
        Args:
            db: Database session
            user_id: User ID
            device_info: Device/browser info
            ip_address: IP address
            user_agent: User agent string
            
        Returns:
            Tuple of (session_key, DBSession object)
        """
        session_key, encrypted_key, session_hash = generate_session_key()
        expires_at = datetime.utcnow() + timedelta(hours=config.SESSION_EXPIRE_HOURS)
        
        session = DBSession(
            user_id=user_id,
            session_key=encrypted_key,
            session_hash=session_hash,
            device_info=device_info,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            is_active=True
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info(f"Created session for user: {user_id}")
        return session_key, session
    
    @staticmethod
    def revoke_session(db: Session, session_hash: str) -> bool:
        """Revoke a session."""
        session = db.query(DBSession).filter(
            DBSession.session_hash == session_hash,
            DBSession.is_active == True
        ).first()
        
        if not session:
            return False
        
        session.is_active = False
        db.commit()
        return True
    
    @staticmethod
    def create_api_key(
        db: Session,
        user_id: int,
        name: Optional[str] = None,
        expires_days: Optional[int] = None
    ) -> Tuple[str, APIKey]:
        """
        Create API key for user.
        
        Args:
            db: Database session
            user_id: User ID
            name: Optional key name
            expires_days: Optional expiration in days
            
        Returns:
            Tuple of (full_access_key, APIKey object)
        """
        access_key, encrypted_key, key_hash = generate_api_key()
        key_prefix = access_key[:8]
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key = APIKey(
            user_id=user_id,
            access_key=encrypted_key,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            expires_at=expires_at,
            is_active=True
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        logger.info(f"Created API key for user: {user_id}")
        
        return access_key, api_key
    
    @staticmethod
    def revoke_api_key(db: Session, key_id: int) -> bool:
        """Revoke an API key."""
        api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
        if not api_key:
            return False
        
        api_key.is_active = False
        db.commit()
        logger.info(f"Revoked API key: {key_id}")
        return True
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_user_by_login_id(db: Session, login_id: str) -> Optional[User]:
        """Get user by login ID."""
        return db.query(User).filter(User.login_id == login_id).first()
    
    @staticmethod
    def create_college(
        db: Session,
        college_code: str,
        name: str,
        created_by: int,
        address: Optional[str] = None,
        contact_email: Optional[str] = None,
        contact_phone: Optional[str] = None
    ) -> College:
        """
        Create a new college (by master admin).
        
        Args:
            db: Database session
            college_code: Unique college code
            name: College name
            created_by: Master admin user ID
            address: College address
            contact_email: Contact email
            contact_phone: Contact phone
            
        Returns:
            Created College
        """
        # Check if college exists
        existing = db.query(College).filter(College.college_code == college_code).first()
        if existing:
            raise ValueError("College with this code already exists")
        
        college = College(
            college_code=college_code,
            name=name,
            address=address,
            contact_email=contact_email,
            contact_phone=contact_phone,
            created_by=created_by,
            is_active=True
        )
        db.add(college)
        db.commit()
        db.refresh(college)
        logger.info(f"Created college: {college_code} - {name}")
        return college
