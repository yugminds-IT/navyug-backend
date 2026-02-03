"""
Authentication dependencies for FastAPI.
"""
from typing import Optional
from datetime import datetime
import hashlib
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError

from database.connection import Database
from database.models import User, APIKey, UserRole
from auth.security import (
    security, security_optional, api_key_header, decode_access_token, verify_api_key
)
import config


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db_session)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        db: Database session
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    # Decode token
    payload = decode_access_token(token, config.SECRET_KEY)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    login_id: str = payload.get("sub")
    if login_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database using login_id (which matches what's stored in token)
    user = db.query(User).filter(User.login_id == login_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    return user


def get_management_college_id(user: User) -> Optional[int]:
    """
    For management users, return the college they manage.
    Uses managed_college_id if set, else college_id (fallback for legacy data).
    """
    if user.role != UserRole.MANAGEMENT:
        return None
    return user.managed_college_id if user.managed_college_id is not None else user.college_id


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_optional),
    db: Session = Depends(get_db_session)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.
    
    Args:
        credentials: Optional HTTP Bearer token credentials
        db: Database session
        
    Returns:
        Current user or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


async def get_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header),
    db: Session = Depends(get_db_session)
) -> Optional[User]:
    """
    Get user from API key.
    
    Args:
        api_key: API key from header
        db: Database session
        
    Returns:
        User if API key is valid, None otherwise
    """
    if not api_key:
        return None
    
    # Hash the provided key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Find API key in database
    db_api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()
    
    if db_api_key is None:
        return None
    
    # Check expiration
    if db_api_key.expires_at and db_api_key.expires_at < datetime.utcnow():
        return None
    
    # Update last used timestamp
    db_api_key.last_used_at = datetime.utcnow()
    db.commit()
    
    # Get user
    user = db.query(User).filter(User.id == db_api_key.user_id).first()
    if user and user.is_active:
        return user
    
    return None


async def get_current_user_or_api_key(
    user: Optional[User] = Depends(get_current_user_optional),
    api_user: Optional[User] = Depends(get_user_from_api_key)
) -> User:
    """
    Get current user from either JWT token or API key.
    
    Args:
        user: User from JWT token (optional)
        api_user: User from API key (optional)
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if user:
        return user
    if api_user:
        return api_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(allowed_roles: list[str]):
    """
    Dependency factory for role-based access control.
    
    Args:
        allowed_roles: List of allowed roles
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_user_or_api_key)
    ) -> User:
        if current_user.role.value not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        return current_user
    
    return role_checker


# Role dependencies for new system
require_master_admin = require_role(["master_admin"])
require_management = require_role(["master_admin", "management"])
require_faculty = require_role(["master_admin", "management", "faculty"])
require_student = require_role(["master_admin", "management", "faculty", "student"])

# Legacy role dependencies (for backward compatibility)
require_admin = require_role(["master_admin"])
require_user = require_role(["master_admin", "management", "faculty", "student"])
require_viewer = require_role(["master_admin", "management", "faculty", "student"])
