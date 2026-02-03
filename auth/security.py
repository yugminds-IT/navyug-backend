"""
Advanced security utilities for authentication and authorization.
Includes encryption, JWT tokens, refresh tokens, sessions, and API keys.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
import bcrypt
from fastapi import HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from cryptography.fernet import Fernet
import secrets
import hashlib
import base64
import logging

from core.logger import logger
import config

# Password hashing
# Configure to avoid wrap bug detection issues
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__ident="2b",  # Use bcrypt 2b identifier
    bcrypt__rounds=12  # Standard rounds
)

# JWT settings
SECRET_KEY_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30
SESSION_EXPIRE_HOURS = 24 * 7  # 7 days

# Security schemes
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)  # For optional authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_header = APIKeyHeader(name="X-Session-Key", auto_error=False)


# Encryption utilities
def get_encryption_key() -> bytes:
    """
    Get encryption key from config.
    If not set, generate one (not recommended for production).
    """
    encryption_key = getattr(config, 'ENCRYPTION_KEY', None)
    if not encryption_key:
        logger.warning("ENCRYPTION_KEY not set, generating temporary key (not secure for production!)")
        encryption_key = Fernet.generate_key().decode()
        config.ENCRYPTION_KEY = encryption_key
    
    # Ensure key is 32 bytes (Fernet requirement)
    if isinstance(encryption_key, str):
        # If it's a base64 string, decode it
        try:
            key_bytes = base64.urlsafe_b64decode(encryption_key + '==')
            if len(key_bytes) != 32:
                # Generate new key if invalid
                key_bytes = Fernet.generate_key()
        except:
            # If not valid base64, hash it to get 32 bytes
            key_bytes = hashlib.sha256(encryption_key.encode()).digest()
    else:
        key_bytes = encryption_key
    
    # Ensure it's exactly 32 bytes and base64-encoded
    if len(key_bytes) != 32:
        key_bytes = hashlib.sha256(str(encryption_key).encode()).digest()[:32]
    
    return base64.urlsafe_b64encode(key_bytes)


def encrypt_data(data: str) -> str:
    """
    Encrypt sensitive data using Fernet symmetric encryption.
    
    Args:
        data: Data to encrypt
        
    Returns:
        Encrypted string (base64)
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise


def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypt data using Fernet symmetric encryption.
    
    Args:
        encrypted_data: Encrypted string (base64)
        
    Returns:
        Decrypted string
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise


# Password utilities
def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validate password strength.
    
    Requirements:
    - Minimum 6 characters
    - At least 1 number
    - At least 1 special character
    - Maximum 72 bytes (bcrypt limit)
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    # Check length (minimum 6 characters)
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Check maximum length (72 bytes for bcrypt)
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        return False, "Password cannot be longer than 72 bytes. Please use a shorter password."
    
    # Check for at least one number
    has_number = any(char.isdigit() for char in password)
    if not has_number:
        return False, "Password must contain at least one number"
    
    # Check for at least one special character
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    has_special = any(char in special_chars for char in password)
    if not has_special:
        return False, "Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)"
    
    return True, None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        # Try direct bcrypt first
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception:
        # Fallback to passlib
        return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Note: Password validation should be done before calling this function.
    Use validate_password() to check password requirements.
    """
    # Ensure password doesn't exceed 72 bytes
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        raise ValueError("Password cannot be longer than 72 bytes. Please use a shorter password.")
    
    # Use bcrypt directly to avoid passlib initialization issues
    # This is more reliable and avoids the wrap bug detection problem
    try:
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        # Return as string (passlib format: $2b$12$...)
        return hashed.decode('utf-8')
    except Exception as e:
        # Fallback to passlib if direct bcrypt fails
        try:
            return pwd_context.hash(password)
        except ValueError as ve:
            if "cannot be longer than 72 bytes" in str(ve):
                raise ValueError("Password cannot be longer than 72 bytes. Please use a shorter password.")
            raise


# JWT Token utilities
def create_access_token(
    data: Dict[str, Any],
    secret_key: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in token
        secret_key: Secret key for signing
        expires_delta: Optional expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=SECRET_KEY_ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    secret_key: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in token
        secret_key: Secret key for signing
        expires_delta: Optional expiration time
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=SECRET_KEY_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string
        secret_key: Secret key for verification
        
    Returns:
        Decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[SECRET_KEY_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


def decode_refresh_token(token: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT refresh token.
    
    Args:
        token: JWT refresh token string
        secret_key: Secret key for verification
        
    Returns:
        Decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[SECRET_KEY_ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except JWTError:
        return None


# API Key utilities
def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate a new API key with access key.
    
    Returns:
        Tuple of (full_access_key, encrypted_access_key, key_hash)
    """
    # Generate 32-byte random key
    access_key = secrets.token_urlsafe(32)
    # Hash the key for storage/verification
    key_hash = hashlib.sha256(access_key.encode()).hexdigest()
    # Encrypt the key for storage
    encrypted_key = encrypt_data(access_key)
    return access_key, encrypted_key, key_hash


def hash_api_key(key: str) -> str:
    """
    Hash an API key for storage/comparison.
    
    Args:
        key: API key string
        
    Returns:
        Hashed key
    """
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against stored hash.
    
    Args:
        provided_key: API key to verify
        stored_hash: Stored hash to compare against
        
    Returns:
        True if key matches
    """
    provided_hash = hash_api_key(provided_key)
    return secrets.compare_digest(provided_hash, stored_hash)


# Session utilities
def generate_session_key() -> Tuple[str, str, str]:
    """
    Generate a new session key.
    
    Returns:
        Tuple of (session_key, encrypted_session_key, session_hash)
    """
    session_key = secrets.token_urlsafe(32)
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    encrypted_key = encrypt_data(session_key)
    return session_key, encrypted_key, session_hash


def hash_session_key(key: str) -> str:
    """
    Hash a session key for storage/comparison.
    
    Args:
        key: Session key string
        
    Returns:
        Hashed key
    """
    return hashlib.sha256(key.encode()).hexdigest()


def verify_session_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify a session key against stored hash.
    
    Args:
        provided_key: Session key to verify
        stored_hash: Stored hash to compare against
        
    Returns:
        True if key matches
    """
    provided_hash = hash_session_key(provided_key)
    return secrets.compare_digest(provided_hash, stored_hash)


# Refresh token utilities
def generate_refresh_token_hash(token: str) -> str:
    """
    Generate hash for refresh token.
    
    Args:
        token: Refresh token string
        
    Returns:
        Hashed token
    """
    return hashlib.sha256(token.encode()).hexdigest()


def verify_refresh_token(provided_token: str, stored_hash: str) -> bool:
    """
    Verify a refresh token against stored hash.
    
    Args:
        provided_token: Refresh token to verify
        stored_hash: Stored hash to compare against
        
    Returns:
        True if token matches
    """
    provided_hash = generate_refresh_token_hash(provided_token)
    return secrets.compare_digest(provided_hash, stored_hash)


# Role checkers
class RoleChecker:
    """Check if user has required role."""
    
    def __init__(self, allowed_roles: list[str]):
        """
        Initialize role checker.
        
        Args:
            allowed_roles: List of allowed roles
        """
        self.allowed_roles = allowed_roles
    
    def __call__(self, user_role: str) -> bool:
        """
        Check if user role is allowed.
        
        Args:
            user_role: User's role
            
        Returns:
            True if allowed
        """
        return user_role in self.allowed_roles


# Role checkers for new system
require_master_admin = RoleChecker(["master_admin"])
require_management = RoleChecker(["master_admin", "management"])
require_faculty = RoleChecker(["master_admin", "management", "faculty"])
require_student = RoleChecker(["master_admin", "management", "faculty", "student"])
