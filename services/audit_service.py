"""
Audit logging service for security and compliance.
"""
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import Request

from database.models import AuditLog
from core.logger import logger


class AuditService:
    """Service for audit logging."""
    
    @staticmethod
    def log_action(
        db: Session,
        action: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log an action to audit log.
        
        Args:
            db: Database session
            action: Action name (e.g., "video_upload", "user_login")
            user_id: Optional user ID
            resource_type: Type of resource (e.g., "video_job", "person")
            resource_id: ID of resource
            ip_address: IP address
            user_agent: User agent string
            details: Additional details
            
        Returns:
            Created AuditLog
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)
        return audit_log
    
    @staticmethod
    def log_from_request(
        db: Session,
        request: Request,
        action: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log an action from a FastAPI request.
        
        Args:
            db: Database session
            request: FastAPI request object
            action: Action name
            user_id: Optional user ID
            resource_type: Type of resource
            resource_id: ID of resource
            details: Additional details
            
        Returns:
            Created AuditLog
        """
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        return AuditService.log_action(
            db=db,
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
