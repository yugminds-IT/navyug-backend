"""
System Logs APIs (Master Admin).
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import and_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import csv
import io

from database.connection import Database
from database.models import User, UserRole, SystemLog, LogLevel, LogCategory
from auth.dependencies import get_current_user, require_master_admin
from core.logger import logger
import config


router = APIRouter(prefix="/api/logs", tags=["logs"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


class LogEntry(BaseModel):
    """Log entry response."""
    id: int
    timestamp: str
    level: str
    category: str
    message: str
    userId: Optional[int]
    ip: Optional[str]
    metadata: Optional[dict]


class LogListResponse(BaseModel):
    """Log list response."""
    data: List[dict]
    total: int
    page: int
    limit: int


@router.get("", response_model=LogListResponse)
async def list_logs(
    level: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    user: Optional[int] = Query(None, alias="user"),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    List system logs.
    Master admin only.
    """
    query = db.query(SystemLog)
    
    # Filter by level
    if level:
        try:
            level_enum = LogLevel[level.upper()]
            query = query.filter(SystemLog.level == level_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid log level: {level}"
            )
    
    # Filter by category
    if category:
        try:
            category_enum = LogCategory[category.upper()]
            query = query.filter(SystemLog.category == category_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid log category: {category}"
            )
    
    # Filter by user
    if user:
        query = query.filter(SystemLog.user_id == user)
    
    # Filter by date range
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            query = query.filter(SystemLog.created_at >= from_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid from date format. Use ISO 8601 format."
            )
    
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
            query = query.filter(SystemLog.created_at <= to_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid to date format. Use ISO 8601 format."
            )
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    logs = query.order_by(SystemLog.created_at.desc()).offset(offset).limit(limit).all()
    
    # Build response
    log_list = []
    for log in logs:
        log_dict = {
            "id": log.id,
            "timestamp": log.created_at.isoformat(),
            "level": log.level.value,
            "category": log.category.value,
            "message": log.message,
            "userId": log.user_id,
            "ip": log.ip,
            "metadata": log.extra_metadata
        }
        log_list.append(log_dict)
    
    return LogListResponse(
        data=log_list,
        total=total,
        page=page,
        limit=limit
    )


@router.get("/export")
async def export_logs(
    level: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    user: Optional[int] = Query(None, alias="user"),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    format: str = Query("csv", regex="^(csv|json)$"),
    current_user: User = Depends(require_master_admin),
    db: Session = Depends(get_db_session)
):
    """
    Export logs (CSV/JSON).
    Master admin only.
    """
    query = db.query(SystemLog)
    
    # Apply same filters as list_logs
    if level:
        try:
            level_enum = LogLevel[level.upper()]
            query = query.filter(SystemLog.level == level_enum)
        except KeyError:
            pass
    
    if category:
        try:
            category_enum = LogCategory[category.upper()]
            query = query.filter(SystemLog.category == category_enum)
        except KeyError:
            pass
    
    if user:
        query = query.filter(SystemLog.user_id == user)
    
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            query = query.filter(SystemLog.created_at >= from_dt)
        except ValueError:
            pass
    
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
            query = query.filter(SystemLog.created_at <= to_dt)
        except ValueError:
            pass
    
    # Get all logs (no pagination for export)
    logs = query.order_by(SystemLog.created_at.desc()).all()
    
    if format == "csv":
        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["id", "timestamp", "level", "category", "message", "userId", "ip", "metadata"])
        
        # Write data
        for log in logs:
            writer.writerow([
                log.id,
                log.created_at.isoformat(),
                log.level.value,
                log.category.value,
                log.message,
                log.user_id,
                log.ip,
                str(log.extra_metadata) if log.extra_metadata else ""
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    
    else:  # JSON
        import json
        log_list = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "timestamp": log.created_at.isoformat(),
                "level": log.level.value,
                "category": log.category.value,
                "message": log.message,
                "userId": log.user_id,
                "ip": log.ip,
                "metadata": log.extra_metadata
            }
            log_list.append(log_dict)
        
        json_content = json.dumps(log_list, indent=2)
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
