"""
Dashboard APIs for all roles.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from database.connection import Database
from database.models import (
    User, UserRole, Report, ReportStatus, IncidentType,
    DetectedFace, DisciplinaryAction
)
from auth.dependencies import get_current_user, get_management_college_id
from core.logger import logger
import config


router = APIRouter(prefix="/api/dashboard", tags=["dashboards"])


def get_db_session():
    """Get database session."""
    if not config.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    with config.db.get_session() as session:
        yield session


class DashboardResponse(BaseModel):
    """Dashboard response base."""
    pass


@router.get("/master")
async def master_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Master dashboard stats.
    Master admin only.
    """
    if current_user.role != UserRole.MASTER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Master admin only."
        )
    
    # Total reports
    total_reports = db.query(func.count(Report.id)).scalar() or 0
    
    # Total users
    total_users = db.query(func.count(User.id)).scalar() or 0
    
    # Total colleges
    from database.models import College
    total_colleges = db.query(func.count(College.id)).scalar() or 0
    
    # Recent reports (last 10)
    recent_reports = db.query(Report).order_by(Report.created_at.desc()).limit(10).all()
    recent_reports_list = []
    for report in recent_reports:
        recent_reports_list.append({
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "status": report.status.value,
            "createdAt": report.created_at.isoformat()
        })
    
    # Recent logs (last 10)
    from database.models import SystemLog
    recent_logs = db.query(SystemLog).order_by(SystemLog.created_at.desc()).limit(10).all()
    recent_logs_list = []
    for log in recent_logs:
        recent_logs_list.append({
            "id": log.id,
            "level": log.level.value,
            "category": log.category.value,
            "message": log.message,
            "timestamp": log.created_at.isoformat()
        })
    
    return {
        "totalReports": total_reports,
        "totalUsers": total_users,
        "totalColleges": total_colleges,
        "recentReports": recent_reports_list,
        "recentLogs": recent_logs_list
    }


@router.get("/management")
async def management_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Management dashboard stats.
    Management only.
    """
    if current_user.role != UserRole.MANAGEMENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Management only."
        )
    
    college_id = get_management_college_id(current_user)
    if not college_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is not assigned to a college. Please contact the administrator to assign you to a college."
        )
    
    # Total reports for college
    total_reports = db.query(func.count(Report.id)).filter(
        Report.college_id == college_id
    ).scalar() or 0
    
    # Pending reports
    pending_count = db.query(func.count(Report.id)).filter(
        Report.college_id == college_id,
        Report.status == ReportStatus.PENDING
    ).scalar() or 0
    
    # Resolved reports
    resolved_count = db.query(func.count(Report.id)).filter(
        Report.college_id == college_id,
        Report.status == ReportStatus.RESOLVED
    ).scalar() or 0
    
    # Recent reports (last 10)
    recent_reports = db.query(Report).filter(
        Report.college_id == college_id
    ).order_by(Report.created_at.desc()).limit(10).all()
    recent_reports_list = []
    for report in recent_reports:
        recent_reports_list.append({
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "status": report.status.value,
            "createdAt": report.created_at.isoformat()
        })
    
    # Stats by status
    stats_by_status = {}
    for status_enum in ReportStatus:
        count = db.query(func.count(Report.id)).filter(
            Report.college_id == college_id,
            Report.status == status_enum
        ).scalar() or 0
        stats_by_status[status_enum.value] = count
    
    # Stats by incident type
    stats_by_type = {}
    for incident_type in IncidentType:
        count = db.query(func.count(Report.id)).filter(
            Report.college_id == college_id,
            Report.incident_type == incident_type
        ).scalar() or 0
        stats_by_type[incident_type.value] = count
    
    # AI detections count
    ai_detections = db.query(func.count(DetectedFace.id)).join(Report).filter(
        Report.college_id == college_id
    ).scalar() or 0
    
    return {
        "collegeId": college_id,
        "totalReports": total_reports,
        "pendingCount": pending_count,
        "resolvedCount": resolved_count,
        "recentReports": recent_reports_list,
        "statsByStatus": stats_by_status,
        "statsByType": stats_by_type,
        "statsAiDetections": ai_detections
    }


@router.get("/faculty")
async def faculty_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Faculty dashboard stats.
    Faculty only.
    """
    if current_user.role != UserRole.FACULTY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Faculty only."
        )
    
    # Reports submitted by this faculty
    reports_submitted = db.query(func.count(Report.id)).filter(
        Report.reporter_id == current_user.id
    ).scalar() or 0
    
    # Reports resolved (where this faculty was the reporter)
    reports_resolved = db.query(func.count(Report.id)).filter(
        Report.reporter_id == current_user.id,
        Report.status == ReportStatus.RESOLVED
    ).scalar() or 0
    
    # Recent reports (last 10)
    recent_reports = db.query(Report).filter(
        Report.reporter_id == current_user.id
    ).order_by(Report.created_at.desc()).limit(10).all()
    recent_reports_list = []
    for report in recent_reports:
        recent_reports_list.append({
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "status": report.status.value,
            "createdAt": report.created_at.isoformat()
        })
    
    return {
        "reportsSubmitted": reports_submitted,
        "reportsResolved": reports_resolved,
        "recentReports": recent_reports_list
    }


@router.get("/student")
async def student_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Student dashboard stats.
    Student only.
    """
    if current_user.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Student only."
        )
    
    # Reports submitted by this student
    reports_submitted = db.query(func.count(Report.id)).filter(
        Report.reporter_id == current_user.id
    ).scalar() or 0
    
    # Points earned (from resolved reports)
    # Simple rule: 10 points per resolved report
    resolved_reports = db.query(func.count(Report.id)).filter(
        Report.reporter_id == current_user.id,
        Report.status == ReportStatus.RESOLVED
    ).scalar() or 0
    points_earned = resolved_reports * 10
    
    # Recent reports (last 10)
    recent_reports = db.query(Report).filter(
        Report.reporter_id == current_user.id
    ).order_by(Report.created_at.desc()).limit(10).all()
    recent_reports_list = []
    for report in recent_reports:
        recent_reports_list.append({
            "id": report.id,
            "reportId": report.report_id,
            "incidentType": report.incident_type.value,
            "location": report.location,
            "status": report.status.value,
            "createdAt": report.created_at.isoformat()
        })
    
    return {
        "reportsSubmitted": reports_submitted,
        "pointsEarned": points_earned,
        "recentReports": recent_reports_list
    }
