"""
Video job service for database operations.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import VideoJob, JobStatus, MatchedFace, Person
from core.logger import logger
import json
import numpy as np


class VideoService:
    """Service for video job operations."""
    
    @staticmethod
    def create_job(
        db: Session,
        video_id: str,
        filename: str,
        user_id: Optional[int] = None,
        s3_video_path: Optional[str] = None,
        local_video_path: Optional[str] = None,
        file_size_mb: Optional[float] = None
    ) -> VideoJob:
        """
        Create a new video job.
        
        Args:
            db: Database session
            video_id: Unique video ID (UUID)
            filename: Original filename
            user_id: Optional user ID
            s3_video_path: S3 path to video
            local_video_path: Local path to video
            file_size_mb: File size in MB
            
        Returns:
            Created VideoJob
        """
        job = VideoJob(
            video_id=video_id,
            user_id=user_id,
            filename=filename,
            s3_video_path=s3_video_path,
            local_video_path=local_video_path,
            file_size_mb=file_size_mb,
            status=JobStatus.PENDING
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info(f"Created video job: {video_id}")
        return job
    
    @staticmethod
    def get_job(db: Session, video_id: str) -> Optional[VideoJob]:
        """Get video job by ID."""
        return db.query(VideoJob).filter(VideoJob.video_id == video_id).first()
    
    @staticmethod
    def update_job_status(
        db: Session,
        video_id: str,
        status: JobStatus,
        progress_percentage: Optional[float] = None,
        progress_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[VideoJob]:
        """Update job status."""
        job = VideoService.get_job(db, video_id)
        if not job:
            return None
        
        job.status = status
        if progress_percentage is not None:
            job.progress_percentage = progress_percentage
        if progress_data is not None:
            job.progress_data = progress_data
        if error_message is not None:
            job.error_message = error_message
        
        # Update timestamps
        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status == JobStatus.COMPLETED:
            job.completed_at = datetime.utcnow()
            if job.started_at:
                job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()
        elif status == JobStatus.FAILED:
            job.failed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def save_results(
        db: Session,
        video_id: str,
        result: Dict[str, Any],
        matched_persons: list[Dict[str, Any]]
    ) -> Optional[VideoJob]:
        """Save processing results."""
        job = VideoService.get_job(db, video_id)
        if not job:
            return None
        
        job.result = result
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        
        if job.started_at:
            job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()
        
        # Save matched faces
        for match_data in matched_persons:
            matched_face = MatchedFace(
                video_job_id=job.id,
                person_id=match_data.get("person_id"),
                confidence=match_data.get("confidence", 0.0),
                total_appearances=match_data.get("total_appearances", 1),
                frames_seen=match_data.get("frames_seen", []),
                best_face_quality=match_data.get("best_face_quality"),
                metadata=match_data.get("metadata")
            )
            db.add(matched_face)
        
        db.commit()
        db.refresh(job)
        logger.info(f"Saved results for video job: {video_id}")
        return job
    
    @staticmethod
    def get_user_jobs(
        db: Session,
        user_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> list[VideoJob]:
        """Get user's video jobs."""
        return db.query(VideoJob).filter(
            VideoJob.user_id == user_id
        ).order_by(desc(VideoJob.uploaded_at)).limit(limit).offset(offset).all()
    
    @staticmethod
    def delete_job(db: Session, video_id: str) -> bool:
        """Delete video job and related data."""
        job = VideoService.get_job(db, video_id)
        if not job:
            return False
        
        db.delete(job)
        db.commit()
        logger.info(f"Deleted video job: {video_id}")
        return True
