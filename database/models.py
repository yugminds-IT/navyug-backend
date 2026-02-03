"""
Database models for the face recognition system.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, 
    ForeignKey, JSON, Enum as SQLEnum, Index, TypeDecorator
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


# ============================================================================
# Custom Type Decorator for Enum Values
# ============================================================================

class EnumValue(TypeDecorator):
    """Type decorator to ensure enum values (not names) are stored."""
    impl = String
    cache_ok = True
    
    def __init__(self, enum_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_class = enum_class
    
    def process_bind_param(self, value, dialect):
        """Convert enum to its value when writing to database."""
        if value is None:
            return None
        if isinstance(value, enum.Enum):
            return value.value
        return value
    
    def process_result_value(self, value, dialect):
        """Convert database value back to enum when reading."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return self.enum_class(value)
            except ValueError:
                return value
        return value


# ============================================================================
# Enums - Must be defined before models that use them
# ============================================================================

class JobStatus(str, enum.Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserRole(str, enum.Enum):
    """User roles for authorization."""
    MASTER_ADMIN = "master_admin"
    MASTER = "master"  # Alias for master_admin per spec
    MANAGEMENT = "management"
    FACULTY = "faculty"
    STUDENT = "student"


class UserStatus(str, enum.Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class IncidentType(str, enum.Enum):
    """Incident report types."""
    RAGGING = "ragging"
    FIGHTING = "fighting"
    MISBEHAVIOUR = "misbehaviour"
    MONEY = "money"
    FACULTY = "faculty"
    OTHERS = "others"


class ReportStatus(str, enum.Enum):
    """Report lifecycle status."""
    PENDING = "pending"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    FAKE = "fake"  # Management marks when student uploads fake video; reporter data visible to management


class ReportReporterType(str, enum.Enum):
    """Reporter type for reports."""
    ANONYMOUS = "anonymous"
    IDENTIFIED = "identified"


class ActionType(str, enum.Enum):
    """Disciplinary action types."""
    WARNING = "warning"
    SUSPENSION_1D = "suspension_1d"
    SUSPENSION_3D = "suspension_3d"
    SUSPENSION_7D = "suspension_7d"
    COUNSELING = "counseling"
    EXPULSION = "expulsion"


class StudentStatus(str, enum.Enum):
    """Student disciplinary status."""
    ACTIVE = "active"
    WARNING = "warning"
    SUSPENDED = "suspended"


class FacultyStatus(str, enum.Enum):
    """Faculty account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"


class CollegeStatus(str, enum.Enum):
    """College status."""
    ACTIVE = "active"
    INACTIVE = "inactive"


class LogLevel(str, enum.Enum):
    """System log levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class LogCategory(str, enum.Enum):
    """System log categories."""
    AUTHENTICATION = "authentication"
    REPORT = "report"
    USER_MANAGEMENT = "user_management"
    SYSTEM = "system"
    DATABASE = "database"
    AI_DETECTION = "ai_detection"


class MediaType(str, enum.Enum):
    """Media types for reports."""
    IMAGE = "image"
    VIDEO = "video"


class StudentMediaType(str, enum.Enum):
    """Student profile media types (S3-linked)."""
    PASSPORT = "passport"
    FACE_GALLERY = "face_gallery"


# ============================================================================
# Models
# ============================================================================

class College(Base):
    """College/Institution model for multi-tenant support."""
    __tablename__ = "colleges"
    
    id = Column(Integer, primary_key=True, index=True)
    college_code = Column(String(50), unique=True, index=True, nullable=False)  # Unique college identifier
    name = Column(String(255), nullable=False)
    address = Column(Text, nullable=True)
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    website = Column(String(255), nullable=True)
    status = Column(EnumValue(CollegeStatus), default=CollegeStatus.ACTIVE, nullable=False)
    total_students = Column(Integer, default=0, nullable=False)
    total_faculty = Column(Integer, default=0, nullable=False)
    total_reports = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)  # Keep for backward compatibility
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)  # Master admin who created
    
    # Relationships
    users = relationship("User", back_populates="college", foreign_keys="User.college_id")
    management_users = relationship("User", back_populates="managed_college", foreign_keys="User.managed_college_id")
    departments = relationship("CollegeDepartment", back_populates="college", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="college")
    
    __table_args__ = (
        Index('idx_college_code', 'college_code'),
        Index('idx_college_active', 'is_active'),
        Index('idx_college_status', 'status'),
    )


class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    login_id = Column(String(100), unique=True, index=True, nullable=False)  # Login ID (username for most, college_id for students)
    email = Column(String(255), nullable=True, index=True)  # Email (required for master admin, management, faculty, student)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)  # Name (required for faculty, student)
    role = Column(EnumValue(UserRole), nullable=False)
    
    # Master Admin fields
    # (email and password are in base fields)
    
    # Management fields
    college_name = Column(String(255), nullable=True)  # College name (required for management)
    phone = Column(String(50), nullable=True)  # Phone number (required for management)
    address = Column(Text, nullable=True)  # Address (optional for management)
    username = Column(String(100), nullable=True)  # Username (required for management)
    
    # Faculty fields
    faculty_id = Column(String(100), nullable=True)  # Faculty ID (optional)
    department_name = Column(String(255), nullable=True)  # Department name (required for faculty)
    
    # Student fields
    roll_number = Column(String(100), nullable=True)  # Roll number (optional)
    branch = Column(String(100), nullable=True)  # Branch (optional)
    year = Column(String(50), nullable=True)  # Year (optional)
    
    # College association
    college_id = Column(Integer, ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True)  # For faculty/students
    managed_college_id = Column(Integer, ForeignKey("colleges.id", ondelete="SET NULL"), nullable=True)  # For management
    department_id = Column(Integer, ForeignKey("college_departments.id", ondelete="SET NULL"), nullable=True)  # Department the student/faculty is tagged to
    
    # Management association (for students and faculty)
    management_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)  # Management user who manages this student/faculty
    
    # Student specific
    student_id = Column(String(50), unique=True, nullable=True, index=True)  # Generated unique ID (STU-{college_id}-{seq})
    college_student_id = Column(String(100), nullable=True)  # Alias for S3/paths; set to student_id when generated
    
    # Status flags
    status = Column(EnumValue(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)  # Keep for backward compatibility
    is_verified = Column(Boolean, default=False, nullable=False)
    is_locked = Column(Boolean, default=False, nullable=False)  # Account lockout
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)  # Temporary lockout
    
    # Additional fields per spec
    avatar_url = Column(String(512), nullable=True)
    incidents = Column(Integer, default=0, nullable=False)  # Count of incidents (students)
    last_incident_at = Column(DateTime, nullable=True)  # Last incident date (students)
    face_registered = Column(Boolean, default=False, nullable=False)  # Face in DB for AI (students)
    face_image_url = Column(String(512), nullable=True)  # Student face photo URL (or s3://)
    s3_bucket = Column(String(255), nullable=True)  # S3 bucket for face/passport (e.g. campus-security-bucket)
    s3_face_key = Column(String(512), nullable=True)  # S3 key for face/passport so we can get data from S3 via DB
    reports_submitted = Column(Integer, default=0, nullable=False)  # Faculty report count
    reports_resolved = Column(Integer, default=0, nullable=False)  # Faculty resolved count
    points = Column(Integer, default=0, nullable=False)  # Student reward points
    last_login_at = Column(DateTime, nullable=True)  # Last login (alias for last_login)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)
    
    # Created by (for management accounts created by master admin)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    college = relationship("College", foreign_keys=[college_id], back_populates="users")
    managed_college = relationship("College", foreign_keys=[managed_college_id], back_populates="management_users")
    college_department = relationship("CollegeDepartment", foreign_keys=[department_id], back_populates="users")
    creator = relationship("User", remote_side=[id], foreign_keys=[created_by])
    management = relationship("User", remote_side=[id], foreign_keys=[management_id], back_populates="managed_users", post_update=True)  # Management user for students/faculty (many-to-one)
    managed_users = relationship("User", foreign_keys=[management_id], back_populates="management")  # Students/faculty managed by this user (one-to-many)
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    video_jobs = relationship("VideoJob", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    reports_as_reporter = relationship("Report", back_populates="reporter", foreign_keys="Report.reporter_id")
    disciplinary_actions_created = relationship("DisciplinaryAction", back_populates="created_by_user", foreign_keys="DisciplinaryAction.created_by")
    disciplinary_actions_as_student = relationship("DisciplinaryAction", back_populates="student", foreign_keys="DisciplinaryAction.student_id")
    detected_faces = relationship("DetectedFace", back_populates="student")
    system_logs = relationship("SystemLog", back_populates="user")
    student_media = relationship("StudentMedia", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_login_id', 'login_id'),
        Index('idx_user_college', 'college_id'),
        Index('idx_user_department', 'department_id'),
        Index('idx_user_managed_college', 'managed_college_id'),
        Index('idx_user_management', 'management_id'),
        Index('idx_user_role', 'role'),
    )


class StudentMedia(Base):
    """Student profile media (passport + face-gallery) linked to S3. Get data from S3 using DB."""
    __tablename__ = "student_media"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    media_type = Column(EnumValue(StudentMediaType), nullable=False)  # passport | face_gallery
    s3_bucket = Column(String(255), nullable=True)  # S3 bucket name
    s3_key = Column(String(512), nullable=True)  # S3 object key (full path)
    file_url = Column(String(512), nullable=True)  # Denormalized URL (s3:// or presigned) for display
    filename = Column(String(255), nullable=True)  # e.g. passport.jpg, img_01.jpg
    display_order = Column(Integer, default=0, nullable=False)  # Order in face-gallery
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="student_media")
    
    __table_args__ = (
        Index('idx_student_media_user', 'user_id'),
        Index('idx_student_media_type', 'media_type'),
    )


class APIKey(Base):
    """API key model for programmatic access."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    access_key = Column(String(255), unique=True, index=True, nullable=False)  # Encrypted access key
    key_hash = Column(String(255), unique=True, index=True, nullable=False)  # Hashed for verification
    key_prefix = Column(String(20), nullable=False)  # First 8 chars for identification
    name = Column(String(255), nullable=True)  # User-friendly name
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_access', 'access_key'),
    )


class RefreshToken(Base):
    """Refresh token model for JWT refresh."""
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)  # Hashed token
    device_info = Column(String(255), nullable=True)  # Device/browser info
    ip_address = Column(String(45), nullable=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
    
    __table_args__ = (
        Index('idx_refresh_token_hash', 'token_hash'),
        Index('idx_refresh_expires', 'expires_at'),
        Index('idx_refresh_user', 'user_id'),
    )


class Session(Base):
    """User session model for session management."""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_key = Column(String(255), unique=True, index=True, nullable=False)  # Encrypted session key
    session_hash = Column(String(255), unique=True, index=True, nullable=False)  # Hashed for verification
    device_info = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_session_key', 'session_key'),
        Index('idx_session_hash', 'session_hash'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_user', 'user_id'),
    )


class Person(Base):
    """Person model for face database."""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(100), unique=True, index=True, nullable=False)  # e.g., STUDENT_001
    name = Column(String(255), nullable=False)
    s3_image_path = Column(String(500), nullable=True)  # S3 path to face image
    local_image_path = Column(String(500), nullable=True)  # Fallback local path
    embedding = Column(Text, nullable=True)  # JSON array of embeddings
    num_images = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata (renamed from 'metadata' - reserved keyword)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_person_id', 'person_id'),
        Index('idx_person_active', 'is_active'),
    )


class VideoJob(Base):
    """Video processing job model."""
    __tablename__ = "video_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(255), nullable=False)
    s3_video_path = Column(String(500), nullable=True)  # S3 path to video
    local_video_path = Column(String(500), nullable=True)  # Fallback local path
    file_size_mb = Column(Float, nullable=True)
    status = Column(EnumValue(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    progress_percentage = Column(Float, default=0.0, nullable=False)
    progress_data = Column(JSON, nullable=True)  # Detailed progress info
    result = Column(JSON, nullable=True)  # Processing results
    error_message = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="video_jobs")
    matched_faces = relationship("MatchedFace", back_populates="video_job", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_video_status', 'status'),
        Index('idx_video_user', 'user_id'),
        Index('idx_video_uploaded', 'uploaded_at'),
    )


class MatchedFace(Base):
    """Matched face results for a video job."""
    __tablename__ = "matched_faces"
    
    id = Column(Integer, primary_key=True, index=True)
    video_job_id = Column(Integer, ForeignKey("video_jobs.id", ondelete="CASCADE"), nullable=False)
    person_id = Column(String(100), ForeignKey("persons.person_id", ondelete="SET NULL"), nullable=True)
    confidence = Column(Float, nullable=False)
    total_appearances = Column(Integer, default=1, nullable=False)
    frames_seen = Column(JSON, nullable=True)  # List of frame IDs
    best_face_quality = Column(Float, nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata (renamed from 'metadata' - reserved keyword)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    video_job = relationship("VideoJob", back_populates="matched_faces")
    person = relationship("Person")
    
    __table_args__ = (
        Index('idx_matched_video', 'video_job_id'),
        Index('idx_matched_person', 'person_id'),
        Index('idx_matched_confidence', 'confidence'),
    )


class AuditLog(Base):
    """Audit log for security and compliance."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g., "video_upload", "person_add", "user_login"
    resource_type = Column(String(50), nullable=True)  # e.g., "video_job", "person", "user"
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    details = Column(JSON, nullable=True)  # Audit log details (not using 'metadata' to avoid conflict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_created', 'created_at'),
    )


class CollegeDepartment(Base):
    """College departments."""
    __tablename__ = "college_departments"
    
    id = Column(Integer, primary_key=True, index=True)
    college_id = Column(Integer, ForeignKey("colleges.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    college = relationship("College", back_populates="departments")
    users = relationship("User", back_populates="college_department", foreign_keys="User.department_id")

    __table_args__ = (
        Index('idx_dept_college', 'college_id'),
        Index('idx_dept_college_name', 'college_id', 'name', unique=True),
    )


class Report(Base):
    """Incident report model."""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(50), unique=True, index=True, nullable=False)  # Display ID e.g. RPT-FAC-013
    incident_type = Column(EnumValue(IncidentType), nullable=False)
    location = Column(String(255), nullable=False)
    occurred_at = Column(DateTime, nullable=False)
    description = Column(Text, nullable=True)
    witnesses = Column(String(500), nullable=True)
    reporter_type = Column(EnumValue(ReportReporterType), nullable=False)
    reporter_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # owner_id: internal owner (student/faculty who submitted the report), even when reporter_type=anonymous.
    # Used for permissions and "my reports" views; not exposed to other roles.
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    college_id = Column(Integer, ForeignKey("colleges.id", ondelete="CASCADE"), nullable=False)
    status = Column(EnumValue(ReportStatus), default=ReportStatus.PENDING, nullable=False)
    has_video = Column(Boolean, default=False, nullable=False)
    has_photo = Column(Boolean, default=False, nullable=False)
    ai_processed = Column(Boolean, default=False, nullable=False)
    reporter_reward_points = Column(Integer, nullable=True)  # Stars/points when status=resolved; visible to management and reporter
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    college = relationship("College", back_populates="reports")
    reporter = relationship("User", back_populates="reports_as_reporter", foreign_keys=[reporter_id])
    owner = relationship("User", foreign_keys=[owner_id])
    media = relationship("ReportMedia", back_populates="report", cascade="all, delete-orphan")
    detected_faces = relationship("DetectedFace", back_populates="report", cascade="all, delete-orphan")
    disciplinary_actions = relationship("DisciplinaryAction", back_populates="report")
    
    __table_args__ = (
        Index('idx_report_id', 'report_id'),
        Index('idx_report_college', 'college_id'),
        Index('idx_report_reporter', 'reporter_id'),
        Index('idx_report_status', 'status'),
        Index('idx_report_incident_type', 'incident_type'),
        Index('idx_report_college_status', 'college_id', 'status'),
        Index('idx_report_created', 'created_at'),
    )


class ReportMedia(Base):
    """Report media files (images/videos). S3-linked so we can get data from S3 via DB."""
    __tablename__ = "report_media"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    media_type = Column(EnumValue(MediaType), nullable=False)
    file_url = Column(String(512), nullable=False)  # S3 URL or local path
    s3_bucket = Column(String(255), nullable=True)  # S3 bucket (e.g. campus-security-bucket)
    s3_key = Column(String(512), nullable=True)  # S3 object key to get data from S3
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    report = relationship("Report", back_populates="media")
    
    __table_args__ = (
        Index('idx_media_report', 'report_id'),
    )


class DetectedFace(Base):
    """AI face detection results per report."""
    __tablename__ = "detected_faces"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)  # Snapshot of name
    department = Column(String(255), nullable=True)
    year = Column(String(50), nullable=True)
    confidence = Column(Integer, nullable=False)  # 0-100
    previous_incidents = Column(Integer, default=0, nullable=False)
    bounding_box = Column(JSON, nullable=True)  # { x, y, width, height }
    reference_image_url = Column(String(512), nullable=True)  # DB face image
    detected_image_url = Column(String(512), nullable=True)  # Cropped from media
    s3_bucket = Column(String(255), nullable=True)  # S3 bucket for detected crop
    s3_detected_key = Column(String(512), nullable=True)  # S3 key for detected crop (get from S3 via DB)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    report = relationship("Report", back_populates="detected_faces")
    student = relationship("User", back_populates="detected_faces")
    
    __table_args__ = (
        Index('idx_detected_report', 'report_id'),
        Index('idx_detected_student', 'student_id'),
    )


class DisciplinaryAction(Base):
    """Actions taken against students."""
    __tablename__ = "disciplinary_actions"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id", ondelete="SET NULL"), nullable=True)
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    action_type = Column(EnumValue(ActionType), nullable=False)
    notes = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)  # Management user
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    report = relationship("Report", back_populates="disciplinary_actions")
    student = relationship("User", back_populates="disciplinary_actions_as_student", foreign_keys=[student_id])
    created_by_user = relationship("User", back_populates="disciplinary_actions_created", foreign_keys=[created_by])
    
    __table_args__ = (
        Index('idx_action_student', 'student_id'),
        Index('idx_action_report', 'report_id'),
        Index('idx_action_created', 'created_at'),
    )


class OtpStore(Base):
    """OTP storage for email verification."""
    __tablename__ = "otp_store"
    
    email = Column(String(255), primary_key=True)
    otp = Column(String(10), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_otp_expires', 'expires_at'),
    )


class SystemLog(Base):
    """System logs for auditing and monitoring."""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(EnumValue(LogLevel), nullable=False)
    category = Column(EnumValue(LogCategory), nullable=False)
    message = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    ip = Column(String(45), nullable=True)  # IPv6 compatible
    extra_metadata = Column(JSON, nullable=True)  # Extra context (renamed from 'metadata' - reserved keyword)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="system_logs")
    
    __table_args__ = (
        Index('idx_log_level', 'level'),
        Index('idx_log_category', 'category'),
        Index('idx_log_user', 'user_id'),
        Index('idx_log_created', 'created_at'),
    )
