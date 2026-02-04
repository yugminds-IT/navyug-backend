"""
Configuration settings for the face recognition system.
Supports both development and production environments via environment variables.
"""
import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
COLLEGE_FACES_DIR = BASE_DIR / "college_faces"
DEBUG_FACES_DIR = BASE_DIR / "debug_faces"

# Create directories (skip in read-only containers; app will still start)
try:
    UPLOADS_DIR.mkdir(exist_ok=True)
    if not os.getenv("USE_S3_ONLY", "true").lower() == "true":
        COLLEGE_FACES_DIR.mkdir(exist_ok=True)
        DEBUG_FACES_DIR.mkdir(exist_ok=True)
except OSError:
    pass

# ============================================================================
# Database Configuration (PostgreSQL)
# ============================================================================
_raw_url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/face_recognition"
)
# SQLAlchemy 2 loads dialect "postgresql", not "postgres"; normalize Heroku-style URLs
if _raw_url.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + _raw_url[len("postgres://"):]
else:
    DATABASE_URL = _raw_url
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

# ============================================================================
# S3 Configuration
# ============================================================================
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
# Default bucket name (used for master admin or fallback)
# Note: Each college gets its own bucket (bucket name = college code)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "college")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", None)
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", None)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", None)  # For MinIO, etc.

# Campus Security Bucket: single bucket with structured keys
# When True, all S3 objects go to CAMPUS_SECURITY_BUCKET_NAME with keys like:
#   colleges/college_id=COL001/students/student_id=STU001/profile/passport.jpg
#   colleges/college_id=COL001/students/student_id=STU001/face-gallery/img_01.jpg
#   colleges/college_id=COL001/reports/report_id=RPT001/raw-video/incident.mp4
USE_CAMPUS_SECURITY_BUCKET = os.getenv("USE_CAMPUS_SECURITY_BUCKET", "true").lower() == "true"
CAMPUS_SECURITY_BUCKET_NAME = os.getenv("CAMPUS_SECURITY_BUCKET_NAME", "campus-security-bucket")

# When True: do not load/store college_faces, debug_faces, or report media from local disk.
# Face DB for detection is fetched from S3 first (students/.../profile, face-gallery, embeddings);
# DB/Person table is used only as fallback when S3 has no data.
USE_S3_ONLY = os.getenv("USE_S3_ONLY", "true").lower() == "true"

# S3 Paths (when not using campus security bucket: bucket = college code)
S3_VIDEOS_PREFIX = "videos"
S3_FACES_PREFIX = "faces"
S3_RESULTS_PREFIX = "results"
S3_DEBUG_PREFIX = "debug"

# ============================================================================
# Security Configuration
# ============================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", None)  # For encrypting sensitive data (Fernet key)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", "168"))  # 7 days
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))  # Max failed login attempts before lockout
LOCKOUT_DURATION_MINUTES = int(os.getenv("LOCKOUT_DURATION_MINUTES", "30"))  # Account lockout duration

# CORS Settings - include your frontend origin (e.g. Vite default 5173, Next 3000)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000,http://localhost:5173,http://localhost:5174,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8000").split(",") if o.strip()]
CORS_ALLOW_CREDENTIALS = True

# Trusted Hosts
TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

# ============================================================================
# Face Recognition Settings
# ============================================================================
FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))
FRAME_EXTRACTION_RATE = int(os.getenv("FRAME_EXTRACTION_RATE", "1"))

# Enhanced Matching Settings
USE_QUALITY_WEIGHTED_MATCHING = os.getenv("USE_QUALITY_WEIGHTED_MATCHING", "true").lower() == "true"
USE_ADAPTIVE_THRESHOLD = os.getenv("USE_ADAPTIVE_THRESHOLD", "true").lower() == "true"
USE_STATISTICAL_MATCHING = os.getenv("USE_STATISTICAL_MATCHING", "true").lower() == "true"

# Multi-Model Embedding Fusion Settings
USE_MULTI_MODEL_FUSION = os.getenv("USE_MULTI_MODEL_FUSION", "true").lower() == "true"
FUSION_METHOD = os.getenv("FUSION_METHOD", "average")
USE_ARCFACE = os.getenv("USE_ARCFACE", "true").lower() == "true"
USE_FACENET = os.getenv("USE_FACENET", "true").lower() == "true"

# Face Detection Enhancement Settings
ENABLE_IMAGE_PREPROCESSING = os.getenv("ENABLE_IMAGE_PREPROCESSING", "true").lower() == "true"
ENABLE_GAMMA_CORRECTION = os.getenv("ENABLE_GAMMA_CORRECTION", "true").lower() == "true"
GAMMA_VALUE = float(os.getenv("GAMMA_VALUE", "1.2"))
ENABLE_FACE_ALIGNMENT = os.getenv("ENABLE_FACE_ALIGNMENT", "true").lower() == "true"
ENABLE_MULTI_SCALE_DETECTION = os.getenv("ENABLE_MULTI_SCALE_DETECTION", "false").lower() == "true"
DETECTION_SCALES = [0.75, 1.0, 1.25]

# Detection Model Settings
DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "30"))

# NMS Settings
ENABLE_NMS = os.getenv("ENABLE_NMS", "true").lower() == "true"
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.5"))

# Duplicate Detection Settings
REMOVE_DUPLICATE_IMAGES = os.getenv("REMOVE_DUPLICATE_IMAGES", "true").lower() == "true"
DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.85"))
DUPLICATE_DETECTION_DRY_RUN = os.getenv("DUPLICATE_DETECTION_DRY_RUN", "false").lower() == "true"
VIDEO_DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("VIDEO_DUPLICATE_SIMILARITY_THRESHOLD", "0.80"))

# Debug Settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SAVE_DETECTED_FACES = os.getenv("SAVE_DETECTED_FACES", "true").lower() == "true"
MAX_DEBUG_FACES_PER_VIDEO = int(os.getenv("MAX_DEBUG_FACES_PER_VIDEO", "50"))
MAX_IMAGES_PER_FACE = int(os.getenv("MAX_IMAGES_PER_FACE", "5"))  # Max detected face images saved per unique face

# Report S3 upload (frames + detected-faces folders)
REPORT_UPLOAD_FRAMES_TO_S3 = os.getenv("REPORT_UPLOAD_FRAMES_TO_S3", "true").lower() == "true"
REPORT_FRAME_UPLOAD_INTERVAL = int(os.getenv("REPORT_FRAME_UPLOAD_INTERVAL", "5"))  # upload every Nth frame (1 = every frame)

# Performance Optimization Settings
ENABLE_BATCH_MATCHING = os.getenv("ENABLE_BATCH_MATCHING", "true").lower() == "true"
BATCH_SIZE_FOR_MATCHING = int(os.getenv("BATCH_SIZE_FOR_MATCHING", "100"))
OPTIMIZE_DEDUPLICATION_CACHE = os.getenv("OPTIMIZE_DEDUPLICATION_CACHE", "true").lower() == "true"

# Model Settings
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
GPU_ID = int(os.getenv("GPU_ID", "0"))
CTX_ID = -1 if not USE_GPU else GPU_ID

# ============================================================================
# API Settings
# ============================================================================
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}

# ============================================================================
# Email Configuration (SMTP)
# ============================================================================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Campus Safety App")
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))

# ============================================================================
# Application Settings
# ============================================================================
APP_NAME = os.getenv("APP_NAME", "Face Recognition API")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

# ============================================================================
# Global Instances (initialized at startup)
# ============================================================================
# Database instance (initialized in app.py)
db: Optional[object] = None

# S3 client instance (initialized in app.py)
s3_client: Optional[object] = None

# Face Database (global instance)
face_db = None

# Job Status Store (deprecated - now using database)
# Kept for backward compatibility during migration
job_store = {}
