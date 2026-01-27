import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
COLLEGE_FACES_DIR = BASE_DIR / "college_faces"
DEBUG_FACES_DIR = BASE_DIR / "debug_faces"  # NEW: Save detected faces for debugging

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
COLLEGE_FACES_DIR.mkdir(exist_ok=True)
DEBUG_FACES_DIR.mkdir(exist_ok=True)  # NEW

# Face Recognition Settings
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection (lowered for more detections)
FACE_MATCH_THRESHOLD = 0.4  # Minimum similarity for face matching (lowered for robustness)
# Note: Lower threshold = more lenient matching (better for beard, age, costume variations)
# Recommended: 0.3-0.5 for robust matching, 0.6-0.8 for strict matching
FRAME_EXTRACTION_RATE = 1  # Frames per second to extract

# Enhanced Matching Settings
USE_QUALITY_WEIGHTED_MATCHING = True  # Weight matches by face quality (improves accuracy)
USE_ADAPTIVE_THRESHOLD = True  # Adjust matching threshold based on face quality
USE_STATISTICAL_MATCHING = True  # Use statistical analysis (top-3 average) for better robustness
# Quality weighting: High-quality faces get confidence boost, low-quality get penalty
# Adaptive threshold: High-quality faces use lower threshold (more lenient), low-quality use higher (more strict)

# Multi-Model Embedding Fusion Settings
USE_MULTI_MODEL_FUSION = True  # Enable multi-model embedding fusion (ArcFace + FaceNet + InsightFace)
# Note: ArcFace uses InsightFace's built-in ArcFace model (no extra install needed)
# FaceNet requires: pip install facenet-pytorch torch torchvision
FUSION_METHOD = "average"  # "average" or "concatenate"
# "average": Average normalized embeddings (keeps 512-dim, better for similarity)
# "concatenate": Concatenate embeddings (1536-dim with all 3 models, more information)
USE_ARCFACE = True  # Enable ArcFace (uses InsightFace's ArcFace - already available!)
USE_FACENET = True  # Enable FaceNet model (requires: pip install facenet-pytorch torch torchvision)
# If FaceNet installation fails, set USE_FACENET = False and system will use InsightFace + ArcFace

# Face Detection Enhancement Settings
ENABLE_IMAGE_PREPROCESSING = True  # Enable image preprocessing (brightness/contrast/sharpening) before detection
ENABLE_GAMMA_CORRECTION = True  # Enable gamma correction for better low-light/high-contrast handling
GAMMA_VALUE = 1.2  # Gamma value (1.0 = no change, >1.0 = lighten shadows, <1.0 = darken)

ENABLE_FACE_ALIGNMENT = True  # Enable face alignment before embedding (improves accuracy)
ENABLE_MULTI_SCALE_DETECTION = False  # Enable multi-scale detection (slower but detects more faces)
# Multi-scale detection tries different image sizes to catch small/large faces
# Scales to try: [0.75, 1.0, 1.25] - set to False to disable (faster)
DETECTION_SCALES = [0.75, 1.0, 1.25]  # Scales for multi-scale detection (only if ENABLE_MULTI_SCALE_DETECTION = True)

# Detection Model Settings
DETECTION_SIZE = (640, 640)  # Input size for detection model. (640, 640) is standard.
# Larger (e.g. 1280, 1280) = detects smaller faces but slower.
DETECTION_THRESHOLD = 0.5  # Internal score threshold for the detection model

MIN_FACE_SIZE = 30  # Minimum face size in pixels (width or height) - filters out very small faces

# NMS (Non-Maximum Suppression) Settings
ENABLE_NMS = True  # Enable NMS to remove duplicate face detections
NMS_IOU_THRESHOLD = 0.5  # IoU threshold for NMS (0.0-1.0)
# Lower values = more aggressive (removes more overlapping faces)
# Higher values = less aggressive (keeps more faces)
# Recommended: 0.4-0.6 for most cases

# Duplicate Image Detection Settings (for database images)
REMOVE_DUPLICATE_IMAGES = True  # Enable automatic duplicate detection and removal
DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # Minimum face embedding similarity to consider duplicates (0-1)
# Higher values (0.9+) = only exact duplicates, Lower values (0.8-) = more aggressive removal
DUPLICATE_DETECTION_DRY_RUN = False  # If True, only report duplicates without deleting files
# Set to True first to see what would be removed, then set to False to actually remove

# Video Face Deduplication Settings (for detected faces in videos)
VIDEO_DUPLICATE_SIMILARITY_THRESHOLD = 0.80  # Similarity threshold for video face deduplication
# Lower than database threshold because video faces may have more variation
# 0.75-0.85 recommended for video processing

# Debug Settings
DEBUG = True  # Enable debug logging (shows full tracebacks for errors)
SAVE_DETECTED_FACES = True  # Save detected faces to debug_faces/ for debugging
MAX_DEBUG_FACES_PER_VIDEO = 50  # Limit number of saved faces per video

# Performance Optimization Settings
ENABLE_BATCH_MATCHING = True  # Use batch matching for faster face recognition (recommended: True)
BATCH_SIZE_FOR_MATCHING = 100  # Process faces in batches of this size (larger = faster but more memory)
OPTIMIZE_DEDUPLICATION_CACHE = True  # Cache similarity matrix during deduplication (recommended: True)

# Model Settings
USE_GPU = False  # Set to True if GPU available
GPU_ID = 0  # GPU device ID
CTX_ID = -1 if not USE_GPU else GPU_ID  # -1 for CPU, 0+ for GPU

# API Settings
MAX_VIDEO_SIZE_MB = 500  # Maximum video upload size in MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}

# Job Status Store (in-memory for POC)
job_store = {}
# Structure: {video_id: {status, progress, result, error}}

# Face Database (global instance)
face_db = None
