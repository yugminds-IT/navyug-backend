"""
Input validation utilities for the Face Recognition System.
"""
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for use
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove directory separators and path components
    filename = os.path.basename(filename)
    
    # Remove dangerous characters (keep alphanumeric, dots, dashes, underscores)
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in "._-":
            safe_chars.append(char)
        else:
            safe_chars.append("_")  # Replace dangerous chars with underscore
    
    sanitized = "".join(safe_chars)
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        raise ValueError("Filename became empty after sanitization")
    
    return sanitized


def validate_video_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file is a valid, readable video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if file is valid video
        - error_message: Error description if invalid, None if valid
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file size (basic check)
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "File is empty"
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "Could not open video file (invalid format or corrupted)"
        
        # Check if we can read at least one frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if frame_count <= 0:
            return False, "Video has no frames"
        
        if fps <= 0:
            return False, "Video has invalid frame rate"
        
        if width <= 0 or height <= 0:
            return False, "Video has invalid dimensions"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating video file: {str(e)}"


def validate_video_extension(filename: str, allowed_extensions: set) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: Set of allowed extensions (e.g., {".mp4", ".avi"})
        
    Returns:
        True if extension is allowed
    """
    if not filename:
        return False
    
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def validate_file_size(file_size: int, max_size_bytes: int) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Args:
        file_size: File size in bytes
        max_size_bytes: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size <= 0:
        return False, "File size must be greater than 0"
    
    if file_size > max_size_bytes:
        max_size_mb = max_size_bytes / (1024 * 1024)
        actual_size_mb = file_size / (1024 * 1024)
        return False, f"File too large: {actual_size_mb:.2f}MB (max: {max_size_mb:.2f}MB)"
    
    return True, None


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string format.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        True if valid UUID format
    """
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, AttributeError):
        return False
