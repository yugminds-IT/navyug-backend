"""
S3 path generation utilities.
Supports (1) campus-security single bucket with structured keys, (2) legacy college-per-bucket.
"""
from typing import Optional
from database.models import User, UserRole

try:
    import config as _config
except ImportError:
    _config = None


def _use_campus_bucket() -> bool:
    return bool(_config and getattr(_config, "USE_CAMPUS_SECURITY_BUCKET", False))


def _campus_college_prefix(college_code: str) -> str:
    """colleges/college_id=COL001"""
    return f"colleges/college_id={college_code}"


def campus_student_profile_passport_path(college_code: str, student_id_str: str) -> str:
    """colleges/college_id=COL001/students/student_id=STU001/profile/passport.jpg"""
    return f"{_campus_college_prefix(college_code)}/students/student_id={student_id_str}/profile/passport.jpg"


def campus_student_face_gallery_path(college_code: str, student_id_str: str, filename: str) -> str:
    """colleges/college_id=COL001/students/student_id=STU001/face-gallery/img_01.jpg"""
    return f"{_campus_college_prefix(college_code)}/students/student_id={student_id_str}/face-gallery/{filename}"


def campus_student_embeddings_path(college_code: str, student_id_str: str, filename: str = "face_embedding.npy") -> str:
    """colleges/college_id=COL001/students/student_id=STU001/embeddings/face_embedding.npy"""
    return f"{_campus_college_prefix(college_code)}/students/student_id={student_id_str}/embeddings/{filename}"


def campus_report_raw_video_path(college_code: str, report_id_str: str, filename: str) -> str:
    """colleges/college_id=COL001/reports/report_id=RPT001/raw-video/incident.mp4"""
    return f"{_campus_college_prefix(college_code)}/reports/report_id={report_id_str}/raw-video/{filename}"


def campus_report_frames_path(college_code: str, report_id_str: str, frame_filename: str) -> str:
    """colleges/college_id=COL001/reports/report_id=RPT001/frames/frame_000001.jpg"""
    return f"{_campus_college_prefix(college_code)}/reports/report_id={report_id_str}/frames/{frame_filename}"


def campus_report_detected_faces_path(college_code: str, report_id_str: str, filename: str) -> str:
    """colleges/college_id=COL001/reports/report_id=RPT001/detected-faces/detected_001.jpg"""
    return f"{_campus_college_prefix(college_code)}/reports/report_id={report_id_str}/detected-faces/{filename}"


def campus_report_embeddings_path(college_code: str, report_id_str: str, filename: str = "detected_embeddings.npy") -> str:
    """colleges/college_id=COL001/reports/report_id=RPT001/embeddings/detected_embeddings.npy"""
    return f"{_campus_college_prefix(college_code)}/reports/report_id={report_id_str}/embeddings/{filename}"


def get_college_code_from_user(user: User) -> Optional[str]:
    """
    Get college code from user based on their role.
    
    Args:
        user: User object
        
    Returns:
        College code or None
    """
    if user.role == UserRole.MASTER_ADMIN:
        # Master admin doesn't belong to a college
        return None
    elif user.role == UserRole.MANAGEMENT:
        # Management manages a college (use managed_college_id or college_id for legacy data)
        college = user.managed_college if user.managed_college_id is not None else user.college
        if college:
            return college.college_code
        return None
    elif user.role in [UserRole.FACULTY, UserRole.STUDENT]:
        # Faculty and students belong to a college
        if user.college:
            return user.college.college_code
        return None
    
    return None


def build_s3_path(
    college_code: Optional[str],
    file_type: str,
    identifier: str,
    filename: Optional[str] = None
) -> str:
    """
    Build S3 path (bucket name is college code, so path is just file_type/identifier/filename).
    
    Structure: {file_type}/{identifier}/{filename}
    Note: Bucket name itself is the college code, so path doesn't include it.
    
    Args:
        college_code: College code (used as bucket name, not in path)
        file_type: Type of file (videos, faces, results, debug)
        identifier: Unique identifier (video_id, person_id, etc.)
        filename: Optional filename
        
    Returns:
        S3 key path (without bucket name)
    """
    base_path = f"{file_type}/{identifier}"
    
    if filename:
        return f"{base_path}/{filename}"
    return base_path


def get_video_s3_path(college_code: Optional[str], video_id: str, file_ext: str) -> str:
    """
    Get S3 path for video file.
    
    Structure: {college_code}/videos/{video_id}{file_ext}
    
    Args:
        college_code: College code
        video_id: Video ID
        file_ext: File extension
        
    Returns:
        S3 key path
    """
    return build_s3_path(college_code, "videos", video_id, f"{video_id}{file_ext}")


def get_face_s3_path(college_code: Optional[str], person_id: str, filename: str = "face.jpg") -> str:
    """
    Get S3 path for face image.
    When USE_CAMPUS_SECURITY_BUCKET: colleges/college_id=X/students/student_id=Y/profile/passport.jpg
    Otherwise: faces/{person_id}/{filename}
    """
    if college_code and _use_campus_bucket():
        return campus_student_profile_passport_path(college_code, person_id)
    return build_s3_path(college_code, "faces", person_id, filename)


def get_result_s3_path(college_code: Optional[str], video_id: str, filename: str = "results.json") -> str:
    """
    Get S3 path for processing results.
    
    Structure: {college_code}/results/{video_id}/{filename}
    
    Args:
        college_code: College code
        video_id: Video ID
        filename: Results filename
        
    Returns:
        S3 key path
    """
    return build_s3_path(college_code, "results", video_id, filename)


def get_debug_s3_path(college_code: Optional[str], video_id: str, face_filename: str) -> str:
    """
    Get S3 path for debug face images.
    
    Structure: {college_code}/debug/{video_id}/{face_filename}
    
    Args:
        college_code: College code
        video_id: Video ID
        face_filename: Debug face image filename
        
    Returns:
        S3 key path
    """
    return build_s3_path(college_code, "debug", video_id, face_filename)


def get_report_media_s3_path(
    college_code: Optional[str], report_id: int, media_id: str, file_ext: str, report_id_str: Optional[str] = None
) -> str:
    """
    Get S3 path for report media files.
    When USE_CAMPUS_SECURITY_BUCKET: .../reports/report_id=RPT001/raw-video/{media_id}{file_ext}
    Otherwise: reports/{report_id}/{media_id}{file_ext}
    """
    if college_code and _use_campus_bucket():
        rid = report_id_str or str(report_id)
        return campus_report_raw_video_path(college_code, rid, f"{media_id}{file_ext}")
    return build_s3_path(college_code, "reports", str(report_id), f"{media_id}{file_ext}")


def get_report_detected_face_s3_path(
    college_code: Optional[str], report_id: int, person_id: str, face_id: str, report_id_str: Optional[str] = None
) -> str:
    """
    Get S3 path for a detected face crop.
    When USE_CAMPUS_SECURITY_BUCKET: .../reports/report_id=RPT001/detected-faces/detected_001.jpg
    Otherwise: reports/{report_id}/detected/{person_id}_{face_id}.jpg
    """
    if college_code and _use_campus_bucket():
        rid = report_id_str or str(report_id)
        return campus_report_detected_faces_path(college_code, rid, f"detected_{face_id}.jpg")
    return build_s3_path(college_code, "reports", str(report_id), f"detected/{person_id}_{face_id}.jpg")
