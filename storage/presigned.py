"""
Presigned URL helper for API responses.
Converts S3 paths/keys to HTTPS presigned URLs so browsers can load images.
"""
from typing import Optional


def get_presigned_url(s3_path_or_key: Optional[str], expires_in: int = 3600) -> Optional[str]:
    """
    Turn an S3 path or key into a presigned HTTPS URL for use in API JSON.
    s3_path_or_key: either "s3://bucket-name/key/path" or just "key/path".
    Returns None if invalid or S3 not configured; otherwise HTTPS presigned URL.
    """
    if not s3_path_or_key or not str(s3_path_or_key).strip():
        return None
    try:
        import config
        if not getattr(config, "s3_client", None):
            return None
        return config.s3_client.get_presigned_url_for_path(
            s3_path_or_key.strip(), expires_in=expires_in
        )
    except Exception:
        return None
