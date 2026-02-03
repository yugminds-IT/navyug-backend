"""
S3 client for file storage.
"""
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from pathlib import Path
from typing import Optional, BinaryIO, List
import logging
from io import BytesIO

from core.logger import logger

try:
    import config as _config
except ImportError:
    _config = None


class S3Client:
    """S3 client for storing and retrieving files. Supports college-specific buckets."""
    
    def __init__(
        self,
        default_bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        endpoint_url: Optional[str] = None,  # For S3-compatible services (MinIO, etc.)
        auto_create_buckets: bool = True
    ):
        """
        Initialize S3 client.
        
        Args:
            default_bucket_name: Default bucket name (for master admin or fallback)
            aws_access_key_id: AWS access key (or from env)
            aws_secret_access_key: AWS secret key (or from env)
            region_name: AWS region
            endpoint_url: Custom endpoint URL (for MinIO, etc.)
            auto_create_buckets: Automatically create buckets if they don't exist
        """
        self.default_bucket_name = default_bucket_name
        self.region_name = region_name
        self.auto_create_buckets = auto_create_buckets
        
        # Initialize boto3 client
        client_kwargs = {
            "region_name": region_name
        }
        
        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        
        self.s3_client = boto3.client("s3", **client_kwargs)
        self.s3_resource = boto3.resource("s3", **client_kwargs)
        
        # Cache for verified buckets
        self._verified_buckets = set()
        
        # Verify default bucket if provided
        if default_bucket_name:
            self._ensure_bucket_exists(default_bucket_name)
        
        logger.info(f"S3 client initialized (default bucket: {default_bucket_name})")
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure bucket exists, create if it doesn't."""
        if bucket_name in self._verified_buckets:
            return
        
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.debug(f"Bucket {bucket_name} exists")
            self._verified_buckets.add(bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404" or error_code == "403":
                # Bucket doesn't exist or no access, create it if auto_create enabled
                if self.auto_create_buckets:
                    try:
                        if self.region_name == "us-east-1":
                            self.s3_client.create_bucket(Bucket=bucket_name)
                        else:
                            self.s3_client.create_bucket(
                                Bucket=bucket_name,
                                CreateBucketConfiguration={"LocationConstraint": self.region_name}
                            )
                        logger.info(f"Created bucket: {bucket_name}")
                        self._verified_buckets.add(bucket_name)
                    except ClientError as create_error:
                        logger.error(f"Failed to create bucket {bucket_name}: {create_error}")
                        raise
                else:
                    logger.error(f"Bucket {bucket_name} does not exist and auto-create is disabled")
                    raise
            else:
                logger.error(f"Error checking bucket {bucket_name}: {e}")
                raise
    
    def get_bucket_name(self, college_code: Optional[str] = None) -> str:
        """
        Get bucket name. When USE_CAMPUS_SECURITY_BUCKET is True, always returns
        CAMPUS_SECURITY_BUCKET_NAME (single bucket). Otherwise uses college_code as bucket.
        """
        if _config and getattr(_config, "USE_CAMPUS_SECURITY_BUCKET", False):
            return self.default_bucket_name or ""
        if college_code:
            # Sanitize college code for S3 bucket name requirements
            # S3 requirements: lowercase, 3-63 chars, alphanumeric and hyphens only
            bucket_name = college_code.lower()
            # Replace spaces and underscores with hyphens
            bucket_name = bucket_name.replace(" ", "-").replace("_", "-")
            # Remove any invalid characters (keep only alphanumeric and hyphens)
            import re
            bucket_name = re.sub(r'[^a-z0-9\-]', '', bucket_name)
            # Remove consecutive hyphens
            bucket_name = re.sub(r'-+', '-', bucket_name)
            # Remove leading/trailing hyphens
            bucket_name = bucket_name.strip('-')
            # Ensure length (3-63 chars)
            if len(bucket_name) < 3:
                bucket_name = bucket_name + "001"  # Pad if too short
            if len(bucket_name) > 63:
                bucket_name = bucket_name[:63]
            
            # Ensure bucket exists
            self._ensure_bucket_exists(bucket_name)
            return bucket_name
        else:
            # Use default bucket (for master admin)
            if self.default_bucket_name:
                return self.default_bucket_name
            else:
                raise ValueError("No bucket name available: college_code not provided and no default bucket set")
    
    def upload_file(
        self,
        file_path: str,
        s3_key: str,
        college_code: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            file_path: Local file path
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            S3 URL of uploaded file
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
            
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            url = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"Uploaded file to S3: {url}")
            return url
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise
    
    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        college_code: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload a file-like object to S3.
        
        Args:
            file_obj: File-like object (BytesIO, file handle, etc.)
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            S3 URL of uploaded file
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
            
            self.s3_client.upload_fileobj(
                file_obj,
                bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            url = f"s3://{bucket_name}/{s3_key}"
            logger.info(f"Uploaded file object to S3: {url}")
            return url
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to upload file object to S3: {e}")
            raise
    
    def list_objects(self, prefix: str, college_code: Optional[str] = None, max_keys: int = 1000) -> List[str]:
        """
        List S3 object keys under a prefix.

        Args:
            prefix: S3 key prefix (e.g. colleges/college_id=CODE/students/)
            college_code: College code (used as bucket name)
            max_keys: Maximum number of keys to return

        Returns:
            List of S3 object keys (full keys, not just suffixes)
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            keys = []
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys):
                for obj in page.get("Contents") or []:
                    key = obj.get("Key")
                    if key:
                        keys.append(key)
            return keys
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to list S3 objects under {prefix}: {e}")
            raise

    def download_file(self, s3_key: str, local_path: str, college_code: Optional[str] = None) -> str:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key (path)
            local_path: Local file path to save to
            college_code: College code (used as bucket name)
            
        Returns:
            Local file path
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(bucket_name, s3_key, local_path)
            logger.info(f"Downloaded file from S3: {bucket_name}/{s3_key} -> {local_path}")
            return local_path
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise
    
    def download_fileobj(self, s3_key: str, college_code: Optional[str] = None) -> BytesIO:
        """
        Download a file from S3 as BytesIO.
        
        Args:
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            
        Returns:
            BytesIO object with file content
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            file_obj = BytesIO()
            self.s3_client.download_fileobj(bucket_name, s3_key, file_obj)
            file_obj.seek(0)
            logger.debug(f"Downloaded file object from S3: {bucket_name}/{s3_key}")
            return file_obj
            
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                logger.debug(f"Object not found in S3: {s3_key}")
            else:
                logger.error(f"Failed to download file object from S3: {e}")
            raise
        except BotoCoreError as e:
            logger.error(f"Failed to download file object from S3: {e}")
            raise
    
    def delete_file(self, s3_key: str, college_code: Optional[str] = None) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            
        Returns:
            True if successful
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.info(f"Deleted file from S3: {bucket_name}/{s3_key}")
            return True
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to delete file from S3: {e}")
            raise
    
    def file_exists(self, s3_key: str, college_code: Optional[str] = None) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            
        Returns:
            True if file exists
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise
    
    def get_presigned_url(self, s3_key: str, college_code: Optional[str] = None, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            s3_key: S3 object key (path)
            college_code: College code (used as bucket name)
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Presigned URL
        """
        try:
            bucket_name = self.get_bucket_name(college_code)
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": s3_key},
                ExpiresIn=expiration
            )
            return url
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def get_presigned_url_for_path(self, s3_path_or_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Turn an S3 path or key into a presigned HTTPS URL.
        s3_path_or_key: either "s3://bucket-name/key/path" or just "key/path".
        Returns None if invalid or on error; otherwise HTTPS presigned URL.
        """
        if not s3_path_or_key or not s3_path_or_key.strip():
            return None
        s3_path = s3_path_or_key.strip()
        if s3_path.startswith("s3://"):
            parts = s3_path.replace("s3://", "", 1).split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.default_bucket_name
            if not bucket:
                try:
                    import os
                    bucket = os.environ.get("AWS_S3_BUCKET") or (
                        _config.CAMPUS_SECURITY_BUCKET_NAME if _config and getattr(_config, "CAMPUS_SECURITY_BUCKET_NAME", None) else "campus-security-bucket"
                    )
                except Exception:
                    bucket = "campus-security-bucket"
            key = s3_path
        if not key:
            return None
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError:
            return None
