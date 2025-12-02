"""S3-based caching service for processed images."""
import hashlib
import io
import os
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError
import httpx


class S3Cache:
    """S3-based cache for storing and retrieving processed images."""

    def __init__(self):
        """Initialize S3 client with configuration from environment variables."""
        self.enabled = False
        # Support both S3_BUCKET_NAME and AWS_S3_BUCKET_NAME
        self.bucket_name = os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")

        # Check if S3 is configured
        if not self.bucket_name:
            print("S3 caching disabled: S3_BUCKET_NAME not set")
            return

        try:
            # Initialize S3 client (uses AWS credentials from environment)
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            # Verify bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.enabled = True
            print(f"S3 caching enabled: bucket={self.bucket_name}, region={self.region}")

        except ClientError as e:
            print(f"S3 caching disabled: Failed to initialize - {e}")
        except Exception as e:
            print(f"S3 caching disabled: Unexpected error - {e}")

    @staticmethod
    def generate_cache_key_from_hash(content: bytes, hat_scale: float = 1.0) -> str:
        """
        Generate a cache key from file content hash.

        Args:
            content: Raw file bytes
            hat_scale: Hat scale parameter

        Returns:
            Cache key string
        """
        # Create hash of content + hat_scale
        hasher = hashlib.sha256()
        hasher.update(content)
        hasher.update(str(hat_scale).encode())
        content_hash = hasher.hexdigest()

        return f"processed/{content_hash[:2]}/{content_hash}.jpg"

    @staticmethod
    async def generate_cache_key_from_url(url: str, hat_scale: float = 1.0) -> Optional[str]:
        """
        Generate a cache key from URL using ETag or Last-Modified headers.
        Falls back to URL hash if headers unavailable.

        Args:
            url: Image URL
            hat_scale: Hat scale parameter

        Returns:
            Cache key string or None if URL is unreachable
        """
        try:
            # Make HEAD request to get ETag/Last-Modified without downloading
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(url, follow_redirects=True)
                response.raise_for_status()

                # Try to get ETag (most reliable for cache invalidation)
                etag = response.headers.get("etag", "").strip('"')
                last_modified = response.headers.get("last-modified", "")

                # Create identifier from available headers
                identifier = etag or last_modified or url

                # Hash the identifier + hat_scale
                hasher = hashlib.sha256()
                hasher.update(identifier.encode())
                hasher.update(str(hat_scale).encode())
                cache_hash = hasher.hexdigest()

                return f"processed/{cache_hash[:2]}/{cache_hash}.jpg"

        except Exception as e:
            print(f"Failed to generate cache key from URL headers: {e}")
            # Fallback to URL hash
            hasher = hashlib.sha256()
            hasher.update(url.encode())
            hasher.update(str(hat_scale).encode())
            cache_hash = hasher.hexdigest()
            return f"processed/{cache_hash[:2]}/{cache_hash}.jpg"

    async def get_cached_image(self, cache_key: str) -> Optional[bytes]:
        """
        Retrieve cached image from S3.

        Args:
            cache_key: S3 object key

        Returns:
            Image bytes if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=cache_key
            )
            return response['Body'].read()

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # Cache miss - this is normal
                return None
            else:
                print(f"Error retrieving from S3 cache: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error retrieving from S3 cache: {e}")
            return None

    async def store_cached_image(
        self,
        cache_key: str,
        image_data: bytes,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Store processed image in S3 cache.

        Args:
            cache_key: S3 object key
            image_data: Processed image bytes
            metadata: Optional metadata to store with the image

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Prepare metadata
            s3_metadata = metadata or {}
            s3_metadata['ContentType'] = 'image/jpeg'

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=cache_key,
                Body=image_data,
                ContentType='image/jpeg',
                Metadata={k: str(v) for k, v in s3_metadata.items()},
                # Set cache control for CDN
                CacheControl='public, max-age=31536000'  # 1 year
            )

            print(f"Cached image to S3: {cache_key}")
            return True

        except Exception as e:
            print(f"Error storing to S3 cache: {e}")
            return False
