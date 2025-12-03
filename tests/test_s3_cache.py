"""Tests for S3 cache module."""
import hashlib
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from botocore.exceptions import ClientError


class TestS3Cache:
    """Tests for the S3Cache class."""

    def test_init_disabled_when_no_bucket_name(self, monkeypatch):
        """Test that S3Cache is disabled when no bucket name is set."""
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        
        from app.s3_cache import S3Cache
        cache = S3Cache()
        
        assert cache.enabled is False

    def test_init_enabled_with_s3_bucket_name(self, monkeypatch):
        """Test that S3Cache initializes with S3_BUCKET_NAME."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            assert cache.enabled is True
            assert cache.bucket_name == "test-bucket"

    def test_init_enabled_with_aws_s3_bucket_name(self, monkeypatch):
        """Test that S3Cache initializes with AWS_S3_BUCKET_NAME."""
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "aws-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            assert cache.enabled is True
            assert cache.bucket_name == "aws-bucket"

    def test_init_disabled_on_client_error(self, monkeypatch):
        """Test that S3Cache is disabled when bucket access fails."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.side_effect = ClientError(
                {'Error': {'Code': '403', 'Message': 'Forbidden'}},
                'HeadBucket'
            )
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            assert cache.enabled is False

    def test_init_uses_default_region(self, monkeypatch):
        """Test that S3Cache uses default region."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        monkeypatch.delenv("AWS_REGION", raising=False)
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            assert cache.region == "us-east-1"


class TestGenerateCacheKeyFromHash:
    """Tests for generate_cache_key_from_hash static method."""

    def test_generates_consistent_key(self):
        """Test that same content generates same key."""
        from app.s3_cache import S3Cache
        
        content = b"test image content"
        key1 = S3Cache.generate_cache_key_from_hash(content, 1.0)
        key2 = S3Cache.generate_cache_key_from_hash(content, 1.0)
        
        assert key1 == key2

    def test_different_content_generates_different_key(self):
        """Test that different content generates different keys."""
        from app.s3_cache import S3Cache
        
        key1 = S3Cache.generate_cache_key_from_hash(b"content1", 1.0)
        key2 = S3Cache.generate_cache_key_from_hash(b"content2", 1.0)
        
        assert key1 != key2

    def test_different_scale_generates_different_key(self):
        """Test that different hat_scale generates different keys."""
        from app.s3_cache import S3Cache
        
        content = b"test content"
        key1 = S3Cache.generate_cache_key_from_hash(content, 1.0)
        key2 = S3Cache.generate_cache_key_from_hash(content, 1.5)
        
        assert key1 != key2

    def test_key_format_is_correct(self):
        """Test that key has correct format (processed/{hash[:2]}/{hash}.jpg)."""
        from app.s3_cache import S3Cache
        
        key = S3Cache.generate_cache_key_from_hash(b"content", 1.0)
        
        assert key.startswith("processed/")
        assert key.endswith(".jpg")
        
        parts = key.split("/")
        assert len(parts) == 3
        assert len(parts[1]) == 2  # First 2 chars of hash
        assert len(parts[2]) == 64 + 4  # SHA256 hex (64) + ".jpg" (4)


class TestGenerateCacheKeyFromUrl:
    """Tests for generate_cache_key_from_url async method."""

    @pytest.mark.asyncio
    async def test_uses_etag_when_available(self):
        """Test that ETag is used for cache key generation."""
        with patch('app.s3_cache.httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.headers = {
                'etag': '"abc123"',
                'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT'
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.head.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            from app.s3_cache import S3Cache
            key = await S3Cache.generate_cache_key_from_url("https://example.com/image.jpg", 1.0)
            
            assert key is not None
            assert key.startswith("processed/")
            assert key.endswith(".jpg")

    @pytest.mark.asyncio
    async def test_uses_last_modified_when_no_etag(self):
        """Test that Last-Modified is used when ETag unavailable."""
        with patch('app.s3_cache.httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.headers = {
                'etag': '',  # Empty ETag
                'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT'
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.head.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            from app.s3_cache import S3Cache
            key = await S3Cache.generate_cache_key_from_url("https://example.com/image.jpg", 1.0)
            
            assert key is not None

    @pytest.mark.asyncio
    async def test_falls_back_to_url_hash_on_error(self):
        """Test that URL hash is used when HEAD request fails."""
        with patch('app.s3_cache.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.head.side_effect = Exception("Connection error")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            from app.s3_cache import S3Cache
            key = await S3Cache.generate_cache_key_from_url("https://example.com/image.jpg", 1.0)
            
            # Should still return a key (fallback to URL hash)
            assert key is not None
            assert key.startswith("processed/")

    @pytest.mark.asyncio
    async def test_different_scale_generates_different_key(self):
        """Test that different hat_scale generates different keys."""
        with patch('app.s3_cache.httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.headers = {'etag': '"abc123"'}
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.head.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            from app.s3_cache import S3Cache
            key1 = await S3Cache.generate_cache_key_from_url("https://example.com/image.jpg", 1.0)
            key2 = await S3Cache.generate_cache_key_from_url("https://example.com/image.jpg", 2.0)
            
            assert key1 != key2


class TestGetCachedImage:
    """Tests for get_cached_image async method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, monkeypatch):
        """Test that get_cached_image returns None when cache is disabled."""
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        
        from app.s3_cache import S3Cache
        cache = S3Cache()
        
        result = await cache.get_cached_image("some/key.jpg")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_data_on_hit(self, monkeypatch):
        """Test that cached data is returned on cache hit."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_body = MagicMock()
            mock_body.read.return_value = b"cached_image_data"
            
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.get_object.return_value = {'Body': mock_body}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            result = await cache.get_cached_image("processed/ab/abc123.jpg")
            
            assert result == b"cached_image_data"
            mock_client.get_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="processed/ab/abc123.jpg"
            )

    @pytest.mark.asyncio
    async def test_returns_none_on_cache_miss(self, monkeypatch):
        """Test that None is returned on cache miss (NoSuchKey)."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.get_object.side_effect = ClientError(
                {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}},
                'GetObject'
            )
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            result = await cache.get_cached_image("processed/ab/nonexistent.jpg")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_other_error(self, monkeypatch):
        """Test that None is returned on other S3 errors."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.get_object.side_effect = ClientError(
                {'Error': {'Code': 'AccessDenied', 'Message': 'Forbidden'}},
                'GetObject'
            )
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            result = await cache.get_cached_image("processed/ab/forbidden.jpg")
            
            assert result is None


class TestStoreCachedImage:
    """Tests for store_cached_image async method."""

    @pytest.mark.asyncio
    async def test_returns_false_when_disabled(self, monkeypatch):
        """Test that store returns False when cache is disabled."""
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        
        from app.s3_cache import S3Cache
        cache = S3Cache()
        
        result = await cache.store_cached_image("key", b"data")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_stores_image_successfully(self, monkeypatch):
        """Test that image is stored successfully."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.put_object.return_value = {}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            result = await cache.store_cached_image(
                "processed/ab/abc123.jpg",
                b"image_data",
                metadata={"faces_detected": 2}
            )
            
            assert result is True
            mock_client.put_object.assert_called_once()
            call_kwargs = mock_client.put_object.call_args[1]
            assert call_kwargs['Bucket'] == "test-bucket"
            assert call_kwargs['Key'] == "processed/ab/abc123.jpg"
            assert call_kwargs['Body'] == b"image_data"
            assert call_kwargs['ContentType'] == 'image/jpeg'

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, monkeypatch):
        """Test that False is returned on storage error."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.put_object.side_effect = Exception("Upload failed")
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            result = await cache.store_cached_image("key", b"data")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_sets_cache_control_header(self, monkeypatch):
        """Test that Cache-Control header is set for CDN caching."""
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        
        with patch('app.s3_cache.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_client.head_bucket.return_value = {}
            mock_client.put_object.return_value = {}
            mock_boto.client.return_value = mock_client
            
            from app.s3_cache import S3Cache
            cache = S3Cache()
            
            await cache.store_cached_image("key", b"data")
            
            call_kwargs = mock_client.put_object.call_args[1]
            assert 'CacheControl' in call_kwargs
            assert 'max-age=31536000' in call_kwargs['CacheControl']
