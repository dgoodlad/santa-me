"""Tests for configuration module."""
import os
import pytest
from unittest.mock import patch


class TestConfigLimits:
    """Tests for configuration limit values."""

    def test_default_max_file_size(self):
        """Test default max file size is 10MB."""
        # Clear any env override
        with patch.dict(os.environ, {}, clear=True):
            # Need to reload to get defaults
            import importlib
            from app import config
            importlib.reload(config)
            
            assert config.Config.MAX_FILE_SIZE_MB == 10
            assert config.Config.MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024

    def test_max_file_size_from_env(self):
        """Test max file size can be set from environment."""
        with patch.dict(os.environ, {"MAX_FILE_SIZE_MB": "20"}):
            import importlib
            from app import config
            importlib.reload(config)
            
            assert config.Config.MAX_FILE_SIZE_MB == 20
            assert config.Config.MAX_FILE_SIZE_BYTES == 20 * 1024 * 1024

    def test_default_image_dimensions(self):
        """Test default image dimension limits."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            from app import config
            importlib.reload(config)
            
            assert config.Config.MAX_IMAGE_WIDTH == 4000
            assert config.Config.MAX_IMAGE_HEIGHT == 4000
            assert config.Config.MAX_IMAGE_PIXELS == 16000000

    def test_default_max_faces(self):
        """Test default max faces limit."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            from app import config
            importlib.reload(config)
            
            assert config.Config.MAX_FACES == 10

    def test_default_url_timeout(self):
        """Test default URL fetch timeout."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            from app import config
            importlib.reload(config)
            
            assert config.Config.URL_FETCH_TIMEOUT_SECONDS == 30


class TestAllowedImageTypes:
    """Tests for allowed image type configuration."""

    def test_allowed_mime_types(self):
        """Test that common image MIME types are allowed."""
        from app.config import Config
        
        assert "image/jpeg" in Config.ALLOWED_IMAGE_TYPES
        assert "image/png" in Config.ALLOWED_IMAGE_TYPES
        assert "image/webp" in Config.ALLOWED_IMAGE_TYPES
        assert "image/gif" in Config.ALLOWED_IMAGE_TYPES
        assert "image/bmp" in Config.ALLOWED_IMAGE_TYPES

    def test_allowed_pil_formats(self):
        """Test that PIL format names are allowed."""
        from app.config import Config
        
        assert "JPEG" in Config.ALLOWED_PIL_FORMATS
        assert "PNG" in Config.ALLOWED_PIL_FORMATS
        assert "WEBP" in Config.ALLOWED_PIL_FORMATS
        assert "GIF" in Config.ALLOWED_PIL_FORMATS
        assert "BMP" in Config.ALLOWED_PIL_FORMATS


class TestBlockedUrlPatterns:
    """Tests for blocked URL patterns (SSRF protection)."""

    def test_localhost_blocked(self):
        """Test that localhost URLs are blocked."""
        from app.config import Config
        
        assert "localhost" in Config.BLOCKED_URL_PATTERNS

    def test_loopback_blocked(self):
        """Test that loopback IPs are blocked."""
        from app.config import Config
        
        assert "127.0.0.1" in Config.BLOCKED_URL_PATTERNS
        assert "0.0.0.0" in Config.BLOCKED_URL_PATTERNS
        assert "[::1]" in Config.BLOCKED_URL_PATTERNS

    def test_aws_metadata_blocked(self):
        """Test that AWS metadata service IP is blocked."""
        from app.config import Config
        
        assert "169.254.169.254" in Config.BLOCKED_URL_PATTERNS

    def test_private_networks_blocked(self):
        """Test that private network ranges are blocked."""
        from app.config import Config
        
        assert "10." in Config.BLOCKED_URL_PATTERNS
        assert "172.16." in Config.BLOCKED_URL_PATTERNS
        assert "192.168." in Config.BLOCKED_URL_PATTERNS


class TestValidateUrlSafety:
    """Tests for URL safety validation."""

    def test_valid_https_url(self):
        """Test that valid HTTPS URLs pass validation."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("https://example.com/image.jpg")
        
        assert is_valid is True
        assert error == ""

    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://example.com/image.jpg")
        
        assert is_valid is True
        assert error == ""

    def test_localhost_blocked(self):
        """Test that localhost URLs are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://localhost:8000/image.jpg")
        
        assert is_valid is False
        assert "private" in error.lower() or "internal" in error.lower()

    def test_loopback_ip_blocked(self):
        """Test that loopback IPs are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://127.0.0.1/image.jpg")
        
        assert is_valid is False

    def test_aws_metadata_blocked(self):
        """Test that AWS metadata service URL is blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety(
            "http://169.254.169.254/latest/meta-data/"
        )
        
        assert is_valid is False

    def test_private_network_10_blocked(self):
        """Test that 10.x.x.x URLs are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://10.0.0.1/image.jpg")
        
        assert is_valid is False

    def test_private_network_172_blocked(self):
        """Test that 172.16.x.x URLs are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://172.16.0.1/image.jpg")
        
        assert is_valid is False

    def test_private_network_192_blocked(self):
        """Test that 192.168.x.x URLs are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("http://192.168.1.1/image.jpg")
        
        assert is_valid is False

    def test_url_too_long(self):
        """Test that URLs exceeding max length are rejected."""
        from app.config import Config
        
        long_url = "https://example.com/" + "a" * 3000
        is_valid, error = Config.validate_url_safety(long_url)
        
        assert is_valid is False
        assert "too long" in error.lower()

    def test_non_http_protocol_blocked(self):
        """Test that non-HTTP protocols are blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("ftp://example.com/image.jpg")
        
        assert is_valid is False
        assert "http" in error.lower()

    def test_file_protocol_blocked(self):
        """Test that file:// protocol is blocked."""
        from app.config import Config
        
        is_valid, error = Config.validate_url_safety("file:///etc/passwd")
        
        assert is_valid is False

    def test_case_insensitive_blocking(self):
        """Test that URL validation is case insensitive."""
        from app.config import Config
        
        is_valid, _ = Config.validate_url_safety("http://LOCALHOST/image.jpg")
        assert is_valid is False
        
        is_valid, _ = Config.validate_url_safety("http://LocalHost/image.jpg")
        assert is_valid is False


class TestGetLimitsInfo:
    """Tests for get_limits_info method."""

    def test_returns_all_limits(self):
        """Test that all limit info is returned."""
        from app.config import Config
        
        limits = Config.get_limits_info()
        
        assert "max_file_size_mb" in limits
        assert "max_image_width" in limits
        assert "max_image_height" in limits
        assert "max_image_pixels" in limits
        assert "max_faces" in limits
        assert "url_fetch_timeout_seconds" in limits
        assert "allowed_image_types" in limits

    def test_allowed_image_types_is_list(self):
        """Test that allowed_image_types is returned as a list."""
        from app.config import Config
        
        limits = Config.get_limits_info()
        
        assert isinstance(limits["allowed_image_types"], list)
        assert len(limits["allowed_image_types"]) > 0
