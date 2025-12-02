"""Configuration and safety limits for the Santa Hat API."""
import os


class Config:
    """Application configuration and safety limits."""

    # File upload limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))  # 10 MB default
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    # Image processing limits
    MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "4000"))  # 4000px default
    MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "4000"))  # 4000px default
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "16000000"))  # 16 megapixels default

    # Face detection limits
    MAX_FACES = int(os.getenv("MAX_FACES", "10"))  # Maximum faces to process per image

    # URL fetching limits
    URL_FETCH_TIMEOUT_SECONDS = int(os.getenv("URL_FETCH_TIMEOUT_SECONDS", "30"))
    MAX_URL_LENGTH = int(os.getenv("MAX_URL_LENGTH", "2048"))

    # Allowed image formats (MIME types)
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp"
    }

    # Allowed image formats (Pillow format names)
    ALLOWED_PIL_FORMATS = {
        "JPEG",
        "PNG",
        "WEBP",
        "GIF",
        "BMP"
    }

    # Blocked URL patterns (to prevent SSRF attacks)
    BLOCKED_URL_PATTERNS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",  # AWS metadata service
        "[::1]",  # IPv6 localhost
        "10.",  # Private network
        "172.16.",  # Private network
        "192.168.",  # Private network
    ]

    @classmethod
    def validate_url_safety(cls, url: str) -> tuple[bool, str]:
        """
        Validate URL for safety (prevent SSRF attacks).

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(url) > cls.MAX_URL_LENGTH:
            return False, f"URL too long (max {cls.MAX_URL_LENGTH} characters)"

        # Convert to lowercase for checking
        url_lower = url.lower()

        # Check for blocked patterns
        for pattern in cls.BLOCKED_URL_PATTERNS:
            if pattern in url_lower:
                return False, f"URLs pointing to private/internal networks are not allowed"

        # Must be HTTP or HTTPS
        if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
            return False, "URL must start with http:// or https://"

        return True, ""

    @classmethod
    def get_limits_info(cls) -> dict:
        """Get current configuration limits as a dictionary."""
        return {
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "max_image_width": cls.MAX_IMAGE_WIDTH,
            "max_image_height": cls.MAX_IMAGE_HEIGHT,
            "max_image_pixels": cls.MAX_IMAGE_PIXELS,
            "max_faces": cls.MAX_FACES,
            "url_fetch_timeout_seconds": cls.URL_FETCH_TIMEOUT_SECONDS,
            "allowed_image_types": list(cls.ALLOWED_IMAGE_TYPES)
        }
