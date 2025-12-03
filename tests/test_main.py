"""Tests for FastAPI main application endpoints."""
import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image
from fastapi.testclient import TestClient

# Import the module so patch targets exist
import app.main


@pytest.fixture
def mock_app_dependencies(sample_rgba_image, sample_face_data):
    """Setup mocked dependencies for API tests."""
    # Create mock face detector
    mock_detector = MagicMock()
    mock_detector.detect_faces.return_value = [sample_face_data]
    
    # Create mock hat processor
    mock_processor = MagicMock()
    mock_processor.process_image.return_value = sample_rgba_image
    
    # Create mock S3 cache (disabled)
    mock_cache = MagicMock()
    mock_cache.enabled = False
    mock_cache.generate_cache_key_from_url = AsyncMock(return_value=None)
    mock_cache.generate_cache_key_from_hash = MagicMock(return_value=None)
    mock_cache.get_cached_image = AsyncMock(return_value=None)
    mock_cache.store_cached_image = AsyncMock(return_value=False)
    
    return {
        'detector': mock_detector,
        'processor': mock_processor,
        'cache': mock_cache
    }


@pytest.fixture
def test_client(mock_app_dependencies):
    """Create test client with mocked dependencies."""
    with patch.object(app.main, 'face_detector', mock_app_dependencies['detector']), \
         patch.object(app.main, 'hat_processor', mock_app_dependencies['processor']), \
         patch.object(app.main, 's3_cache', mock_app_dependencies['cache']):
        
        with TestClient(app.main.app) as client:
            yield client


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_welcome_message(self, test_client):
        """Test that root endpoint returns welcome and endpoints info."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"]
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_status(self, test_client):
        """Test that health endpoint returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "face_detector" in data
        assert "hat_processor" in data
        assert "s3_cache" in data
        assert "limits" in data


class TestSantaHatifyPostEndpoint:
    """Tests for POST /santa-hatify endpoint."""

    def test_upload_image_success(
        self, 
        test_client, 
        sample_image_bytes, 
        mock_app_dependencies
    ):
        """Test successful image upload and processing."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert "X-Faces-Detected" in response.headers

    def test_upload_returns_face_count_header(
        self,
        test_client,
        sample_image_bytes,
        mock_app_dependencies
    ):
        """Test that response includes face count header."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.headers["X-Faces-Detected"] == "1"

    def test_upload_png_image(
        self,
        test_client,
        sample_png_bytes,
        mock_app_dependencies
    ):
        """Test uploading PNG image."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.png", sample_png_bytes, "image/png")}
        )
        
        assert response.status_code == 200

    def test_rejects_unsupported_mime_type(self, test_client):
        """Test that unsupported MIME types are rejected."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Unsupported image type" in response.json()["detail"]

    def test_rejects_oversized_file(self, test_client):
        """Test that files exceeding size limit are rejected."""
        # Create a file larger than 10MB
        large_data = b"x" * (11 * 1024 * 1024)
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("large.jpg", large_data, "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_returns_404_when_no_faces(
        self, 
        test_client, 
        sample_image_bytes,
        mock_app_dependencies
    ):
        """Test that 404 is returned when no faces detected."""
        mock_app_dependencies['detector'].detect_faces.return_value = []
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 404
        assert "No faces detected" in response.json()["detail"]

    def test_hat_scale_parameter(
        self,
        test_client,
        sample_image_bytes,
        mock_app_dependencies
    ):
        """Test that hat_scale parameter is passed to processor."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"hat_scale": "1.5"}
        )
        
        assert response.status_code == 200
        # Verify processor was called with hat_scale
        mock_app_dependencies['processor'].process_image.assert_called()
        call_args = mock_app_dependencies['processor'].process_image.call_args
        assert call_args[0][2] == 1.5  # hat_scale argument

    def test_rejects_invalid_hat_scale(self, test_client, sample_image_bytes):
        """Test that invalid hat_scale values are rejected."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"hat_scale": "10"}  # Exceeds max of 5
        )
        
        assert response.status_code == 400
        assert "hat_scale" in response.json()["detail"]

    def test_rejects_negative_hat_scale(self, test_client, sample_image_bytes):
        """Test that negative hat_scale is rejected."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"hat_scale": "-1"}
        )
        
        assert response.status_code == 400

    def test_requires_file_or_url(self, test_client):
        """Test that either file or URL must be provided."""
        response = test_client.post("/santa-hatify")
        
        assert response.status_code == 400
        assert "file" in response.json()["detail"].lower() or "url" in response.json()["detail"].lower()

    def test_rejects_both_file_and_url(self, test_client, sample_image_bytes):
        """Test that providing both file and URL is rejected."""
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"url": "https://example.com/image.jpg"}
        )
        
        assert response.status_code == 400
        assert "not both" in response.json()["detail"].lower()


class TestSantaHatifyUrlProcessing:
    """Tests for URL-based image processing."""

    def test_url_via_form_data(
        self, 
        test_client, 
        sample_image_bytes, 
        mock_app_dependencies
    ):
        """Test processing image from URL via form data."""
        with patch('app.main.httpx.AsyncClient') as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = sample_image_bytes
            mock_response.headers = {
                'content-type': 'image/jpeg',
                'content-length': str(len(sample_image_bytes))
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.return_value = mock_client
            
            response = test_client.post(
                "/santa-hatify",
                data={"url": "https://example.com/image.jpg"}
            )
            
            assert response.status_code == 200

    def test_rejects_localhost_url(self, test_client):
        """Test that localhost URLs are rejected (SSRF protection)."""
        response = test_client.post(
            "/santa-hatify",
            data={"url": "http://localhost:8000/image.jpg"}
        )
        
        assert response.status_code == 400
        assert "private" in response.json()["detail"].lower() or "internal" in response.json()["detail"].lower()

    def test_rejects_private_ip_url(self, test_client):
        """Test that private IP URLs are rejected (SSRF protection)."""
        response = test_client.post(
            "/santa-hatify",
            data={"url": "http://192.168.1.1/image.jpg"}
        )
        
        assert response.status_code == 400

    def test_rejects_aws_metadata_url(self, test_client):
        """Test that AWS metadata service URL is rejected (SSRF protection)."""
        response = test_client.post(
            "/santa-hatify",
            data={"url": "http://169.254.169.254/latest/meta-data/"}
        )
        
        assert response.status_code == 400


class TestSantaHatifyGetEndpoint:
    """Tests for GET /santa-hatify endpoint (Slack-friendly)."""

    def test_get_with_url_param(
        self, 
        test_client, 
        sample_image_bytes, 
        mock_app_dependencies
    ):
        """Test GET endpoint with URL parameter."""
        with patch('app.main.httpx.AsyncClient') as mock_httpx:
            mock_response = MagicMock()
            mock_response.content = sample_image_bytes
            mock_response.headers = {
                'content-type': 'image/jpeg',
                'content-length': str(len(sample_image_bytes))
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_httpx.return_value = mock_client
            
            response = test_client.get(
                "/santa-hatify",
                params={"url": "https://example.com/image.jpg"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"

    def test_get_rejects_invalid_hat_scale(self, test_client):
        """Test GET endpoint rejects invalid hat_scale."""
        response = test_client.get(
            "/santa-hatify",
            params={"url": "https://example.com/image.jpg", "hat_scale": 10}
        )
        
        assert response.status_code == 400
        assert "hat_scale" in response.json()["detail"]

    def test_get_rejects_private_url(self, test_client):
        """Test GET endpoint rejects private URLs."""
        response = test_client.get(
            "/santa-hatify",
            params={"url": "http://localhost/image.jpg"}
        )
        
        assert response.status_code == 400


class TestCachingBehavior:
    """Tests for S3 caching behavior."""

    def test_cache_hit_returns_cached_response(
        self, 
        sample_image_bytes,
        sample_face_data,
        sample_rgba_image
    ):
        """Test that cache hit returns cached image without reprocessing."""
        # Create mock dependencies with cache enabled
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [sample_face_data]
        
        mock_processor = MagicMock()
        mock_processor.process_image.return_value = sample_rgba_image
        
        mock_cache = MagicMock()
        mock_cache.enabled = True
        mock_cache.generate_cache_key_from_hash = MagicMock(return_value="cached/key.jpg")
        mock_cache.get_cached_image = AsyncMock(return_value=sample_image_bytes)
        
        with patch.object(app.main, 'face_detector', mock_detector), \
             patch.object(app.main, 'hat_processor', mock_processor), \
             patch.object(app.main, 's3_cache', mock_cache):
            
            with TestClient(app.main.app) as client:
                response = client.post(
                    "/santa-hatify",
                    files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
                )
                
                assert response.status_code == 200
                assert response.headers.get("X-Cache-Status") == "HIT"
                # Processor should NOT be called on cache hit
                mock_processor.process_image.assert_not_called()

    def test_cache_miss_processes_and_stores(
        self,
        sample_image_bytes,
        sample_face_data,
        sample_rgba_image
    ):
        """Test that cache miss processes image and stores result."""
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = [sample_face_data]
        
        mock_processor = MagicMock()
        mock_processor.process_image.return_value = sample_rgba_image
        
        mock_cache = MagicMock()
        mock_cache.enabled = True
        mock_cache.generate_cache_key_from_hash = MagicMock(return_value="cached/key.jpg")
        mock_cache.get_cached_image = AsyncMock(return_value=None)  # Cache miss
        mock_cache.store_cached_image = AsyncMock(return_value=True)
        
        with patch.object(app.main, 'face_detector', mock_detector), \
             patch.object(app.main, 'hat_processor', mock_processor), \
             patch.object(app.main, 's3_cache', mock_cache):
            
            with TestClient(app.main.app) as client:
                response = client.post(
                    "/santa-hatify",
                    files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
                )
                
                assert response.status_code == 200
                assert response.headers.get("X-Cache-Status") == "MISS"
                # Processor SHOULD be called on cache miss
                mock_processor.process_image.assert_called_once()
                # Result should be stored in cache
                mock_cache.store_cached_image.assert_called_once()


class TestHatProcessorNotConfigured:
    """Tests for when hat processor is not configured."""

    def test_returns_503_when_processor_none(self, sample_face_data):
        """Test that 503 is returned when hat processor is None."""
        mock_detector = MagicMock()
        mock_cache = MagicMock()
        mock_cache.enabled = False
        
        with patch.object(app.main, 'face_detector', mock_detector), \
             patch.object(app.main, 'hat_processor', None), \
             patch.object(app.main, 's3_cache', mock_cache):
            
            with TestClient(app.main.app) as client:
                # Create a simple test image
                img = Image.new('RGB', (100, 100), color='red')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                
                response = client.post(
                    "/santa-hatify",
                    files={"file": ("test.jpg", buffer.getvalue(), "image/jpeg")}
                )
                
                assert response.status_code == 503
                assert "not configured" in response.json()["detail"]


class TestImageValidation:
    """Tests for image validation."""

    def test_rejects_oversized_dimensions(
        self,
        test_client,
        mock_app_dependencies
    ):
        """Test that images exceeding dimension limits are rejected."""
        # Create image that exceeds limits (need to pass initial file checks)
        # The actual dimension check happens after PIL opens the image
        img = Image.new('RGB', (5000, 5000), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=50)
        buffer.seek(0)
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("large.jpg", buffer.getvalue(), "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "dimensions" in response.json()["detail"].lower()

    def test_converts_grayscale_to_rgb(
        self,
        test_client,
        sample_grayscale_image,
        mock_app_dependencies
    ):
        """Test that grayscale images are converted to RGB."""
        buffer = io.BytesIO()
        sample_grayscale_image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("gray.jpg", buffer.getvalue(), "image/jpeg")}
        )
        
        # Should succeed - grayscale is converted internally
        assert response.status_code == 200


class TestMultipleFaces:
    """Tests for handling multiple faces."""

    def test_processes_multiple_faces(
        self,
        test_client,
        sample_image_bytes,
        sample_multiple_faces,
        sample_rgba_image,
        mock_app_dependencies
    ):
        """Test that multiple faces are all processed."""
        mock_app_dependencies['detector'].detect_faces.return_value = sample_multiple_faces
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        assert response.headers["X-Faces-Detected"] == "2"

    def test_limits_faces_to_max(
        self,
        test_client,
        sample_image_bytes,
        sample_face_data,
        mock_app_dependencies
    ):
        """Test that face count is limited to MAX_FACES."""
        # Create 15 faces (more than default MAX_FACES of 10)
        many_faces = [sample_face_data.copy() for _ in range(15)]
        mock_app_dependencies['detector'].detect_faces.return_value = many_faces
        
        response = test_client.post(
            "/santa-hatify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        # Should only process MAX_FACES (10 by default)
        faces_detected = int(response.headers["X-Faces-Detected"])
        assert faces_detected <= 10
