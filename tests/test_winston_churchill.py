"""Integration test for applying Santa hat to Winston Churchill image."""
import io
import os
import pytest
import httpx
from PIL import Image
from pathlib import Path


# The actual image URL from Wikimedia Commons
# Source: https://en.wikipedia.org/wiki/File:Sir_Winston_Churchill_-_19086236948.jpg
CHURCHILL_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "b/bc/Sir_Winston_Churchill_-_19086236948.jpg/"
    "440px-Sir_Winston_Churchill_-_19086236948.jpg"
)

# Wikimedia requires a proper User-Agent header
HEADERS = {
    "User-Agent": "SantaHatAPI/1.0 (https://github.com/example/santa-hat-api; contact@example.com) httpx/0.25"
}

# Cache directory for downloaded test images
CACHE_DIR = Path(__file__).parent / ".cache"
CHURCHILL_CACHE_PATH = CACHE_DIR / "churchill.jpg"


@pytest.fixture(scope="session")
def churchill_image():
    """Load Winston Churchill image, downloading and caching if needed."""
    # Check if image is already cached
    if CHURCHILL_CACHE_PATH.exists():
        return Image.open(CHURCHILL_CACHE_PATH)
    
    # Download and cache the image
    response = httpx.get(CHURCHILL_IMAGE_URL, headers=HEADERS, timeout=30.0)
    response.raise_for_status()
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Save to cache
    image = Image.open(io.BytesIO(response.content))
    image.save(CHURCHILL_CACHE_PATH, format='JPEG', quality=95)
    
    return image


@pytest.fixture
def santa_hat_path():
    """Get path to the Santa hat image."""
    base_dir = Path(__file__).parent.parent
    return str(base_dir / "static" / "santa_hat.png")


class TestWinstonChurchillSantaHat:
    """Integration tests for applying Santa hat to Winston Churchill image."""

    def test_download_churchill_image(self, churchill_image):
        """Test that we can download the Churchill image."""
        assert churchill_image is not None
        assert churchill_image.size[0] > 0
        assert churchill_image.size[1] > 0
        # Churchill image should be a portrait
        assert churchill_image.mode in ('RGB', 'RGBA', 'L')

    def test_detect_face_in_churchill_image(self, churchill_image):
        """Test face detection on Churchill image."""
        from app.face_detection import FaceDetector
        
        # Convert to RGB if needed
        if churchill_image.mode != 'RGB':
            churchill_image = churchill_image.convert('RGB')
        
        detector = FaceDetector()
        faces = detector.detect_faces(churchill_image)
        
        # Should detect exactly one face
        assert len(faces) == 1, f"Expected 1 face, detected {len(faces)}"
        
        # Verify face data structure
        face = faces[0]
        assert 'forehead_top' in face
        assert 'eye_midpoint' in face
        assert 'eye_distance' in face
        assert 'forehead_width' in face
        assert 'angle' in face
        
        # Eye distance should be reasonable for a face
        assert face['eye_distance'] > 20, "Eye distance too small"
        assert face['eye_distance'] < 300, "Eye distance too large"
        
        # Angle should be relatively straight (Churchill is facing forward)
        assert -15 < face['angle'] < 15, f"Unexpected head angle: {face['angle']}"

    def test_apply_santa_hat_to_churchill(self, churchill_image, santa_hat_path):
        """Test applying Santa hat to Churchill image."""
        # Skip if santa hat image doesn't exist
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        from app.image_processing import SantaHatProcessor
        
        # Convert to RGB if needed
        if churchill_image.mode != 'RGB':
            churchill_image = churchill_image.convert('RGB')
        
        # Detect faces
        detector = FaceDetector()
        faces = detector.detect_faces(churchill_image)
        
        assert len(faces) > 0, "No faces detected"
        
        # Apply Santa hat
        processor = SantaHatProcessor(hat_image_path=santa_hat_path)
        result = processor.process_image(churchill_image, faces)
        
        # Verify result
        assert result is not None
        assert result.size == churchill_image.size
        assert result.mode == 'RGBA'
        
        # The result should be different from the original
        # (hat was added, so pixel data should differ)
        original_rgba = churchill_image.convert('RGBA')
        assert list(result.getdata()) != list(original_rgba.getdata()), \
            "Result image is identical to original - hat may not have been applied"

    def test_apply_santa_hat_with_scale(self, churchill_image, santa_hat_path):
        """Test applying Santa hat with different scale values."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        from app.image_processing import SantaHatProcessor
        
        if churchill_image.mode != 'RGB':
            churchill_image = churchill_image.convert('RGB')
        
        detector = FaceDetector()
        faces = detector.detect_faces(churchill_image)
        
        assert len(faces) > 0, "No faces detected"
        
        processor = SantaHatProcessor(hat_image_path=santa_hat_path)
        
        # Test different scales
        for scale in [0.5, 1.0, 1.5, 2.0]:
            result = processor.process_image(churchill_image.copy(), faces, hat_scale=scale)
            assert result is not None
            assert result.size == churchill_image.size
            assert result.mode == 'RGBA'

    def test_full_pipeline_churchill(self, churchill_image, santa_hat_path, tmp_path):
        """Test full pipeline: download, detect, apply hat, save."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        from app.image_processing import SantaHatProcessor
        
        if churchill_image.mode != 'RGB':
            churchill_image = churchill_image.convert('RGB')
        
        # Detect faces
        detector = FaceDetector()
        faces = detector.detect_faces(churchill_image)
        
        assert len(faces) == 1, f"Expected 1 face, got {len(faces)}"
        
        # Apply Santa hat
        processor = SantaHatProcessor(hat_image_path=santa_hat_path)
        result = processor.process_image(churchill_image, faces)
        
        # Save result to temp file
        output_path = tmp_path / "churchill_santa.jpg"
        result_rgb = result.convert('RGB')
        result_rgb.save(output_path, format='JPEG', quality=95)
        
        # Verify file was created and is valid
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Re-open and verify it's a valid image
        reopened = Image.open(output_path)
        assert reopened.size == churchill_image.size

    def test_churchill_via_api(self, churchill_image, santa_hat_path):
        """Test processing Churchill image via the API endpoint."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from fastapi.testclient import TestClient
        from app.main import app
        
        # Convert image to bytes
        buffer = io.BytesIO()
        if churchill_image.mode != 'RGB':
            churchill_image = churchill_image.convert('RGB')
        churchill_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        
        # Test via API
        with TestClient(app) as client:
            response = client.post(
                "/santa-hatify",
                files={"file": ("churchill.jpg", image_bytes, "image/jpeg")}
            )
            
            assert response.status_code == 200, f"API error: {response.text}"
            assert response.headers["content-type"] == "image/jpeg"
            assert "X-Faces-Detected" in response.headers
            assert response.headers["X-Faces-Detected"] == "1"
            
            # Verify response is a valid image
            result_image = Image.open(io.BytesIO(response.content))
            assert result_image.size == churchill_image.size
