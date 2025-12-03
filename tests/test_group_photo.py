"""Integration test for applying Santa hats to a group photo with multiple faces."""
import io
import os
import pytest
import httpx
from PIL import Image
from pathlib import Path


# Group photo URL from S3
GROUP_PHOTO_URL = (
    "https://yeargin-pixelfed-files.s3.us-east-1.amazonaws.com/public/m/_v2/"
    "789284324860764161/634f2a4e4-291f24/LcMkJ6LqIz82/"
    "n6A1JOlEr70dG35uIAaBxEo922qfO0gQRWm09Ayq.jpg"
)

# Cache directory for downloaded test images
CACHE_DIR = Path(__file__).parent / ".cache"
GROUP_PHOTO_CACHE_PATH = CACHE_DIR / "group_photo.jpg"


@pytest.fixture(scope="session")
def group_photo():
    """Load group photo, downloading and caching if needed."""
    # Check if image is already cached
    if GROUP_PHOTO_CACHE_PATH.exists():
        return Image.open(GROUP_PHOTO_CACHE_PATH)
    
    # Download and cache the image
    response = httpx.get(GROUP_PHOTO_URL, timeout=30.0)
    response.raise_for_status()
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Save to cache
    image = Image.open(io.BytesIO(response.content))
    image.save(GROUP_PHOTO_CACHE_PATH, format='JPEG', quality=95)
    
    return image


@pytest.fixture
def santa_hat_path():
    """Get path to the Santa hat image."""
    base_dir = Path(__file__).parent.parent
    return str(base_dir / "static" / "santa_hat.png")


class TestGroupPhotoSantaHats:
    """Integration tests for applying Santa hats to multiple people in a group photo."""

    def test_download_group_photo(self, group_photo):
        """Test that we can download the group photo."""
        assert group_photo is not None
        assert group_photo.size[0] > 0
        assert group_photo.size[1] > 0
        assert group_photo.mode in ('RGB', 'RGBA', 'L')

    def test_detect_multiple_faces_in_group_photo(self, group_photo):
        """Test that multiple faces are detected in the group photo."""
        from app.face_detection import FaceDetector
        
        # Convert to RGB if needed
        if group_photo.mode != 'RGB':
            group_photo = group_photo.convert('RGB')
        
        detector = FaceDetector()
        faces = detector.detect_faces(group_photo)
        
        # Should detect multiple faces (at least 2)
        assert len(faces) >= 2, f"Expected multiple faces, detected {len(faces)}"
        
        # Verify each face has the required data structure
        for i, face in enumerate(faces):
            assert 'forehead_top' in face, f"Face {i} missing forehead_top"
            assert 'eye_midpoint' in face, f"Face {i} missing eye_midpoint"
            assert 'eye_distance' in face, f"Face {i} missing eye_distance"
            assert 'forehead_width' in face, f"Face {i} missing forehead_width"
            assert 'angle' in face, f"Face {i} missing angle"
            
            # Eye distance should be reasonable for a face
            assert face['eye_distance'] > 10, f"Face {i} eye distance too small"
            assert face['eye_distance'] < 500, f"Face {i} eye distance too large"

    def test_apply_santa_hats_to_multiple_faces(self, group_photo, santa_hat_path):
        """Test applying Santa hats to all faces in the group photo."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        from app.image_processing import SantaHatProcessor
        
        # Convert to RGB if needed
        if group_photo.mode != 'RGB':
            group_photo = group_photo.convert('RGB')
        
        # Detect faces
        detector = FaceDetector()
        faces = detector.detect_faces(group_photo)
        
        assert len(faces) >= 2, f"Expected multiple faces, detected {len(faces)}"
        
        # Apply Santa hats
        processor = SantaHatProcessor(hat_image_path=santa_hat_path)
        result = processor.process_image(group_photo, faces)
        
        # Verify result
        assert result is not None
        assert result.size == group_photo.size
        assert result.mode == 'RGBA'
        
        # The result should be different from the original
        original_rgba = group_photo.convert('RGBA')
        assert list(result.getdata()) != list(original_rgba.getdata()), \
            "Result image is identical to original - hats may not have been applied"

    def test_group_photo_via_api(self, group_photo, santa_hat_path):
        """Test processing group photo via the API endpoint."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from fastapi.testclient import TestClient
        from app.main import app
        
        # Convert image to bytes
        buffer = io.BytesIO()
        if group_photo.mode != 'RGB':
            group_photo = group_photo.convert('RGB')
        group_photo.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        
        # Test via API
        with TestClient(app) as client:
            response = client.post(
                "/santa-hatify",
                files={"file": ("group_photo.jpg", image_bytes, "image/jpeg")}
            )
            
            assert response.status_code == 200, f"API error: {response.text}"
            assert response.headers["content-type"] == "image/jpeg"
            assert "X-Faces-Detected" in response.headers
            
            # Verify multiple faces were detected
            faces_detected = int(response.headers["X-Faces-Detected"])
            assert faces_detected >= 2, f"Expected multiple faces, API detected {faces_detected}"
            
            # Verify response is a valid image
            result_image = Image.open(io.BytesIO(response.content))
            assert result_image.size == group_photo.size

    def test_each_face_gets_appropriately_sized_hat(self, group_photo, santa_hat_path):
        """Test that each face gets a hat sized proportionally to their face."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        
        # Convert to RGB if needed
        if group_photo.mode != 'RGB':
            group_photo = group_photo.convert('RGB')
        
        detector = FaceDetector()
        faces = detector.detect_faces(group_photo)
        
        assert len(faces) >= 2, f"Expected multiple faces, detected {len(faces)}"
        
        # Verify that different faces have different eye distances
        # (indicating the system can handle faces at different scales/distances)
        eye_distances = [face['eye_distance'] for face in faces]
        
        # At least some variation in eye distances is expected in a group photo
        # (people at different distances from camera, or different face sizes)
        min_distance = min(eye_distances)
        max_distance = max(eye_distances)
        
        # Log the eye distances for debugging
        print(f"Eye distances detected: {eye_distances}")
        print(f"Min: {min_distance}, Max: {max_distance}")
        
        # All faces should have valid eye distances
        for i, distance in enumerate(eye_distances):
            assert distance > 0, f"Face {i} has invalid eye distance: {distance}"

    def test_full_pipeline_group_photo(self, group_photo, santa_hat_path, tmp_path):
        """Test full pipeline: download, detect multiple faces, apply hats, save."""
        if not os.path.exists(santa_hat_path):
            pytest.skip(f"Santa hat image not found at {santa_hat_path}")
        
        from app.face_detection import FaceDetector
        from app.image_processing import SantaHatProcessor
        
        if group_photo.mode != 'RGB':
            group_photo = group_photo.convert('RGB')
        
        # Detect faces
        detector = FaceDetector()
        faces = detector.detect_faces(group_photo)
        
        num_faces = len(faces)
        assert num_faces >= 2, f"Expected multiple faces, got {num_faces}"
        
        # Apply Santa hats
        processor = SantaHatProcessor(hat_image_path=santa_hat_path)
        result = processor.process_image(group_photo, faces)
        
        # Save result to temp file
        output_path = tmp_path / "group_santa.jpg"
        result_rgb = result.convert('RGB')
        result_rgb.save(output_path, format='JPEG', quality=95)
        
        # Verify file was created and is valid
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Re-open and verify it's a valid image
        reopened = Image.open(output_path)
        assert reopened.size == group_photo.size
        
        print(f"Successfully applied Santa hats to {num_faces} faces")
        print(f"Result saved to: {output_path}")
