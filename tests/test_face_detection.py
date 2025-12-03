"""Tests for face detection module using real mediapipe."""
import io
import pytest
import httpx
from PIL import Image
from pathlib import Path

from app.face_detection import FaceDetector


# Cache directory for downloaded test images
CACHE_DIR = Path(__file__).parent / ".cache"
CHURCHILL_CACHE_PATH = CACHE_DIR / "churchill.jpg"

# Churchill image URL for single face testing
CHURCHILL_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "b/bc/Sir_Winston_Churchill_-_19086236948.jpg/"
    "440px-Sir_Winston_Churchill_-_19086236948.jpg"
)

HEADERS = {
    "User-Agent": "SantaHatAPI/1.0 (https://github.com/example/santa-hat-api; contact@example.com) httpx/0.25"
}


@pytest.fixture(scope="module")
def face_detector():
    """Create a real FaceDetector instance."""
    return FaceDetector()


@pytest.fixture(scope="module")
def churchill_image():
    """Load Churchill image for testing."""
    if CHURCHILL_CACHE_PATH.exists():
        return Image.open(CHURCHILL_CACHE_PATH)
    
    response = httpx.get(CHURCHILL_IMAGE_URL, headers=HEADERS, timeout=30.0)
    response.raise_for_status()
    
    CACHE_DIR.mkdir(exist_ok=True)
    image = Image.open(io.BytesIO(response.content))
    image.save(CHURCHILL_CACHE_PATH, format='JPEG', quality=95)
    
    return image


@pytest.fixture
def blank_image():
    """Create a blank image with no faces."""
    return Image.new('RGB', (640, 480), color=(128, 128, 128))


class TestFaceDetector:
    """Tests for the FaceDetector class."""

    def test_init_creates_face_mesh(self):
        """Test that FaceDetector initializes successfully."""
        detector = FaceDetector()
        assert detector.face_mesh is not None
        assert detector.mp_face_mesh is not None

    def test_detect_faces_returns_empty_list_when_no_faces(self, face_detector, blank_image):
        """Test that detect_faces returns empty list when no faces found."""
        faces = face_detector.detect_faces(blank_image)
        assert faces == []

    def test_detect_faces_returns_face_data(self, face_detector, churchill_image):
        """Test that detect_faces returns proper face data structure."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        face = faces[0]
        
        # Check all required fields exist
        assert 'forehead_top' in face
        assert 'eye_midpoint' in face
        assert 'eye_distance' in face
        assert 'forehead_width' in face
        assert 'angle' in face
        assert 'head_width' in face
        assert 'all_landmarks' in face
        
        # Check coordinate structure
        assert 'x' in face['forehead_top']
        assert 'y' in face['forehead_top']
        assert 'x' in face['eye_midpoint']
        assert 'y' in face['eye_midpoint']

    def test_detect_faces_calculates_eye_distance(self, face_detector, churchill_image):
        """Test that eye distance is calculated and is reasonable."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        eye_distance = faces[0]['eye_distance']
        
        # Eye distance should be positive and reasonable for a face
        assert eye_distance > 20, "Eye distance too small"
        assert eye_distance < 300, "Eye distance too large"

    def test_detect_faces_calculates_angle(self, face_detector, churchill_image):
        """Test that head tilt angle is calculated."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        angle = faces[0]['angle']
        
        # Churchill is facing forward, angle should be close to 0
        assert -15 < angle < 15, f"Unexpected head angle: {angle}"

    def test_detect_faces_calculates_forehead_width(self, face_detector, churchill_image):
        """Test that forehead width is calculated and is reasonable."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        forehead_width = faces[0]['forehead_width']
        
        # Forehead width should be positive and reasonable
        assert forehead_width > 20, "Forehead width too small"
        assert forehead_width < 400, "Forehead width too large"

    def test_detect_faces_calculates_eye_midpoint(self, face_detector, churchill_image):
        """Test that eye midpoint is calculated correctly."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        eye_midpoint = faces[0]['eye_midpoint']
        
        # Eye midpoint should be within image bounds
        img_width, img_height = churchill_image.size
        assert 0 < eye_midpoint['x'] < img_width
        assert 0 < eye_midpoint['y'] < img_height

    def test_all_landmarks_are_returned(self, face_detector, churchill_image):
        """Test that all 468 landmarks are returned."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        all_landmarks = faces[0]['all_landmarks']
        
        # MediaPipe Face Mesh returns 468 landmarks
        assert len(all_landmarks) == 468
        
        # Each landmark should be a tuple of (x, y) coordinates
        for landmark in all_landmarks:
            assert len(landmark) == 2
            assert isinstance(landmark[0], (int, float))
            assert isinstance(landmark[1], (int, float))

    def test_forehead_top_position(self, face_detector, churchill_image):
        """Test that forehead top position is above eye midpoint."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        forehead_top = faces[0]['forehead_top']
        eye_midpoint = faces[0]['eye_midpoint']
        
        # Forehead should be above eyes (smaller y value)
        assert forehead_top['y'] < eye_midpoint['y'], \
            "Forehead should be above eyes"

    def test_head_width_is_proportional_to_eye_distance(self, face_detector, churchill_image):
        """Test that head width is calculated as 2x eye distance."""
        faces = face_detector.detect_faces(churchill_image)
        
        assert len(faces) == 1
        eye_distance = faces[0]['eye_distance']
        head_width = faces[0]['head_width']
        
        # head_width should be 2 * eye_distance
        assert head_width == pytest.approx(eye_distance * 2.0, rel=0.01)

    def test_detector_is_reusable(self, face_detector, churchill_image):
        """Test that the same detector can process multiple images."""
        # First detection
        faces1 = face_detector.detect_faces(churchill_image)
        assert len(faces1) == 1
        
        # Second detection with same image
        faces2 = face_detector.detect_faces(churchill_image)
        assert len(faces2) == 1
        
        # Results should be consistent
        assert faces1[0]['eye_distance'] == pytest.approx(
            faces2[0]['eye_distance'], rel=0.01
        )

    def test_handles_rgb_image(self, face_detector, churchill_image):
        """Test that RGB images are handled correctly."""
        rgb_image = churchill_image.convert('RGB')
        faces = face_detector.detect_faces(rgb_image)
        assert len(faces) == 1

    def test_handles_rgba_image(self, face_detector, churchill_image):
        """Test that RGBA images are handled correctly."""
        rgba_image = churchill_image.convert('RGBA')
        faces = face_detector.detect_faces(rgba_image)
        assert len(faces) == 1

    def test_handles_grayscale_image(self, face_detector, churchill_image):
        """Test that grayscale images are converted and processed."""
        # Convert to grayscale then back to RGB (as the detector expects RGB)
        gray_image = churchill_image.convert('L').convert('RGB')
        faces = face_detector.detect_faces(gray_image)
        # May or may not detect face in grayscale, but shouldn't crash
        assert isinstance(faces, list)
