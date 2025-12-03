"""Shared test fixtures and configuration."""
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from PIL import Image
import numpy as np


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Mock mediapipe if not installed (for local testing without mediapipe)
# ============================================================================

if 'mediapipe' not in sys.modules:
    # Create mock mediapipe module
    mock_mp = MagicMock()
    mock_face_mesh = MagicMock()
    mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
    mock_face_mesh.process.return_value = MagicMock(multi_face_landmarks=None)
    sys.modules['mediapipe'] = mock_mp


# ============================================================================
# Environment Setup (disable AWS/S3 in tests)
# ============================================================================

@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Ensure AWS credentials are not used during tests."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image():
    """Create a simple RGB test image."""
    img = Image.new('RGB', (640, 480), color=(255, 200, 150))
    return img


@pytest.fixture
def sample_rgba_image():
    """Create a simple RGBA test image with transparency."""
    img = Image.new('RGBA', (640, 480), color=(255, 200, 150, 255))
    return img


@pytest.fixture
def sample_grayscale_image():
    """Create a simple grayscale test image."""
    img = Image.new('L', (640, 480), color=128)
    return img


@pytest.fixture
def large_image():
    """Create an image that exceeds dimension limits."""
    img = Image.new('RGB', (5000, 5000), color=(255, 200, 150))
    return img


@pytest.fixture
def sample_image_bytes(sample_rgb_image):
    """Get a sample image as JPEG bytes."""
    buffer = io.BytesIO()
    sample_rgb_image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes(sample_rgba_image):
    """Get a sample image as PNG bytes."""
    buffer = io.BytesIO()
    sample_rgba_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def santa_hat_image():
    """Create a mock Santa hat image (red triangle with transparency)."""
    # Create a simple hat shape (100x80 red triangle)
    img = Image.new('RGBA', (100, 80), color=(0, 0, 0, 0))
    pixels = img.load()
    
    # Draw a simple red triangle
    for y in range(80):
        width_at_y = int((y / 80) * 100)
        start_x = (100 - width_at_y) // 2
        for x in range(start_x, start_x + width_at_y):
            pixels[x, y] = (200, 0, 0, 255)  # Red with full opacity
    
    return img


# ============================================================================
# Face Detection Fixtures
# ============================================================================

@pytest.fixture
def mock_face_data():
    """Sample face detection data for a single face."""
    return {
        'forehead_top': {'x': 320.0, 'y': 100.0},
        'eye_midpoint': {'x': 320.0, 'y': 200.0},
        'eye_distance': 80.0,
        'forehead_width': 100.0,
        'angle': 0.0,
        'head_width': 160.0,
        'all_landmarks': [(320, 100), (280, 200), (360, 200)]  # Simplified
    }


@pytest.fixture
def mock_tilted_face_data():
    """Sample face detection data for a tilted face (15 degrees)."""
    return {
        'forehead_top': {'x': 330.0, 'y': 105.0},
        'eye_midpoint': {'x': 320.0, 'y': 200.0},
        'eye_distance': 80.0,
        'forehead_width': 100.0,
        'angle': 15.0,
        'head_width': 160.0,
        'all_landmarks': [(330, 105), (280, 200), (360, 200)]
    }


@pytest.fixture
def mock_multiple_faces():
    """Sample face detection data for multiple faces."""
    return [
        {
            'forehead_top': {'x': 160.0, 'y': 100.0},
            'eye_midpoint': {'x': 160.0, 'y': 200.0},
            'eye_distance': 60.0,  # Smaller face (child)
            'forehead_width': 75.0,
            'angle': 0.0,
            'head_width': 120.0,
            'all_landmarks': []
        },
        {
            'forehead_top': {'x': 480.0, 'y': 100.0},
            'eye_midpoint': {'x': 480.0, 'y': 200.0},
            'eye_distance': 80.0,  # Larger face (adult)
            'forehead_width': 100.0,
            'angle': -5.0,
            'head_width': 160.0,
            'all_landmarks': []
        }
    ]


@pytest.fixture
def mock_mediapipe_landmark():
    """Create a mock MediaPipe landmark."""
    def create_landmark(x, y, z=0):
        landmark = MagicMock()
        landmark.x = x
        landmark.y = y
        landmark.z = z
        return landmark
    return create_landmark


@pytest.fixture
def mock_mediapipe_face_landmarks(mock_mediapipe_landmark):
    """Create mock MediaPipe face landmarks for a 640x480 image."""
    # Create 468 landmarks (MediaPipe Face Mesh has 468 landmarks)
    landmarks = [mock_mediapipe_landmark(0.5, 0.5) for _ in range(468)]
    
    # Set key landmarks used in face_detection.py
    # Values are normalized (0-1), will be multiplied by image dimensions
    landmarks[10] = mock_mediapipe_landmark(0.5, 0.2)    # forehead_top
    landmarks[109] = mock_mediapipe_landmark(0.35, 0.25)  # forehead_left
    landmarks[338] = mock_mediapipe_landmark(0.65, 0.25)  # forehead_right
    landmarks[151] = mock_mediapipe_landmark(0.5, 0.8)    # chin
    landmarks[33] = mock_mediapipe_landmark(0.375, 0.4)   # left_eye
    landmarks[263] = mock_mediapipe_landmark(0.625, 0.4)  # right_eye
    
    face_landmarks = MagicMock()
    face_landmarks.landmark = landmarks
    return face_landmarks


# ============================================================================
# S3 Cache Fixtures
# ============================================================================

@pytest.fixture
def mock_s3_client():
    """Create a mock boto3 S3 client."""
    client = MagicMock()
    client.head_bucket.return_value = {}
    client.get_object.return_value = {
        'Body': MagicMock(read=lambda: b'cached_image_data')
    }
    client.put_object.return_value = {}
    return client


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    response = MagicMock()
    response.headers = {
        'etag': '"abc123"',
        'last-modified': 'Wed, 01 Jan 2025 00:00:00 GMT',
        'content-type': 'image/jpeg',
        'content-length': '1000'
    }
    response.status_code = 200
    return response


# ============================================================================
# Hat Positioning Fixtures
# ============================================================================

@pytest.fixture
def default_positioning_config():
    """Default hat positioning configuration."""
    return {
        'width_reference': 'eye_distance',
        'width_multiplier': 2.0,
        'hat_anchor_point': {'x': 0.5, 'y': 0.95},
        'horizontal_center': 'midpoint_between_eyes',
        'vertical_anchor': 'forehead_top',
        'vertical_offset_px': 30
    }


@pytest.fixture
def custom_positioning_config():
    """Custom hat positioning configuration for forehead alignment."""
    return {
        'width_reference': 'forehead_width',
        'width_multiplier': 2.5,
        'hat_anchor_point': {'x': 0.5, 'y': 0.9},
        'horizontal_center': 'forehead_top',
        'vertical_anchor': 'forehead_top',
        'vertical_offset_px': 0
    }


# ============================================================================
# API Test Fixtures
# ============================================================================

@pytest.fixture
def mock_face_detector():
    """Create a mock FaceDetector."""
    detector = MagicMock()
    detector.detect_faces.return_value = [{
        'forehead_top': {'x': 320.0, 'y': 100.0},
        'eye_midpoint': {'x': 320.0, 'y': 200.0},
        'eye_distance': 80.0,
        'forehead_width': 100.0,
        'angle': 0.0,
        'head_width': 160.0,
        'all_landmarks': []
    }]
    return detector


@pytest.fixture
def mock_hat_processor(sample_rgba_image):
    """Create a mock SantaHatProcessor."""
    processor = MagicMock()
    processor.process_image.return_value = sample_rgba_image
    return processor


@pytest.fixture
def mock_s3_cache():
    """Create a mock S3Cache with caching disabled."""
    cache = MagicMock()
    cache.enabled = False
    cache.generate_cache_key_from_url = AsyncMock(return_value=None)
    cache.generate_cache_key_from_hash = MagicMock(return_value=None)
    cache.get_cached_image = AsyncMock(return_value=None)
    cache.store_cached_image = AsyncMock(return_value=False)
    return cache
