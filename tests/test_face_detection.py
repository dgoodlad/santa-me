"""Tests for face detection module."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image


class TestFaceDetector:
    """Tests for the FaceDetector class."""

    def test_init_creates_face_mesh(self):
        """Test that FaceDetector initializes MediaPipe Face Mesh."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh:
            from app.face_detection import FaceDetector
            detector = FaceDetector()

            mock_face_mesh.assert_called_once_with(
                static_image_mode=True,
                max_num_faces=5,
                min_detection_confidence=0.5
            )

    def test_detect_faces_returns_empty_list_when_no_faces(self, sample_rgb_image):
        """Test that detect_faces returns empty list when no faces found."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = MagicMock(multi_face_landmarks=None)
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            assert faces == []

    def test_detect_faces_returns_face_data(
        self, 
        sample_rgb_image, 
        mock_mediapipe_face_landmarks
    ):
        """Test that detect_faces returns proper face data structure."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [mock_mediapipe_face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
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

    def test_detect_faces_calculates_eye_distance(
        self, 
        sample_rgb_image,
        mock_mediapipe_landmark
    ):
        """Test that eye distance is calculated correctly."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            # Create landmarks with known eye positions
            # Image is 640x480
            # Left eye at 0.3 (192px), right eye at 0.7 (448px)
            # Expected distance: 448 - 192 = 256px
            landmarks = [mock_mediapipe_landmark(0.5, 0.5) for _ in range(468)]
            landmarks[10] = mock_mediapipe_landmark(0.5, 0.2)
            landmarks[109] = mock_mediapipe_landmark(0.3, 0.25)
            landmarks[338] = mock_mediapipe_landmark(0.7, 0.25)
            landmarks[151] = mock_mediapipe_landmark(0.5, 0.8)
            landmarks[33] = mock_mediapipe_landmark(0.3, 0.4)   # Left eye at x=192
            landmarks[263] = mock_mediapipe_landmark(0.7, 0.4)  # Right eye at x=448
            
            face_landmarks = MagicMock()
            face_landmarks.landmark = landmarks
            
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            # 640 * (0.7 - 0.3) = 256
            assert faces[0]['eye_distance'] == pytest.approx(256.0, rel=0.01)

    def test_detect_faces_calculates_angle(
        self,
        sample_rgb_image,
        mock_mediapipe_landmark
    ):
        """Test that head tilt angle is calculated correctly."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            # Create landmarks with tilted head
            landmarks = [mock_mediapipe_landmark(0.5, 0.5) for _ in range(468)]
            landmarks[10] = mock_mediapipe_landmark(0.5, 0.2)
            landmarks[109] = mock_mediapipe_landmark(0.35, 0.25)
            landmarks[338] = mock_mediapipe_landmark(0.65, 0.25)
            landmarks[151] = mock_mediapipe_landmark(0.5, 0.8)
            
            # Tilted eyes: right eye slightly higher
            # 640px width, 480px height
            # Left eye at (0.3, 0.42) = (192, 201.6)
            # Right eye at (0.7, 0.38) = (448, 182.4)
            # dx = 256, dy = -19.2
            # angle = atan2(-19.2, 256) ≈ -4.29 degrees
            landmarks[33] = mock_mediapipe_landmark(0.3, 0.42)
            landmarks[263] = mock_mediapipe_landmark(0.7, 0.38)
            
            face_landmarks = MagicMock()
            face_landmarks.landmark = landmarks
            
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            # Angle should be negative (right side tilted up)
            assert faces[0]['angle'] < 0
            assert faces[0]['angle'] == pytest.approx(-4.29, abs=0.5)

    def test_detect_multiple_faces(
        self,
        sample_rgb_image,
        mock_mediapipe_face_landmarks
    ):
        """Test detection of multiple faces."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            # Create two faces
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [
                mock_mediapipe_face_landmarks,
                mock_mediapipe_face_landmarks
            ]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            assert len(faces) == 2

    def test_detect_faces_converts_pil_to_numpy(self, sample_rgb_image):
        """Test that PIL image is converted to numpy array."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = MagicMock(multi_face_landmarks=None)
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            detector.detect_faces(sample_rgb_image)
            
            # Check that process was called with a numpy array
            call_args = mock_face_mesh.process.call_args[0][0]
            assert isinstance(call_args, np.ndarray)
            assert call_args.shape == (480, 640, 3)  # Height, Width, Channels

    def test_detector_cleanup_on_del(self):
        """Test that MediaPipe resources are cleaned up."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_face_mesh = MagicMock()
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            detector.__del__()
            
            mock_face_mesh.close.assert_called_once()

    def test_all_landmarks_are_converted(
        self,
        sample_rgb_image,
        mock_mediapipe_face_landmarks
    ):
        """Test that all 468 landmarks are converted to pixel coordinates."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [mock_mediapipe_face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            # Should have 468 landmarks
            assert len(faces[0]['all_landmarks']) == 468
            
            # Each landmark should be a tuple of (x, y) pixel coordinates
            for landmark in faces[0]['all_landmarks']:
                assert len(landmark) == 2
                assert isinstance(landmark[0], (int, float))
                assert isinstance(landmark[1], (int, float))

    def test_forehead_width_calculation(
        self,
        sample_rgb_image,
        mock_mediapipe_landmark
    ):
        """Test that forehead width is calculated correctly."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            # Create landmarks with known forehead positions
            # Image is 640x480
            landmarks = [mock_mediapipe_landmark(0.5, 0.5) for _ in range(468)]
            landmarks[10] = mock_mediapipe_landmark(0.5, 0.2)  # forehead top
            landmarks[109] = mock_mediapipe_landmark(0.3, 0.25)  # forehead left at x=192
            landmarks[338] = mock_mediapipe_landmark(0.7, 0.25)  # forehead right at x=448
            landmarks[151] = mock_mediapipe_landmark(0.5, 0.8)
            landmarks[33] = mock_mediapipe_landmark(0.35, 0.4)
            landmarks[263] = mock_mediapipe_landmark(0.65, 0.4)
            
            face_landmarks = MagicMock()
            face_landmarks.landmark = landmarks
            
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            # 640 * (0.7 - 0.3) = 256
            assert faces[0]['forehead_width'] == pytest.approx(256.0, rel=0.01)

    def test_eye_midpoint_calculation(
        self,
        sample_rgb_image,
        mock_mediapipe_landmark
    ):
        """Test that eye midpoint is calculated correctly."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            landmarks = [mock_mediapipe_landmark(0.5, 0.5) for _ in range(468)]
            landmarks[10] = mock_mediapipe_landmark(0.5, 0.2)
            landmarks[109] = mock_mediapipe_landmark(0.3, 0.25)
            landmarks[338] = mock_mediapipe_landmark(0.7, 0.25)
            landmarks[151] = mock_mediapipe_landmark(0.5, 0.8)
            landmarks[33] = mock_mediapipe_landmark(0.3, 0.4)   # Left eye
            landmarks[263] = mock_mediapipe_landmark(0.7, 0.4)  # Right eye
            
            face_landmarks = MagicMock()
            face_landmarks.landmark = landmarks
            
            mock_results = MagicMock()
            mock_results.multi_face_landmarks = [face_landmarks]
            
            mock_face_mesh = MagicMock()
            mock_face_mesh.process.return_value = mock_results
            mock_face_mesh_class.return_value = mock_face_mesh

            from app.face_detection import FaceDetector
            detector = FaceDetector()
            
            faces = detector.detect_faces(sample_rgb_image)
            
            # Midpoint should be at x=(192+448)/2=320, y=(192+192)/2=192
            assert faces[0]['eye_midpoint']['x'] == pytest.approx(320.0, rel=0.01)
            assert faces[0]['eye_midpoint']['y'] == pytest.approx(192.0, rel=0.01)
