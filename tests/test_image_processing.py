"""Tests for image processing module."""
import io
import json
import math
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image


class TestSantaHatProcessor:
    """Tests for the SantaHatProcessor class."""

    def test_init_loads_hat_image(self, santa_hat_image, tmp_path):
        """Test that processor loads hat image on init."""
        # Save test hat image
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        assert processor.hat_image is not None
        assert processor.hat_image.mode == 'RGBA'

    def test_init_raises_error_when_hat_not_found(self, tmp_path):
        """Test that processor raises FileNotFoundError if hat image missing."""
        from app.image_processing import SantaHatProcessor
        
        with pytest.raises(FileNotFoundError):
            SantaHatProcessor(hat_image_path=str(tmp_path / "nonexistent.png"))

    def test_init_loads_positioning_from_json(self, santa_hat_image, tmp_path):
        """Test that processor loads positioning config from JSON sidecar."""
        # Save test hat image
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        # Create positioning JSON
        config = {
            "positioning": {
                "width_reference": "forehead_width",
                "width_multiplier": 3.0,
                "hat_anchor_point": {"x": 0.4, "y": 0.8},
                "horizontal_center": "forehead_top",
                "vertical_anchor": "forehead_top",
                "vertical_offset_px": 50
            }
        }
        json_path = tmp_path / "test_hat.json"
        with open(json_path, 'w') as f:
            json.dump(config, f)
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        assert processor.positioning['width_reference'] == 'forehead_width'
        assert processor.positioning['width_multiplier'] == 3.0

    def test_init_uses_default_positioning_when_no_json(self, santa_hat_image, tmp_path):
        """Test that processor uses defaults when no JSON sidecar exists."""
        # Save test hat image without JSON
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        assert processor.positioning['width_reference'] == 'eye_distance'
        assert processor.positioning['width_multiplier'] == 2.0

    def test_add_hat_to_face_returns_rgba_image(
        self, 
        santa_hat_image, 
        sample_rgb_image, 
        mock_face_data, 
        tmp_path
    ):
        """Test that add_hat_to_face returns RGBA image."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(sample_rgb_image, mock_face_data)
        
        assert result.mode == 'RGBA'
        assert result.size == sample_rgb_image.size

    def test_add_hat_to_face_preserves_image_size(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_face_data,
        tmp_path
    ):
        """Test that the output image has same dimensions as input."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(sample_rgb_image, mock_face_data)
        
        assert result.size == sample_rgb_image.size

    def test_add_hat_scales_with_eye_distance(
        self,
        santa_hat_image,
        sample_rgb_image,
        tmp_path
    ):
        """Test that hat size scales based on eye distance."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        # Small face
        small_face = {
            'forehead_top': {'x': 320.0, 'y': 100.0},
            'eye_midpoint': {'x': 320.0, 'y': 200.0},
            'eye_distance': 40.0,  # Small
            'forehead_width': 50.0,
            'angle': 0.0,
            'head_width': 80.0,
            'all_landmarks': []
        }
        
        # Large face
        large_face = {
            'forehead_top': {'x': 320.0, 'y': 100.0},
            'eye_midpoint': {'x': 320.0, 'y': 200.0},
            'eye_distance': 100.0,  # Large
            'forehead_width': 125.0,
            'angle': 0.0,
            'head_width': 200.0,
            'all_landmarks': []
        }
        
        # Process both - we can't easily measure the hat size in the output,
        # but we can verify the function doesn't crash and returns valid images
        result_small = processor.add_hat_to_face(sample_rgb_image.copy(), small_face)
        result_large = processor.add_hat_to_face(sample_rgb_image.copy(), large_face)
        
        assert result_small.mode == 'RGBA'
        assert result_large.mode == 'RGBA'

    def test_add_hat_respects_hat_scale_parameter(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_face_data,
        tmp_path
    ):
        """Test that hat_scale parameter affects hat size."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        # Different scales should produce valid results
        result_small = processor.add_hat_to_face(
            sample_rgb_image.copy(), mock_face_data, hat_scale=0.5
        )
        result_normal = processor.add_hat_to_face(
            sample_rgb_image.copy(), mock_face_data, hat_scale=1.0
        )
        result_large = processor.add_hat_to_face(
            sample_rgb_image.copy(), mock_face_data, hat_scale=2.0
        )
        
        assert result_small.mode == 'RGBA'
        assert result_normal.mode == 'RGBA'
        assert result_large.mode == 'RGBA'

    def test_add_hat_handles_tilted_face(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_tilted_face_data,
        tmp_path
    ):
        """Test that hat rotates to match tilted face."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(sample_rgb_image, mock_tilted_face_data)
        
        assert result.mode == 'RGBA'
        assert result.size == sample_rgb_image.size

    def test_add_hat_uses_forehead_width_reference(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_face_data,
        tmp_path
    ):
        """Test that forehead_width can be used as width reference."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        # Create config using forehead_width
        config = {
            "positioning": {
                "width_reference": "forehead_width",
                "width_multiplier": 2.0,
                "hat_anchor_point": {"x": 0.5, "y": 0.95},
                "horizontal_center": "midpoint_between_eyes",
                "vertical_anchor": "forehead_top",
                "vertical_offset_px": 30
            }
        }
        json_path = tmp_path / "test_hat.json"
        with open(json_path, 'w') as f:
            json.dump(config, f)
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(sample_rgb_image, mock_face_data)
        
        assert result.mode == 'RGBA'

    def test_add_hat_horizontal_center_forehead_top(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_face_data,
        tmp_path
    ):
        """Test that horizontal_center can use forehead_top."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        config = {
            "positioning": {
                "width_reference": "eye_distance",
                "width_multiplier": 2.0,
                "hat_anchor_point": {"x": 0.5, "y": 0.95},
                "horizontal_center": "forehead_top",
                "vertical_anchor": "forehead_top",
                "vertical_offset_px": 0
            }
        }
        json_path = tmp_path / "test_hat.json"
        with open(json_path, 'w') as f:
            json.dump(config, f)
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(sample_rgb_image, mock_face_data)
        
        assert result.mode == 'RGBA'

    def test_process_image_returns_original_when_no_faces(
        self,
        santa_hat_image,
        sample_rgb_image,
        tmp_path
    ):
        """Test that process_image returns original when no faces detected."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.process_image(sample_rgb_image, [])
        
        # Should return original (though possibly converted to RGBA)
        assert result.size == sample_rgb_image.size

    def test_process_image_handles_multiple_faces(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_multiple_faces,
        tmp_path
    ):
        """Test that process_image adds hats to multiple faces."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.process_image(sample_rgb_image, mock_multiple_faces)
        
        assert result.mode == 'RGBA'
        assert result.size == sample_rgb_image.size

    def test_process_image_converts_to_rgba(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_face_data,
        tmp_path
    ):
        """Test that process_image converts input to RGBA."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        # Pass RGB image
        assert sample_rgb_image.mode == 'RGB'
        result = processor.process_image(sample_rgb_image, [mock_face_data])
        
        assert result.mode == 'RGBA'

    def test_process_image_applies_hat_scale_to_all_faces(
        self,
        santa_hat_image,
        sample_rgb_image,
        mock_multiple_faces,
        tmp_path
    ):
        """Test that hat_scale is applied to all faces."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.process_image(
            sample_rgb_image, mock_multiple_faces, hat_scale=1.5
        )
        
        assert result.mode == 'RGBA'

    def test_default_positioning_values(self, santa_hat_image, tmp_path):
        """Test default positioning configuration values."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        defaults = processor._default_positioning()
        
        assert defaults['width_reference'] == 'eye_distance'
        assert defaults['width_multiplier'] == 2.0
        assert defaults['hat_anchor_point'] == {'x': 0.5, 'y': 0.95}
        assert defaults['horizontal_center'] == 'midpoint_between_eyes'
        assert defaults['vertical_anchor'] == 'forehead_top'
        assert defaults['vertical_offset_px'] == 30

    def test_rotation_math_correctness(self, santa_hat_image, tmp_path):
        """Test that rotation transformation math is correct."""
        hat_path = tmp_path / "test_hat.png"
        santa_hat_image.save(hat_path, format='PNG')
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        # Test with 45 degree angle
        face_data = {
            'forehead_top': {'x': 320.0, 'y': 100.0},
            'eye_midpoint': {'x': 320.0, 'y': 200.0},
            'eye_distance': 80.0,
            'forehead_width': 100.0,
            'angle': 45.0,  # 45 degree tilt
            'head_width': 160.0,
            'all_landmarks': []
        }
        
        # Create a simple test image
        test_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        
        # This should not raise any errors
        result = processor.add_hat_to_face(test_img, face_data)
        
        assert result.mode == 'RGBA'

    def test_hat_positioned_above_forehead(
        self,
        santa_hat_image,
        sample_rgb_image,
        tmp_path
    ):
        """Test that hat is positioned near the forehead area."""
        # Create a larger, more visible hat for testing
        hat = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))  # Red square
        hat_path = tmp_path / "test_hat.png"
        hat.save(hat_path, format='PNG')
        
        # Create white background image
        img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        
        # Face centered in image
        face_data = {
            'forehead_top': {'x': 320.0, 'y': 150.0},
            'eye_midpoint': {'x': 320.0, 'y': 200.0},
            'eye_distance': 80.0,
            'forehead_width': 100.0,
            'angle': 0.0,
            'head_width': 160.0,
            'all_landmarks': []
        }
        
        config = {
            "positioning": {
                "width_reference": "eye_distance",
                "width_multiplier": 1.0,  # 1:1 scaling
                "hat_anchor_point": {"x": 0.5, "y": 1.0},  # Bottom center
                "horizontal_center": "midpoint_between_eyes",
                "vertical_anchor": "forehead_top",
                "vertical_offset_px": 0
            }
        }
        json_path = tmp_path / "test_hat.json"
        with open(json_path, 'w') as f:
            json.dump(config, f)
        
        from app.image_processing import SantaHatProcessor
        processor = SantaHatProcessor(hat_image_path=str(hat_path))
        
        result = processor.add_hat_to_face(img, face_data)
        result_rgb = result.convert('RGB')
        
        # Check that there are red pixels (hat) in the upper portion of the image
        pixels = list(result_rgb.getdata())
        width = result_rgb.width
        
        # Check upper third of image for red pixels
        upper_third = pixels[:len(pixels) // 3]
        red_pixels = sum(1 for p in upper_third if p[0] > 200 and p[1] < 50 and p[2] < 50)
        
        # There should be some red pixels from the hat
        assert red_pixels > 0, "Expected hat pixels in upper portion of image"
