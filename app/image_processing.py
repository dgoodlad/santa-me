"""Image processing module for adding Santa hats to photos."""
from PIL import Image
import os
import json
from pathlib import Path


class SantaHatProcessor:
    """Handles Santa hat overlay on images."""

    def __init__(self, hat_image_path: str = None):
        """
        Initialize the processor with a Santa hat image.

        Args:
            hat_image_path: Path to the Santa hat PNG with transparent background
        """
        if hat_image_path is None:
            # Default to static/santa_hat.png
            base_dir = Path(__file__).parent.parent
            hat_image_path = base_dir / "static" / "santa_hat.png"

        if not os.path.exists(hat_image_path):
            raise FileNotFoundError(
                f"Santa hat image not found at {hat_image_path}. "
                "Please provide a santa_hat.png file."
            )

        self.hat_image = Image.open(hat_image_path).convert("RGBA")

        # Load hat metadata (semantic positioning configuration)
        metadata_path = Path(hat_image_path).with_suffix('.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.positioning = metadata.get('positioning', self._default_positioning())
        else:
            # Default semantic positioning
            self.positioning = self._default_positioning()

    def _default_positioning(self):
        """Return default semantic positioning configuration."""
        return {
            'width_reference': 'eye_distance',
            'width_multiplier': 2.0,
            'hat_anchor_point': {'x': 0.5, 'y': 0.95},
            'horizontal_center': 'midpoint_between_eyes',
            'vertical_anchor': 'forehead_top',
            'vertical_offset_px': 30
        }

    def add_hat_to_face(
        self,
        image: Image.Image,
        face_data: dict,
        hat_scale: float = 1.0
    ) -> Image.Image:
        """
        Add a Santa hat to a single face using semantic positioning.

        Args:
            image: Original image (will be converted to RGBA)
            face_data: Face detection data with facial measurements
            hat_scale: Optional scale multiplier (default 1.0, uses metadata multiplier)

        Returns:
            Image with Santa hat added
        """
        # Convert image to RGBA if needed
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get the reference measurement based on positioning config
        width_ref = self.positioning['width_reference']
        if width_ref == 'eye_distance':
            reference_width = face_data['eye_distance']
        elif width_ref == 'forehead_width':
            reference_width = face_data['forehead_width']
        else:
            # Default to eye_distance
            reference_width = face_data['eye_distance']

        # Calculate hat width using semantic multiplier
        hat_width = int(reference_width * self.positioning['width_multiplier'] * hat_scale)

        # Maintain aspect ratio
        aspect_ratio = self.hat_image.height / self.hat_image.width
        hat_height = int(hat_width * aspect_ratio)

        # Resize hat
        resized_hat = self.hat_image.resize(
            (hat_width, hat_height),
            Image.Resampling.LANCZOS
        )

        # Rotate hat to match head angle
        angle = face_data['angle']
        rotated_hat = resized_hat.rotate(
            -angle,  # Negative to match head tilt
            expand=True,
            resample=Image.Resampling.BICUBIC
        )

        # Get hat anchor point (which point on the hat should align with target)
        hat_anchor = self.positioning.get('hat_anchor_point', {'x': 0.5, 'y': 0.95})

        # Calculate anchor point on the RESIZED hat (before rotation)
        anchor_x_on_hat = resized_hat.width * hat_anchor['x']
        anchor_y_on_hat = resized_hat.height * hat_anchor['y']

        # Transform the anchor point through rotation
        import math
        rad = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Anchor relative to center of resized hat
        rel_x = anchor_x_on_hat - resized_hat.width / 2
        rel_y = anchor_y_on_hat - resized_hat.height / 2

        # Rotate around origin
        rotated_rel_x = rel_x * cos_a - rel_y * sin_a
        rotated_rel_y = rel_x * sin_a + rel_y * cos_a

        # Transform to rotated hat coordinates
        rotated_anchor_x = rotated_rel_x + rotated_hat.width / 2
        rotated_anchor_y = rotated_rel_y + rotated_hat.height / 2

        # Get target position based on positioning config
        horizontal_center = self.positioning.get('horizontal_center', 'midpoint_between_eyes')
        if horizontal_center == 'midpoint_between_eyes':
            target_x = face_data['eye_midpoint']['x']
        elif horizontal_center == 'forehead_top':
            target_x = face_data['forehead_top']['x']
        else:
            target_x = face_data['eye_midpoint']['x']

        vertical_anchor = self.positioning.get('vertical_anchor', 'forehead_top')
        if vertical_anchor == 'forehead_top':
            target_y = face_data['forehead_top']['y']
        else:
            target_y = face_data['forehead_top']['y']

        # Apply vertical offset
        vertical_offset = self.positioning.get('vertical_offset_px', 0)
        target_y += vertical_offset

        # Position the hat so its rotated anchor point aligns with target
        hat_x = int(target_x - rotated_anchor_x)
        hat_y = int(target_y - rotated_anchor_y)

        # Create a new transparent layer for the hat
        hat_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        hat_layer.paste(rotated_hat, (hat_x, hat_y), rotated_hat)

        # Composite the hat onto the image
        result = Image.alpha_composite(image, hat_layer)

        return result

    def process_image(
        self,
        image: Image.Image,
        faces: list[dict],
        hat_scale: float = 1.0
    ) -> Image.Image:
        """
        Add Santa hats to all detected faces in an image using semantic positioning.

        Args:
            image: Original image
            faces: List of face detection data
            hat_scale: Optional scale multiplier (default 1.0, uses metadata config)

        Returns:
            Image with Santa hats added to all faces
        """
        if not faces:
            # No faces detected, return original
            return image

        # Convert to RGBA for transparency
        result = image.convert("RGBA")

        # Add hat to each face
        for face in faces:
            result = self.add_hat_to_face(result, face, hat_scale)

        return result
