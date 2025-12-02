"""Face detection module using MediaPipe."""
import mediapipe as mp
import numpy as np
from PIL import Image


class FaceDetector:
    """Detects faces and facial landmarks using MediaPipe."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            min_detection_confidence=0.5
        )

    def detect_faces(self, image: Image.Image) -> list[dict]:
        """
        Detect faces in an image and return facial landmarks.

        Args:
            image: PIL Image object

        Returns:
            List of face dictionaries with landmarks and bounding boxes
        """
        # Convert PIL image to numpy array (RGB)
        image_np = np.array(image)

        # Process the image
        results = self.face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return []

        faces = []
        img_height, img_width = image_np.shape[:2]

        for face_landmarks in results.multi_face_landmarks:
            # Get key landmarks for hat positioning
            # Landmark indices (MediaPipe Face Mesh):
            # 10: forehead center top
            # 338: right forehead
            # 109: left forehead
            # 151: chin center
            # 33: left eye outer corner
            # 263: right eye outer corner

            landmarks = face_landmarks.landmark

            # Convert normalized coordinates to pixel coordinates
            forehead_top = landmarks[10]
            forehead_left = landmarks[109]
            forehead_right = landmarks[338]
            chin = landmarks[151]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Calculate hat position and size
            forehead_top_px = {
                'x': forehead_top.x * img_width,
                'y': forehead_top.y * img_height
            }

            forehead_left_px = {
                'x': forehead_left.x * img_width,
                'y': forehead_left.y * img_height
            }

            forehead_right_px = {
                'x': forehead_right.x * img_width,
                'y': forehead_right.y * img_height
            }

            # Calculate facial measurements for semantic positioning
            eye_left_px = {'x': left_eye.x * img_width, 'y': left_eye.y * img_height}
            eye_right_px = {'x': right_eye.x * img_width, 'y': right_eye.y * img_height}

            # Eye distance (outer corners)
            eye_distance = abs(eye_right_px['x'] - eye_left_px['x'])

            # Eye midpoint (for horizontal centering)
            eye_midpoint = {
                'x': (eye_left_px['x'] + eye_right_px['x']) / 2,
                'y': (eye_left_px['y'] + eye_right_px['y']) / 2
            }

            # Forehead width
            forehead_width = abs(forehead_right_px['x'] - forehead_left_px['x'])

            # Calculate head tilt angle
            dx = eye_right_px['x'] - eye_left_px['x']
            dy = eye_right_px['y'] - eye_left_px['y']
            angle = np.degrees(np.arctan2(dy, dx))

            faces.append({
                # Reference points
                'forehead_top': forehead_top_px,
                'eye_midpoint': eye_midpoint,

                # Measurements
                'eye_distance': eye_distance,
                'forehead_width': forehead_width,
                'angle': angle,

                # Legacy (for backwards compatibility)
                'head_width': eye_distance * 2.0,
                'all_landmarks': [(lm.x * img_width, lm.y * img_height)
                                 for lm in landmarks]
            })

        return faces

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
