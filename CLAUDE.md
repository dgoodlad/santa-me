# Santa Hat API - Development Guide

## Project Overview

A FastAPI-based REST service that automatically adds Santa hats to photos using AI-powered face detection. The service uses Google's MediaPipe for facial landmark detection and implements a sophisticated semantic positioning system for accurate hat placement.

## Architecture

### Core Components

1. **Face Detection** (`app/face_detection.py`)
   - Uses MediaPipe Face Mesh for 468-point facial landmark detection
   - Extracts key measurements: eye distance, forehead width, eye midpoint, forehead position
   - Calculates head tilt angle for rotation matching
   - Returns structured face data with all measurements

2. **Image Processing** (`app/image_processing.py`)
   - Implements semantic positioning system
   - Handles hat resizing, rotation, and placement
   - Applies anchor point transformation through rotation
   - Supports configurable positioning via JSON metadata

3. **API Server** (`app/main.py`)
   - FastAPI application with `/santa-hatify` endpoint
   - Handles image upload, processing, and response
   - Includes health checks and error handling
   - Returns processed JPEG with face count in headers

## Semantic Positioning System

### Key Innovation

Unlike traditional anchor-point systems, this uses **semantic positioning** - hats are sized and positioned relative to facial features, not arbitrary image coordinates.

### Configuration Format (`static/santa_hat.json`)

```json
{
  "positioning": {
    "width_reference": "eye_distance",      // Measurement to use for sizing
    "width_multiplier": 2.0,                // Hat width = reference × multiplier
    "hat_anchor_point": {                   // Point on hat (0-1 normalized)
      "x": 0.5,                             // Horizontal position on hat
      "y": 0.95                             // Vertical position on hat
    },
    "horizontal_center": "forehead_top",    // Target horizontal position
    "vertical_anchor": "forehead_top",      // Target vertical position
    "vertical_offset_px": 30                // Pixel offset from anchor
  }
}
```

### Why Semantic Positioning?

**Problem with anchor points:** Traditional systems position based on image coordinates, which don't account for:
- Different head sizes (child vs adult)
- Head tilt/rotation
- Varying facial proportions

**Solution:** Position relative to facial landmarks:
- Hat width calculated from eye distance (scales per face automatically)
- Position uses forehead/eye positions (follows face geometry)
- Handles tilted heads by using forehead_top position (shifts with tilt)

### Positioning Algorithm

1. **Sizing**
   ```python
   reference_width = face_data['eye_distance']  # or 'forehead_width'
   hat_width = reference_width × width_multiplier × hat_scale
   ```

2. **Anchor Point Transformation**
   - Calculate anchor point on resized hat (before rotation)
   - Apply rotation matrix to transform anchor point
   - Account for expanded canvas from `rotate(expand=True)`

3. **Final Positioning**
   ```python
   target_x = face_data[horizontal_center]['x']
   target_y = face_data[vertical_anchor]['y'] + vertical_offset_px
   hat_position = target - rotated_anchor_point
   ```

## Important Design Decisions

### 1. Rotation Around Image Center

The hat rotates around its **image center**, not the anchor point. The anchor point is then **transformed through rotation** to determine final placement. This is correct because:
- PIL's `rotate()` operates on image center
- We mathematically transform the anchor point through the same rotation
- Ensures anchor point ends up at the correct facial landmark

### 2. Tilted Head Handling

For tilted heads, using `forehead_top` for horizontal centering is critical:
- When head tilts, forehead shifts left/right from eye center
- Using `eye_midpoint` for horizontal would misalign on tilted heads
- Using `forehead_top` for both x and y keeps hat aligned with actual head position

### 3. Per-Face Scaling

Each face gets independently sized hats:
- Eye distance measured per face
- Multiplier applied to each face's measurement
- Child faces automatically get smaller hats than adult faces

## File Structure

```
santame/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── face_detection.py       # MediaPipe face detection
│   └── image_processing.py     # Hat overlay with semantic positioning
├── static/
│   ├── santa_hat.png          # Hat image (transparent PNG)
│   ├── santa_hat.json         # Positioning configuration
│   └── README.md              # Configuration documentation
├── Dockerfile                 # Multi-stage build, python:3.11-slim
├── docker-compose.yml         # Development container config
├── requirements.txt           # Python dependencies
└── README.md                  # User documentation
```

## Development Workflow

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Docker Development

```bash
# Build and run
docker-compose up --build

# Rebuild after code changes
docker-compose up --build

# Restart for config changes (static/ is mounted)
docker restart santa-hat-api
```

### Testing Hat Positioning

1. Modify `static/santa_hat.json`
2. Restart container: `docker restart santa-hat-api`
3. Test via `/docs` or curl:
   ```bash
   curl -X POST "http://localhost:8000/santa-hatify" \
     -F "file=@photo.jpg" \
     --output result.jpg
   ```

## Tuning Hat Positioning

### Hat Too Large/Small
Adjust `width_multiplier`:
- Smaller (1.5-2.0): Modest hat size
- Larger (2.5-3.0): Prominent hat size

### Hat Too High/Low
Adjust `vertical_offset_px`:
- Positive values: Move down (into hairline)
- Negative values: Move up (above forehead)
- Typical range: 20-60 pixels

### Hat Misaligned Horizontally on Tilted Heads
Use `"horizontal_center": "forehead_top"` instead of `"midpoint_between_eyes"`

### Hat Anchor Point
Adjust `hat_anchor_point` to change which part of hat aligns with target:
- `{"x": 0.5, "y": 1.0}`: Bottom center
- `{"x": 0.5, "y": 0.9}`: Slightly above bottom center
- Typical y range: 0.85-0.95

## Common Issues

### Hats Not Scaling Per Face
- Verify `width_reference` is set correctly
- Check that face detection returns different `eye_distance` for each face
- Ensure `width_multiplier` isn't too large (makes size differences less noticeable)

### Rotation Looks Wrong
- Check angle calculation in `face_detection.py`
- Verify rotation direction (negative angle in `rotate()`)
- Ensure anchor point transformation uses same angle

### Position Shifts After Rotation
- Anchor point must be transformed through rotation
- Check that rotation matrix matches PIL's rotation direction
- Verify `expand=True` expansion is accounted for

## MediaPipe Face Mesh Landmarks

Key landmarks used:
- **10**: Forehead center top
- **33**: Left eye outer corner
- **263**: Right eye outer corner
- **109**: Left forehead side
- **338**: Right forehead side
- **151**: Chin center

[Full landmark map](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

## Future Enhancements

- [ ] Support multiple hat styles with different configs
- [ ] Add perspective transformation for better 3D fitting
- [ ] Implement face contour-based width calculation
- [ ] Add batch processing endpoint
- [ ] Support video/animation output
- [ ] Add hat style selection via API parameter
- [ ] Implement caching for repeated processing

## Dependencies

- **FastAPI**: Web framework
- **MediaPipe**: Face detection (468 landmarks)
- **Pillow**: Image manipulation
- **NumPy**: Mathematical operations
- **Uvicorn**: ASGI server

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Invalid input (file type, parameters)
- `404`: No faces detected
- `500`: Processing error
- `503`: Hat processor not configured

## Performance Considerations

- MediaPipe face detection is CPU-intensive
- Large images take longer to process
- Consider image size limits for production
- Multiple faces increase processing time linearly
- Docker image is ~200MB (python:3.11-slim + dependencies)

## Security Notes

- Validate uploaded file types
- Implement file size limits
- Sanitize filenames if storing uploads
- Consider rate limiting for public deployment
- Review OWASP Top 10 before production deployment
