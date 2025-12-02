# Santa Hat API

Add Santa hats to photos using AI-powered face detection! This REST API detects faces in images and automatically positions Santa hats on people's heads.

## Features

- **Free face detection** using Google's MediaPipe
- **Multi-face support** - processes up to 5 faces per image
- **Automatic positioning** - intelligently places hats based on facial landmarks
- **Rotation handling** - rotates hats to match head tilt
- **Adjustable sizing** - customize hat size with the `hat_scale` parameter
- **RESTful API** - easy integration with any frontend or application

## Quick Start

### Option 1: Docker (Recommended)

**1. Add your Santa hat image:**

Place a Santa hat PNG with transparent background at `static/santa_hat.png`

**2. Build and run:**

```bash
docker-compose up --build
```

Or using Docker directly:

```bash
docker build -t santa-hat-api .
docker run -p 8000:8000 -v $(pwd)/static:/app/static:ro santa-hat-api
```

The API will be available at `http://localhost:8000`

### Option 2: Local Python

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Add your Santa hat image:**

Place a Santa hat PNG image with a transparent background at `static/santa_hat.png`

**3. Run the server:**

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Try It Out

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### GET `/santa-hatify` (Slack-friendly)

Add Santa hats to an image from a URL using a simple GET request. Perfect for Slack bots and quick integrations!

**Parameters:**
- `url` (required): URL of the image to process
- `hat_scale` (optional): Scale multiplier (default: 1.0)

**Response:**
- Processed JPEG image with Santa hats
- Header `X-Faces-Detected`: Number of faces found
- Header `X-Cache-Status`: `HIT` or `MISS`

**Example:**

```bash
# Simple URL
https://your-api.com/santa-hatify?url=https://example.com/photo.jpg

# With custom hat scale
https://your-api.com/santa-hatify?url=https://example.com/photo.jpg&hat_scale=1.2
```

**Using curl:**

```bash
curl "https://your-api.com/santa-hatify?url=https://example.com/photo.jpg" \
  --output santa_photo.jpg
```

**Slack Integration:**

Simply paste the URL into Slack and it will display the santa-hatted image:

```
https://your-api.com/santa-hatify?url=https://example.com/photo.jpg
```

Slack will automatically unfurl and display the processed image inline!

### POST `/santa-hatify`

Add Santa hats to all faces in an image (supports file uploads and more options).

**Request options:**

**Option 1: File upload (multipart/form-data)**
- `file`: Image file
- `hat_scale`: Optional float (default: 1.0) - Scale factor for hat size

**Option 2: URL (multipart/form-data)**
- `url`: Image URL
- `hat_scale`: Optional float (default: 1.0) - Scale factor for hat size

**Option 3: URL (application/json)**
```json
{
  "url": "https://example.com/photo.jpg",
  "hat_scale": 1.0
}
```

**Response:**
- Processed JPEG image with Santa hats
- Header `X-Faces-Detected`: Number of faces found
- Header `X-Cache-Status`: `HIT` or `MISS` (when S3 caching enabled)

**Example using curl (file upload):**

```bash
curl -X POST "http://localhost:8000/santa-hatify" \
  -F "file=@photo.jpg" \
  -F "hat_scale=1.0" \
  --output santa_photo.jpg
```

**Example using curl (URL with JSON):**

```bash
curl -X POST "http://localhost:8000/santa-hatify" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg", "hat_scale": 1.0}' \
  --output santa_photo.jpg
```

**Example using curl (URL with form data):**

```bash
curl -X POST "http://localhost:8000/santa-hatify" \
  -F "url=https://example.com/photo.jpg" \
  -F "hat_scale=1.0" \
  --output santa_photo.jpg
```

**Example using Python:**

```python
import requests

with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/santa-hatify",
        files={"file": f},
        data={"hat_scale": 1.8}
    )

with open("santa_photo.jpg", "wb") as f:
    f.write(response.content)

print(f"Faces detected: {response.headers.get('X-Faces-Detected')}")
```

**Example using JavaScript/Fetch:**

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('hat_scale', 1.8);

const response = await fetch('http://localhost:8000/santa-hatify', {
  method: 'POST',
  body: formData
});

const blob = await response.blob();
const facesDetected = response.headers.get('X-Faces-Detected');
console.log(`Faces detected: ${facesDetected}`);
```

### GET `/health`

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "face_detector": "ready",
  "hat_processor": "ready"
}
```

## Configuration

### Hat Scale Parameter

The `hat_scale` parameter controls how large the Santa hat appears relative to the detected head:

- `1.0` - Hat is same width as head
- `1.8` - Default, hat is 1.8x the head width (recommended)
- `2.5` - Larger, more prominent hat
- `0.8` - Smaller, more subtle hat

Valid range: `0.1` to `5.0`

### S3 Caching (Optional)

Enable S3-based caching to speed up repeated requests for the same images. When enabled, processed images are cached in S3 and returned instantly on subsequent requests.

**Setup:**

1. Create a `.env` file (see `.env.example`):

```bash
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

2. For Docker deployments, pass environment variables:

```bash
docker run -p 8000:8000 \
  -e S3_BUCKET_NAME=your-bucket-name \
  -e AWS_ACCESS_KEY_ID=your-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret \
  --env-file .env \
  santa-hat-api
```

**How it works:**

- **URL-based requests:** Cache key uses ETag or Last-Modified headers for smart invalidation
- **File uploads:** Cache key uses SHA-256 hash of file content
- **Cache headers:** Response includes `X-Cache-Status: HIT` or `MISS`
- **Automatic storage:** Processed images are automatically cached after generation

**Benefits:**

- Drastically faster response times for repeated requests (cache hits return instantly)
- Reduced CPU usage and processing costs
- Bandwidth savings for CDN delivery
- Automatic cache invalidation when source images change (URL-based)

Check cache status via the `/health` endpoint:

```json
{
  "status": "healthy",
  "face_detector": "ready",
  "hat_processor": "ready",
  "s3_cache": "enabled"
}
```

## Slack Bot Integration

The GET endpoint is specifically designed for Slack integration. Here's how to use it:

### Quick Start

1. Deploy your API (Railway, Fly.io, etc.)
2. In Slack, paste a URL like:
   ```
   https://your-api.com/santa-hatify?url=https://example.com/photo.jpg
   ```
3. Slack automatically displays the santa-hatted image!

### Building a Slack Bot

**Example slash command handler:**

```python
from slack_bolt import App

app = App(token="YOUR_BOT_TOKEN", signing_secret="YOUR_SIGNING_SECRET")

@app.command("/santahatify")
def santahatify_command(ack, command, respond):
    ack()

    # Get image URL from command text
    image_url = command['text'].strip()

    if not image_url:
        respond("Please provide an image URL!")
        return

    # Generate santa-hatify URL
    api_url = f"https://your-api.com/santa-hatify?url={image_url}"

    # Send the processed image URL back to Slack
    respond({
        "response_type": "in_channel",
        "blocks": [
            {
                "type": "image",
                "image_url": api_url,
                "alt_text": "Santa-hatified image"
            }
        ]
    })

if __name__ == "__main__":
    app.start(port=3000)
```

**Example URL unfurling:**

When someone pastes an image URL in Slack, automatically respond with the santa-hatted version:

```python
@app.event("link_shared")
def handle_link_shared(event, client):
    for link in event['links']:
        url = link['url']
        if is_image_url(url):  # Your image detection logic
            santa_url = f"https://your-api.com/santa-hatify?url={url}"
            client.chat_postMessage(
                channel=event['channel'],
                text=f"Here's the Santa Hat version: {santa_url}"
            )
```

### Tips for Slack Integration

- **S3 caching recommended**: Enables instant responses for repeated images
- **Use `hat_scale`**: Adjust hat size per your preference (default 1.0)
- **Error handling**: The API returns 404 if no faces are detected
- **Timeouts**: First request may take 5-10s for processing, cached requests are instant

## Technical Details

### Face Detection

Uses MediaPipe Face Mesh which detects 468 facial landmarks per face. The API specifically uses:

- Landmark 10: Forehead center (for hat positioning)
- Landmarks 109 & 338: Forehead sides (for hat width)
- Landmarks 33 & 263: Eye corners (for rotation angle)

### Image Processing

1. Detects faces and extracts facial landmarks
2. Calculates optimal hat position, size, and rotation
3. Resizes Santa hat to match head width
4. Rotates hat to match head tilt
5. Composites hat onto original image with transparency

### Supported Formats

- **Input:** JPEG, PNG, GIF, BMP, WebP
- **Output:** JPEG (high quality, 95%)

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Invalid input (wrong file type, invalid parameters)
- `404` - No faces detected
- `500` - Server error
- `503` - Santa hat image not configured

## Development

### Project Structure

```
santame/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── face_detection.py    # MediaPipe face detection
│   └── image_processing.py  # Hat overlay logic
├── static/
│   └── santa_hat.png        # Santa hat image (you provide)
├── requirements.txt
└── README.md
```

### Running Tests

```bash
# TODO: Add tests
pytest tests/
```

## Deployment

### Docker (Recommended)

The included Dockerfile uses `python:3.11-slim` for a minimal container size (~200MB).

**Development:**
```bash
docker-compose up
```

**Production:**
```bash
docker build -t santa-hat-api .
docker run -d \
  -p 8000:8000 \
  -v /path/to/static:/app/static:ro \
  --name santa-hat-api \
  --restart unless-stopped \
  santa-hat-api
```

**Features:**
- Minimal system dependencies
- Built-in health checks
- Volume mount for easy santa_hat.png updates
- Auto-restart on failure

### Direct Python (Production)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Cloud Deployment

Works great on:
- **AWS:** ECS, Fargate, or EC2
- **Google Cloud:** Cloud Run, GKE, or Compute Engine
- **Azure:** Container Instances or App Service
- **Fly.io, Railway, Render:** Direct Docker deployment

## Future Enhancements

- [ ] Support for multiple hat styles (elf hat, reindeer antlers, etc.)
- [ ] Option to return PNG with transparency
- [ ] Batch processing endpoint
- [ ] Face detection confidence threshold parameter
- [ ] Cache detection results for video frames

## License

MIT License - feel free to use this for your holiday projects!

## Credits

- Face detection powered by [MediaPipe](https://google.github.io/mediapipe/)
- API framework: [FastAPI](https://fastapi.tiangolo.com/)
- Image processing: [Pillow](https://python-pillow.org/)
