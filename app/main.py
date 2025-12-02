"""Santa Hat API - Add Santa hats to photos using face detection."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from typing import Optional
import httpx

from app.face_detection import FaceDetector
from app.image_processing import SantaHatProcessor


app = FastAPI(
    title="Santa Hat API",
    description="Add Santa hats to photos using AI face detection",
    version="1.0.0"
)

# Initialize face detector and hat processor (singleton pattern)
face_detector = FaceDetector()
try:
    hat_processor = SantaHatProcessor()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("The API will start but /santa-hatify endpoint will not work until you add a Santa hat image.")
    hat_processor = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Santa Hat API!",
        "endpoints": {
            "/santa-hatify": "POST - Add Santa hats to your photos",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "face_detector": "ready",
        "hat_processor": "ready" if hat_processor else "not configured (missing santa_hat.png)"
    }


@app.post("/santa-hatify")
async def santa_hatify(
    file: Optional[UploadFile] = File(None, description="Image file to process"),
    url: Optional[str] = Form(None, description="URL of image to process"),
    hat_scale: Optional[float] = Form(1.0, description="Optional scale multiplier (default: 1.0)")
):
    """
    Add Santa hats to all faces detected in the uploaded image or image from URL.

    Args:
        file: Image file (JPEG, PNG, etc.) - provide either file or url, not both
        url: URL of image to process - provide either file or url, not both
        hat_scale: Optional multiplier for hat size (default: 1.0, uses metadata config)

    Returns:
        Processed image with Santa hats added
    """
    if hat_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Santa hat processor not configured. Please add static/santa_hat.png file."
        )

    # Validate that exactly one input method is provided
    if file is None and url is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide either a file upload or an image URL"
        )

    if file is not None and url is not None:
        raise HTTPException(
            status_code=400,
            detail="Please provide either a file or URL, not both"
        )

    # Validate hat_scale
    if hat_scale <= 0 or hat_scale > 5:
        raise HTTPException(
            status_code=400,
            detail="hat_scale must be between 0 and 5"
        )

    try:
        # Get image data from either file upload or URL
        if file is not None:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}. Please upload an image file."
                )
            contents = await file.read()
            filename = file.filename
        else:
            # Fetch image from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch image from URL: HTTP {e.response.status_code}"
                    )
                except httpx.RequestError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch image from URL: {str(e)}"
                    )

                # Validate content type
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL does not point to an image. Content-Type: {content_type}"
                    )

                contents = response.content
                # Extract filename from URL or use default
                filename = url.split("/")[-1].split("?")[0] or "image.jpg"

        # Read and open image
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        # Detect faces
        faces = face_detector.detect_faces(image)

        if not faces:
            raise HTTPException(
                status_code=404,
                detail="No faces detected in the image. Please upload an image with visible faces."
            )

        # Process image with Santa hats
        result_image = hat_processor.process_image(image, faces, hat_scale)

        # Convert back to RGB for JPEG output (remove alpha channel)
        if result_image.mode == 'RGBA':
            # Create white background
            rgb_image = Image.new('RGB', result_image.size, (255, 255, 255))
            rgb_image.paste(result_image, mask=result_image.split()[3])  # Use alpha as mask
            result_image = rgb_image

        # Save to bytes buffer
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)

        return StreamingResponse(
            img_buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=santa_{filename}",
                "X-Faces-Detected": str(len(faces))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
