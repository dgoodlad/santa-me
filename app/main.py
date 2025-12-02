"""Santa Hat API - Add Santa hats to photos using face detection."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from typing import Optional

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
    file: UploadFile = File(..., description="Image file to process"),
    hat_scale: Optional[float] = Form(1.0, description="Optional scale multiplier (default: 1.0)")
):
    """
    Add Santa hats to all faces detected in the uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)
        hat_scale: Optional multiplier for hat size (default: 1.0, uses metadata config)

    Returns:
        Processed image with Santa hats added
    """
    if hat_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Santa hat processor not configured. Please add static/santa_hat.png file."
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )

    # Validate hat_scale
    if hat_scale <= 0 or hat_scale > 5:
        raise HTTPException(
            status_code=400,
            detail="hat_scale must be between 0 and 5"
        )

    try:
        # Read and open image
        contents = await file.read()
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
                "Content-Disposition": f"inline; filename=santa_{file.filename}",
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
