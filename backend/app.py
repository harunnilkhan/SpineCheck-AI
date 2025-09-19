"""
FastAPI backend for SpineCheck-AI scoliosis detection
with the new minimum enclosing rectangle algorithm.
"""

import os
import sys
import shutil
import uuid
import base64
import numpy as np
import cv2
import torch
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure script dir is added to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import project modules
from backend.models.unet import UNet
from backend.utils.preprocessing import load_and_preprocess_image, postprocess_mask, overlay_mask_on_image
from backend.utils.cobb_angle import analyze_spine_from_image, visualize_results

# Initialize FastAPI app
app = FastAPI(
    title="SpineCheck-AI",
    description="API for scoliosis detection from X-ray images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = os.path.join(project_dir, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Define model path
MODEL_PATH = os.path.join(project_dir, "models", "unet_model_best_colab_harun.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = None

def load_model():
    """Load the UNet model."""
    global model

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        # Look for any pretrained model in the models directory
        models_dir = os.path.join(project_dir, "models")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

        if not model_files:
            print("Warning: No pretrained model found. Please train the model first.")
            return False

        # Use the first available model
        alt_model_path = os.path.join(models_dir, model_files[0])
        print(f"Using alternate model: {alt_model_path}")

        try:
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
            model.load_state_dict(torch.load(alt_model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"Model loaded successfully from {alt_model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        try:
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

@app.on_event("startup")
async def startup_db_client():
    """Initialize resources on startup."""
    print("Initializing SpineCheck-AI backend...")
    load_model()

# Define response models
class VertebraInfo(BaseModel):
    id: int
    center: List[float]
    angle: float
    bbox: List[float]

class AngleInfo(BaseModel):
    angle: float
    vertebra1: int
    vertebra2: int
    vertebra1_center: List[float]
    vertebra2_center: List[float]
    vertebra1_angle: float
    vertebra2_angle: float

class PredictionResult(BaseModel):
    cobb_angles: List[AngleInfo]
    max_angle: float
    classification: str
    vertebrae: List[VertebraInfo]
    visualization_base64: Optional[str] = None
    mask_base64: Optional[str] = None

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "SpineCheck-AI API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        success = load_model()
        if not success:
            return {"status": "warning", "message": "API is running but model is not loaded"}
    return {"status": "healthy", "message": "API is running and model is loaded"}

@app.post("/predict", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    """
    Make predictions on an X-ray image.
    Segments the spine and calculates the Cobb angle.
    """
    print(f"Received prediction request for file: {file.filename}")

    if model is None:
        success = load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)

        # Save uploaded file
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and preprocess image
        input_tensor, original_image = load_and_preprocess_image(temp_filepath)

        # Move tensor to device and make prediction
        input_tensor = input_tensor.to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)

        # Convert output to numpy array
        output_np = output.cpu().numpy()[0, 0]

        # Resize to original size
        original_h, original_w = original_image.shape[:2]
        resized_output = cv2.resize(output_np, (original_w, original_h))

        # Post-process the mask
        processed_mask = postprocess_mask(resized_output)

        # Create a colored representation of the mask for visualization
        mask_colored = np.zeros_like(original_image)
        mask_colored[processed_mask > 0] = [0, 255, 0]  # Green color for vertebrae

        # Overlay the mask on the original image
        mask_overlay = cv2.addWeighted(original_image, 1, mask_colored, 0.5, 0)

        # Analyze spine curvature
        analysis_results = analyze_spine_from_image(mask_overlay)

        # Create visualization
        vis_image = visualize_results(original_image, analysis_results)

        # Convert images to base64
        _, vis_buffer = cv2.imencode('.png', vis_image)
        vis_base64 = base64.b64encode(vis_buffer).decode('utf-8')

        _, mask_buffer = cv2.imencode('.png', mask_overlay)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

        # Add base64 images to response
        analysis_results['visualization_base64'] = vis_base64
        analysis_results['mask_base64'] = mask_base64

        # Clean up
        os.remove(temp_filepath)

        return analysis_results

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-highlighted", response_model=PredictionResult)
async def analyze_highlighted_image(file: UploadFile = File(...)):
    """
    Analyze an X-ray image with already highlighted vertebrae.
    """
    print(f"Received analysis request for highlighted file: {file.filename}")

    try:
        # Read the file
        file_data = await file.read()
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Analyze the spine
        analysis_results = analyze_spine_from_image(image)

        # Create visualization
        vis_image = visualize_results(image, analysis_results)

        # Convert to base64
        _, vis_buffer = cv2.imencode('.png', vis_image)
        vis_base64 = base64.b64encode(vis_buffer).decode('utf-8')

        _, img_buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        analysis_results['visualization_base64'] = vis_base64
        analysis_results['mask_base64'] = img_base64

        return analysis_results

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)