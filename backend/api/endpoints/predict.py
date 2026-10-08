from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil

from core.config import UPLOADS_DIR
from services.prediction_service import process_audio_prediction
from schemas.predict import PredictionResponse

# A "router" is like a mini-app that handles a specific group of URLs.
# In main.py, this router is attached under the "/api" prefix,
# so this endpoint becomes available at: POST /api/predict
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(...),         # The uploaded .wav file from the browser
    model_name: str = Form("svm_dual_modal") # Which AI model to use (sent as a form field)
):
    # Step 1: Make sure the user actually sent a file
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No file was uploaded."})

    # Step 2: Save the uploaded audio file temporarily on disk so our AI can read it
    file_path = os.path.join(UPLOADS_DIR, audio.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Step 3: Send the file path and chosen model to the prediction service
    try:
        prediction_result, confidence_score = process_audio_prediction(file_path, model_name)
        # Step 4: Return the result to the browser as JSON
        return PredictionResponse(prediction=prediction_result, confidence=confidence_score)

    except Exception as e:
        # If anything went wrong, return a readable error message
        return JSONResponse(status_code=500, content={"error": str(e)})
