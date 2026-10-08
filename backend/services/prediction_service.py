import os
from repositories.model_repository import model_repo

def process_audio_prediction(file_path: str, model_name: str):
    """
    This is the "business logic" layer.
    Its only job is to:
      1. Call the AI model to get a prediction
      2. Delete the temporary audio file after it is done
      3. Return the prediction and confidence score back to the API
    """
    try:
        # Ask the model repository to make a prediction.
        # It returns two values: the label (e.g. "dog") and a confidence (e.g. 92.3)
        prediction, confidence = model_repo.predict(file_path, model_name)
        return str(prediction), float(confidence)

    finally:
        # This always runs, even if an error occurred above.
        # It deletes the temporary uploaded file to keep the server clean.
        if os.path.exists(file_path):
            os.remove(file_path)
