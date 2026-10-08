from pydantic import BaseModel

# This defines the shape of a successful prediction response.
# When our AI returns a result, it will always have these two fields.
class PredictionResponse(BaseModel):
    prediction: str   # The sound class name, e.g. "dog", "siren"
    confidence: float # The AI's confidence from 0.0 to 100.0

# This defines the shape of an error response.
class ErrorResponse(BaseModel):
    error: str        # A plain-text description of what went wrong
