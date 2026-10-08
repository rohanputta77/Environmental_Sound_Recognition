from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints.predict import router as predict_router
from core.config import FRONTEND_DIR
import os
import uvicorn
import sys

# Create the FastAPI application
app = FastAPI(title="Environmental Sound Classification API")

# CORS middleware allows the browser (running on port 5000) to talk to
# this API (also on port 5000) without getting blocked by browser security rules.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],      # Allow GET, POST, etc.
    allow_headers=["*"],
)

# Attach all the /api/predict routes from our router
app.include_router(predict_router, prefix='/api')

# --- Static file routes (serve the frontend HTML pages) ---

@app.get("/")
def serve_index():
    # When someone visits http://127.0.0.1:5000/, send them index.html
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.get("/reports")
def serve_reports():
    # When someone visits http://127.0.0.1:5000/reports, send them reports.html
    return FileResponse(os.path.join(FRONTEND_DIR, 'reports.html'))

@app.get("/{filename:path}")
def serve_static(filename: str):
    # This is a "catch-all" route for static files like style.css and script.js.
    # It looks for the requested file inside the frontend folder and returns it.
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    # If file not found, return 404
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'), status_code=404)


# --- Entry point ---
if __name__ == '__main__':
    # If you run:  python main.py --train
    # it will train all the AI models instead of starting the server.
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        from repositories.model_repository import model_repo
        model_repo.train_pipeline()
    else:
        # Otherwise, start the web server on port 5000
        uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)
