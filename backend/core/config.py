import os

# BASE_DIR = the "backend" folder
# We use __file__ (this file's path) and go up two levels to find it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The frontend folder is one level above backend, then into "frontend"
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

# Where temporary audio files will be saved during a prediction request
UPLOADS_DIR = os.path.join(BASE_DIR, 'dataset', 'uploads')

# Where all the audio training files live
AUDIO_DIR = os.path.join(BASE_DIR, 'dataset', 'audio')

# The CSV file that lists all audio files and their category labels
META_CSV = os.path.join(BASE_DIR, 'dataset', 'meta', 'esc50.csv')

# File path where the trained SVM model will be saved
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'resnet_svm_classifier.pkl')

# Create these folders if they do not exist yet
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
