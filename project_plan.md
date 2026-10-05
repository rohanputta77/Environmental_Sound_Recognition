# Environment Sound Classification (ESC-50) Project Plan

Based on the files you shared (a research paper on ESC, a theory document on sound, and a Python hands-on lab), I have formulated an end-to-end approach that perfectly aligns with your mentor's constraints.

## 1. Project Understanding & Constraints
*   **Goal:** Classify environmental sounds using the **ESC-50 dataset** (2,000 environmental audio recordings across 50 classes).
*   **Mentor's Rule:** **Do not build a model from scratch.** You must use existing pre-trained AI models.
*   **Tech Stack:** Simple HTML, CSS, and Vanilla JavaScript for the frontend. Python (Flask/FastAPI) for the backend.
*   **Workflow:**
    *   **Page 1 (Train):** Select an existing AI model and trigger the training process on the dataset.
    *   **Page 2 (Predict):** Upload a single `.wav` audio file, select the trained model, and get the classification result.

---

## 2. Recommended AI Models (Based on your PDFs)

Since you are not allowed to build a model from scratch, you will use a technique called **Transfer Learning / Feature Extraction**. You will pass the audio through a powerful existing model to extract "features" (numerical representations of the sound), and then train a very simple, fast classifier (like Logistic Regression or SVM) on those features.

### Option A: Pre-trained HuBERT (Highly Recommended)
*   **Why:** Your third document ("Audio Basics for Acoustic AI - Hands-On Python Lab") explicitly provides the code to use **HuBERT** (`facebook/hubert-base-ls960`). This is a state-of-the-art transformer model for audio.
*   **Approach:** 
    1. Pass the ESC-50 audio files through the pre-trained HuBERT model to get 1D embeddings (vectors).
    2. Train a simple `scikit-learn` classifier (e.g., Random Forest or Support Vector Machine) on these embeddings.

### Option B: ResNet-18 / VGG-16 with Mel-Spectrograms
*   **Why:** Your first document ("A deep learning framework for environmental sound classification...") discusses using Mel-Spectrograms (converting sound into an image) and feeding them into CNN image models like VGG-16 and ResNet-18.
*   **Approach:** 
    1. Use `librosa` to convert ESC-50 audio files into Log-Mel Spectrogram images.
    2. Pass these images through a pre-trained PyTorch ResNet-18 model to extract image features.
    3. Train a simple classifier on these features.

### Option C: YAMNet
*   **Why:** YAMNet is a lightweight audio event classifier developed by Google, specifically pre-trained on the AudioSet dataset to predict 521 environmental sound classes. It is the industry standard for this exact task.

*Recommendation for the UI:* Provide a dropdown on your website letting the user choose between **HuBERT**, **ResNet-18**, and **YAMNet**.

---

## 3. End-to-End Architecture

### Backend (Python + Flask)
*   **`app.py`**: The main web server.
*   **`/train` endpoint:** Reads the ESC-50 dataset from a local folder on the server. Extracts features using the user-selected model (HuBERT or ResNet), trains a `scikit-learn` classifier in a few seconds, and saves the trained classifier as a `.pkl` file.
*   **`/predict` endpoint:** Receives an uploaded `.wav` file and the model name. Converts the audio to features using the selected base model, runs it through the saved `.pkl` classifier, and returns the predicted label (e.g., "Dog bark" or "Rain").

### Frontend (Simple HTML/CSS/JS)
*   **`index.html` (Training Page):**
    *   A dropdown to select the base model (HuBERT, ResNet-18).
    *   A "Start Training" button. *(Note: Uploading the entire ESC-50 dataset via browser is very slow, so it's better to keep the dataset folder on the backend server and just trigger the training from the UI).*
    *   A status text box showing "Training in progress..." and then "Training Complete! Accuracy: 92%".
*   **`predict.html` (Prediction Page):**
    *   A file input to upload a `.wav` file.
    *   A dropdown to select which trained model to use.
    *   A "Predict" button.
    *   A large text area displaying the result.
*   **`style.css`**: Simple, modern, and clean styling (using gradients, card layouts, and good typography).
*   **`script.js`**: Vanilla JavaScript using `fetch()` to call your Python backend without reloading the page.

---

## 4. Step-by-Step Implementation Plan

### Step 1: Setup & Data Prep
1. Download the **ESC-50 dataset** and place the audio files in a folder named `dataset/audio` inside your backend directory.
2. Install Python requirements: `pip install flask librosa torch torchaudio transformers scikit-learn numpy pandas`.

### Step 2: Build the Backend Logic (`ml_pipeline.py`)
1. **Feature Extraction:** Write a function that takes an audio path, loads it using `librosa` (resampling to 16kHz for HuBERT), and returns the embedding. *(Use the exact code from Section 5.4 of your Hands-On Python Lab PDF!)*
2. **Train Function:** Write a function that loops through the `dataset/audio` folder, extracts features for all files, trains a `sklearn.svm.SVC` model, and saves it using `joblib.dump(model, 'hubert_classifier.pkl')`.
3. **Predict Function:** Write a function that takes a single audio file, extracts its feature, loads the `.pkl` file, and returns `model.predict()`.

### Step 3: Build the Flask API (`app.py`)
1. Setup a Flask app.
2. Serve the static HTML files.
3. Create the `/api/train` POST route.
4. Create the `/api/predict` POST route (handling file uploads).

### Step 4: Build the Frontend
1. Create `index.html` and `predict.html`. 
2. Link them to a shared `style.css`.
3. Write `script.js` to handle form submissions, show loading spinners, and display the JSON responses from Flask.

---

## 5. Summary of how this meets your mentor's rules:
1. **"Don't build the model"**: You are downloading pre-trained HuBERT/ResNet models from HuggingFace/PyTorch. You are only training a basic statistical classifier on top of them.
2. **"Simple frontend"**: No React/Vite. Just pure HTML/CSS/JS served statically.
3. **"2 Pages, Train and Predict"**: The architecture perfectly splits the workflow into these two required pages.
