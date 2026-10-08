# PROJECT BIBLE
## Environmental Sound Classification System — Complete Developer Guide

> **Who is this for?**
> A developer who is building this project from scratch and already has the trained model files
> (`*.pkl`, `*.pth`) ready to copy-paste in. No training required. Just build and run.

---

## TABLE OF CONTENTS

1. [The Big Picture — What Does This Project Do?](#1-the-big-picture)
2. [The Research Idea — Why Two Features?](#2-the-research-idea)
3. [Folder Structure — Why is it Organised This Way?](#3-folder-structure)
4. [Step 1 — Set Up The Environment](#4-step-1--set-up-the-environment)
5. [Step 2 — Build The Config File](#5-step-2--build-the-config-file)
6. [Step 3 — Build The AI Brain (model_repository.py)](#6-step-3--build-the-ai-brain)
7. [Step 4 — Build The Schema (Data Contracts)](#7-step-4--build-the-schema)
8. [Step 5 — Build The Service Layer](#8-step-5--build-the-service-layer)
9. [Step 6 — Build The API Endpoint](#9-step-6--build-the-api-endpoint)
10. [Step 7 — Build The Server (main.py)](#10-step-7--build-the-server)
11. [Step 8 — Build The Frontend](#11-step-8--build-the-frontend)
12. [How a Single Prediction Request Flows Through the System](#12-the-full-request-lifecycle)
13. [The Machine Learning Concepts Explained Simply](#13-ml-concepts-explained-simply)
14. [Running the Project](#14-running-the-project)

---

## 1. The Big Picture

**What does this project do?**

A user opens a website, uploads a `.wav` audio file (like a dog barking, or rain), selects an
AI model, clicks "Execute Analysis", and within a few seconds the website shows them:
- The **name** of the detected sound (e.g. "dog")
- A **confidence score** (e.g. 87.3%) — how sure the AI is

That is it from the user's perspective. Under the hood, there is a full AI pipeline running.

**The technology stack:**

| Layer | Technology | What it does |
|---|---|---|
| Frontend | HTML + CSS + JavaScript | The website the user sees |
| Backend Server | FastAPI + Uvicorn | Receives the audio file from the browser |
| Feature Extraction | LibROSA + PyTorch | Converts audio into numbers the AI understands |
| AI Models | Scikit-learn + PyTorch + HuggingFace | Makes the actual prediction |

---

## 2. The Research Idea

**Why not just use one feature type?**

Most basic audio classifiers convert audio into a **Mel-Spectrogram** and call it done.
This project is based on a research paper (**SResNet**) that argues that using only
one type of feature misses important information.

Sound has two types of information:

| Type | Feature | What it Captures |
|---|---|---|
| **LINEAR** | Mel-Spectrogram | Frequency over time (like a heatmap of musical notes) |
| **NON-LINEAR** | Recurrence Plot | Hidden chaotic patterns and self-similarities in the waveform |

The key innovation is **Dual-Modal Feature Fusion**:
1. Extract 512 numbers from the Mel-Spectrogram using a ResNet-18 neural network
2. Extract 512 numbers from the Recurrence Plot using another ResNet-18
3. **Fuse** them: concatenate side by side → **1024 numbers** total
4. Feed that 1024-dim vector to a classifier (SVM, RF, MLP, or SResNet)

This richer representation is what gives higher accuracy than a single-feature approach.

---

## 3. Folder Structure

```
SOUND_CLASSIFICATION/               ← Project root
│
├── frontend/                       ← Everything the browser sees (HTML, CSS, JS)
│   ├── index.html                  ← Main prediction page
│   ├── reports.html                ← Model comparison dashboard
│   ├── style.css                   ← Scientific instrument theme
│   └── script.js                   ← Waveform drawing + API call logic
│
├── backend/                        ← The Python server
│   ├── main.py                     ← Entry point. Starts the server.
│   │
│   ├── core/
│   │   └── config.py               ← All file paths in one place (like an address book)
│   │
│   ├── api/
│   │   └── endpoints/
│   │       └── predict.py          ← Defines the POST /api/predict route
│   │
│   ├── services/
│   │   └── prediction_service.py   ← Coordinates the prediction + cleans up temp files
│   │
│   ├── repositories/
│   │   └── model_repository.py     ← The entire AI brain. All models live here.
│   │
│   ├── schemas/
│   │   └── predict.py              ← Defines what the JSON request/response looks like
│   │
│   ├── models/                     ← Saved model weight files (not in GitHub)
│   │   ├── resnet_svm_classifier.pkl
│   │   ├── random_forest.pkl
│   │   ├── mlp_neural_net.pkl
│   │   ├── sresnet_model.pth
│   │   └── category_mapping.pkl
│   │
│   └── dataset/
│       ├── audio/                  ← ESC-50 audio files (not in GitHub)
│       ├── meta/esc50.csv          ← Metadata CSV (filenames + labels)
│       └── uploads/                ← Temp folder for incoming audio files
│
├── requirements.txt                ← Python library list
├── .gitignore                      ← Files to exclude from GitHub
├── README.md                       ← Project summary for GitHub
└── PROJECT_BIBLE.md                ← This file
```

**Why this folder structure?**

This pattern is called **Layered Architecture**. Each folder has one single responsibility.
The key benefit: if you want to change HOW predictions work, you only touch `model_repository.py`.
You never need to touch `main.py` or the frontend.

Think of it like a restaurant kitchen:
- `api/endpoints/predict.py` = The **waiter** (takes the order from the browser)
- `services/prediction_service.py` = The **head chef** (coordinates the work)
- `repositories/model_repository.py` = The **kitchen** (does the actual AI cooking)
- `schemas/predict.py` = The **menu format** (what a valid order/dish looks like)
- `core/config.py` = The **kitchen address book** (where everything is stored)

---

## 4. Step 1 — Set Up The Environment

### Create the folder structure

```bash
mkdir SOUND_CLASSIFICATION
cd SOUND_CLASSIFICATION
mkdir frontend
mkdir backend
mkdir backend\core
mkdir backend\api
mkdir backend\api\endpoints
mkdir backend\services
mkdir backend\repositories
mkdir backend\schemas
mkdir backend\models
mkdir backend\dataset\audio
mkdir backend\dataset\meta
mkdir backend\dataset\uploads
```

### Create `__init__.py` files

Python needs an empty `__init__.py` file in every folder you want to import from.
Without it, Python will not recognize the folder as a "package" and your imports
will fail with `ModuleNotFoundError`.

```bash
type nul > backend\core\__init__.py
type nul > backend\api\__init__.py
type nul > backend\api\endpoints\__init__.py
type nul > backend\services\__init__.py
type nul > backend\repositories\__init__.py
type nul > backend\schemas\__init__.py
type nul > backend\models\__init__.py
```

### Install dependencies

```bash
pip install fastapi uvicorn python-multipart librosa torch torchvision scikit-learn numpy pandas pydantic
```

**What each library does:**

| Library | Role |
|---|---|
| `fastapi` | Web framework — handles HTTP requests/responses |
| `uvicorn` | ASGI server — actually runs FastAPI (like Apache for Python) |
| `python-multipart` | Required to handle file uploads in FastAPI |
| `librosa` | Audio processing — loads WAV files, computes spectrograms |
| `torch` + `torchvision` | PyTorch deep learning — runs ResNet-18 and SResNet models |
| `scikit-learn` | Traditional ML — trains/runs SVM, Random Forest, MLP |
| `numpy` | Fast math arrays — used everywhere |
| `pandas` | Read the CSV metadata file during training |
| `pydantic` | Data validation — makes sure API inputs/outputs have the right shape |

---

## 5. Step 2 — Build The Config File

**File:** `backend/core/config.py`

**Why this file exists:** Every file in the project needs to know where things are stored.
Instead of typing the same paths in 10 different files, we define them all in ONE place.
If a path ever changes, you only update this one file.

```python
import os

# os.path.abspath(__file__)  →  full path of THIS file, e.g.:
#   "D:\DOWNLOADS\SOUND_CLASSIFICATION\backend\core\config.py"
#
# os.path.dirname(...)       →  removes the last part of the path
#
# So:
#   dirname("...backend\core\config.py")  →  "...backend\core"
#   dirname("...backend\core")            →  "...backend"
#
# We call dirname TWICE to go up two levels and reach the "backend" folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

> **Key concept:** `os.path.abspath(__file__)` always gives the real path of the
> current file on disk, no matter from which directory you run the script. This is
> why paths will not break if you move the project.

```python
# The frontend folder is one level above "backend", then into "frontend"
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')
```

> `os.path.join()` is the correct way to build paths in Python. Never do
> `BASE_DIR + "\\frontend"` because that breaks on Mac/Linux. `os.path.join`
> uses the correct separator automatically for every operating system.

```python
UPLOADS_DIR    = os.path.join(BASE_DIR, 'dataset', 'uploads')
AUDIO_DIR      = os.path.join(BASE_DIR, 'dataset', 'audio')
META_CSV       = os.path.join(BASE_DIR, 'dataset', 'meta', 'esc50.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'resnet_svm_classifier.pkl')

# Create these folders on disk if they do not exist yet.
# exist_ok=True means: do not crash if the folder already exists.
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
```

---

## 6. Step 3 — Build The AI Brain

**File:** `backend/repositories/model_repository.py`

This is the largest and most important file. It is broken into clear sections.

---

### Section A — Imports

```python
import os, sys, warnings
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import torchvision.models as models
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

| Import | Used For |
|---|---|
| `torch`, `torch.nn` | Building and running the custom SResNet neural network |
| `librosa` | Loading audio files and computing Mel-Spectrograms |
| `numpy` | Matrix math for Recurrence Plots and feature fusion |
| `torchvision.models` | Loading the pre-trained ResNet-18 backbone |
| `joblib` | Saving and loading model files (`.pkl`) from disk |
| `SVC` | Support Vector Machine classifier |
| `RandomForestClassifier` | Random Forest classifier |
| `MLPClassifier` | Multi-Layer Perceptron neural network classifier |

---

### Section B — Path and Cache Setup

```python
# Our transformers library was installed to D:\...\backend\libs
# because C: drive was full. We tell Python to look there first.
libs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
```

> `sys.path` is Python's list of folders to search when you do `import something`.
> By inserting our `libs` path at position 0 (the front), Python checks our custom
> folder before looking in the standard library locations.

```python
# Tell HuggingFace to save the 300MB+ AST model weights to D: drive, not C:
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hf_cache')
```

> `os.environ` lets you set system environment variables from Python code.
> HuggingFace reads the `HF_HOME` variable to decide where to cache downloaded models.

---

### Section C — SwiGLU Activation Function

```python
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)  # Main path
        self.linear2 = nn.Linear(in_features, out_features)  # Gate path
        self.swish   = nn.SiLU()                             # Smooth activation

    def forward(self, x):
        return self.linear1(x) * self.swish(self.linear2(x))
```

**In plain English:**

SwiGLU is a *smart filter*. When you fuse the Mel and Recurrence Plot features into
1024 numbers, not all 1024 are equally important. SwiGLU has two parallel paths:
- `linear1` learns which features carry the actual signal
- `linear2` learns a "gate" (0 to 1) that decides how much of each feature passes through

The final output is `signal × gate` — only the important features survive at full strength.

This is the same activation used by Google's PaLM and Meta's LLaMA language models.

---

### Section D — DualStreamSResNet Architecture

```python
class DualStreamSResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        # STREAM 1: Processes the Mel-Spectrogram image
        self.stream_mel = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_mel.fc = nn.Identity()   # Remove the 1000-class head → get raw 512 features

        # STREAM 2: Processes the Recurrence Plot image
        self.stream_rp = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_rp.fc = nn.Identity()    # Same: get raw 512 features

        # FUSION: Takes 512+512=1024 → compresses to 512 using SwiGLU gate
        self.fusion = SwiGLU(1024, 512)

        # CLASSIFIER: Maps 512 features → 50 class scores
        self.classifier = nn.Linear(512, num_classes)
```

**What `nn.Identity()` does:**

ResNet-18 was originally trained on ImageNet to classify 1000 types of images.
Its final layer (`fc`) converts 512 internal features into 1000 class scores.
We do not want those 1000 ImageNet class scores. We want the raw **512 features**.
Replacing `fc` with `nn.Identity()` means "pass the 512 features straight through unchanged."

```python
    def forward(self, mel, rp):
        f_mel  = self.stream_mel(mel)              # (1, 512) features from spectrogram
        f_rp   = self.stream_rp(rp)               # (1, 512) features from recurrence plot
        fused  = torch.cat([f_mel, f_rp], dim=1)  # (1, 1024) concatenated
        gated  = self.fusion(fused)               # (1, 512) filtered by SwiGLU
        return self.classifier(gated)             # (1, 50) class scores
```

**Visualizing the data flow:**

```
Mel Image (224x224)  →  ResNet-18 stream_mel  →  512 features ─┐
                                                                  ├─ concat → 1024 → SwiGLU → 512 → Linear → 50 scores
RP Image  (224x224)  →  ResNet-18 stream_rp   →  512 features ─┘
```

---

### Section E — ModelRepository Class and Lazy Loading

```python
class ModelRepository:
    def __init__(self):
        self.extractor     = None  # Shared ResNet-18 feature extractor
        self.clf           = None  # SVM
        self.rf_clf        = None  # Random Forest
        self.mlp_clf       = None  # MLP Neural Net
        self.sresnet       = None  # Custom SResNet
        self.ast_extractor = None  # HuggingFace audio preprocessor
        self.ast_model     = None  # HuggingFace AST Transformer
        self.id_to_cat     = None  # {0: "dog", 1: "rain", ...}
```

**Why are all models set to `None` at startup?**

This is called the **Lazy Loading** pattern. Loading 5 AI models into RAM at startup
would consume 500MB+ of memory instantly. Instead, each model is loaded only when
a user first requests a prediction using that specific model.

It is exactly like a restaurant that does not cook every dish in advance.
They cook it when you order it. The first order takes a little longer (loading time),
but subsequent orders for the same dish are instant (already in RAM).

---

### Section F — Loader Methods

Each `_load_*` method loads one model from disk into RAM on first use.

```python
def _load_classifier(self):
    if os.path.exists(MODEL_SAVE_PATH):
        self.clf = joblib.load(MODEL_SAVE_PATH)
```

> `joblib.load()` reads a `.pkl` (pickle) file — a serialized Python object.
> A `.pkl` file is the ML equivalent of saving a Word document. All the model's
> learned weights (numbers trained from thousands of audio files) are stored inside.

```python
def _load_sresnet(self):
    sresnet_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'sresnet_model.pth')
    mapping_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'category_mapping.pkl')

    if os.path.exists(sresnet_path) and os.path.exists(mapping_path):
        cat_to_id      = joblib.load(mapping_path)           # {"dog": 0, "rain": 1, ...}
        self.id_to_cat = {v: k for k, v in cat_to_id.items()}  # Reverse it: {0: "dog", ...}
```

> The `{v: k for k, v in dict.items()}` syntax is a **dictionary comprehension** —
> a compact Python way to build a new dictionary by swapping keys and values.
> During training, we stored `{"dog": 0}`. During prediction, the model outputs `0`.
> We need to look up "what is class 0?" — so we reverse the dictionary.

```python
        self.sresnet = DualStreamSResNet(num_classes=50)
        self.sresnet.load_state_dict(
            torch.load(sresnet_path, map_location=torch.device('cpu'))
        )
        self.sresnet.eval()
```

> `load_state_dict()` fills an empty model architecture with trained weights from disk.
> Think of `DualStreamSResNet()` as creating an empty brain with the right structure.
> `load_state_dict()` pours all the learned knowledge (numbers) into that brain.
>
> `map_location='cpu'` ensures the model loads on CPU even if trained on a GPU.
>
> `.eval()` switches from training mode to inference mode. Training mode has techniques
> like Dropout (randomly disabling neurons) that help prevent overfitting during
> training. In inference mode, these are turned off so predictions are consistent.

---

### Section G — Feature Extraction

#### Mel-Spectrogram (Linear Feature)

```python
def extract_mel_spectrogram(self, file_path):
    # 1. Load audio at 22050 Hz (22,050 samples per second = CD quality)
    y, sr = librosa.load(file_path, sr=22050)

    # 2. Pad or truncate to exactly 5 seconds (5 * 22050 = 110,250 samples)
    y = librosa.util.fix_length(y, size=5 * sr)

    # 3. Compute the Mel-Spectrogram
    #    n_fft=2048    → frequency resolution (size of analysis window)
    #    hop_length=512 → step size between windows (time resolution)
    #    n_mels=224    → number of Mel frequency bins (rows of the image)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=224)

    # 4. Convert to decibels (log scale matches human hearing perception
    #    and makes values numerically stable for neural networks)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 5. Convert numpy array → PyTorch tensor
    #    .unsqueeze(0) adds a dimension:
    #      (224, 224) → (1, 224, 224) → (1, 1, 224, 224)
    #    The two 1s represent: batch_size=1, channels=1
    tensor_img = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 6. Resize to 224×224 (ResNet-18 requires exactly this input size)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, size=(224, 224), mode='bilinear', align_corners=False
    )

    # 7. ResNet expects 3 channels (RGB). Our spectrogram has 1.
    #    .repeat(1, 3, 1, 1) duplicates the single channel 3 times:
    #    (1, 1, 224, 224) → (1, 3, 224, 224)
    tensor_img = tensor_img.repeat(1, 3, 1, 1)
    return tensor_img
```

#### Recurrence Plot (Non-Linear Feature)

```python
def extract_recurrence_plot(self, file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # STEP 1 — Endpoint Detection (EPD)
    # Trim silence from the start and end of the recording.
    # top_db=20: anything quieter than 20dB below the loudest point is silence.
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_fixed = librosa.util.fix_length(y_trimmed, size=5 * sr)

    # STEP 2 — Piecewise Aggregate Approximation (PAA)
    # Reduce 110,250 samples down to 224 representative values.
    # We divide the signal into 224 equal chunks and take the average of each.
    chunk_size = len(y_fixed) // 224
    y_paa = y_fixed[:chunk_size * 224].reshape(224, chunk_size).mean(axis=1)
    # Result: y_paa has shape (224,)

    # STEP 3 — Build the Recurrence Plot (Distance Matrix)
    # RP[i][j] = "how different is the signal at time i vs time j?"
    #
    # Reshape to column (224,1) and row (1,224) vectors.
    # NumPy broadcasting subtracts every pair simultaneously → (224, 224) matrix.
    # No for-loop needed — this is very fast.
    y_paa_col = y_paa.reshape(-1, 1)
    y_paa_row = y_paa.reshape(1, -1)
    rp_matrix = np.abs(y_paa_col - y_paa_row)   # shape: (224, 224)

    # STEP 4 — Normalize to [0, 1] range (like image pixel values)
    # + 1e-8 prevents division by zero if the signal is completely silent/flat.
    rp_matrix = (rp_matrix - np.min(rp_matrix)) / (np.max(rp_matrix) - np.min(rp_matrix) + 1e-8)

    tensor_rp = torch.tensor(rp_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor_rp = tensor_rp.repeat(1, 3, 1, 1)
    return tensor_rp
```

---

### Section H — The Predict Method

Called every time a user clicks "Execute Analysis". Routes to the correct model.

#### Route 1: AST Pretrained Transformer

```python
if model_name == "ast_pretrained":
    if self.ast_model is None:
        self._load_ast()               # Load on first use

    # AST was trained on 16kHz audio (not 22kHz like our other models)
    y, sr = librosa.load(file_path, sr=16000)
    inputs = self.ast_extractor(y, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():              # Disable gradient tracking (not training)
        logits     = self.ast_model(**inputs).logits       # Raw scores for 527 classes
        probs      = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities
        conf_score = probs.max().item() * 100.0            # Best probability as %

    predicted_class_idx = logits.argmax(-1).item()
    label = self.ast_model.config.id2label[predicted_class_idx]
    return label, conf_score
```

> **Logits** — raw output numbers from a neural network. They can be any value.
> They are NOT probabilities yet.
>
> **Softmax** — converts logits to probabilities that sum to exactly 1.0.
> Example: `[-2, 8.4, 3.7]` → `[0.001, 0.83, 0.12]`
>
> **`torch.no_grad()`** — disables gradient calculation. Gradients are only needed
> during training. Disabling them saves memory and speeds up inference significantly.
>
> **`.item()`** — converts a PyTorch tensor with a single value to a plain Python number.

#### Route 2: SResNet

```python
if model_name == "sresnet":
    tensor_mel = self.extract_mel_spectrogram(file_path)
    tensor_rp  = self.extract_recurrence_plot(file_path)

    with torch.no_grad():
        outputs    = self.sresnet(tensor_mel, tensor_rp)   # Forward pass → (1, 50) scores
        probs      = torch.nn.functional.softmax(outputs, dim=1)
        conf_score = probs.max().item() * 100.0
        _, predicted = torch.max(outputs.data, 1)          # Index of highest score
        pred_id    = predicted.item()

    return self.id_to_cat[pred_id], conf_score
```

> `torch.max(tensor, dim)` returns TWO values: the max value AND the index.
> We only care about the index (which class?), so we use `_` to discard the value.

#### Route 3: SVM / Random Forest / MLP (shared pipeline)

```python
    extractor = self._build_feature_extractor()   # Shared ResNet-18 backbone

    tensor_mel = self.extract_mel_spectrogram(file_path)
    tensor_rp  = self.extract_recurrence_plot(file_path)

    with torch.no_grad():
        embedding_mel = extractor(tensor_mel).squeeze().numpy()  # (512,)
        embedding_rp  = extractor(tensor_rp).squeeze().numpy()   # (512,)

    # Feature Fusion: join end-to-end → 1024-dim vector
    embedding_fused = np.concatenate([embedding_mel, embedding_rp])  # (1024,)

    prediction = clf_to_use.predict([embedding_fused])[0]

    probs      = clf_to_use.predict_proba([embedding_fused])[0]  # Probability per class
    conf_score = np.max(probs) * 100.0
    return prediction, conf_score
```

> `.squeeze()` removes dimensions of size 1. ResNet output is `(1, 512)`.
> After `.squeeze()` it becomes `(512,)` — a 1D array scikit-learn can work with.
>
> `.numpy()` converts a PyTorch tensor to a NumPy array. Scikit-learn cannot work
> directly with PyTorch tensors — it needs NumPy arrays.

---

## 7. Step 4 — Build The Schema

**File:** `backend/schemas/predict.py`

A schema defines the exact "shape" of data flowing in and out of the API.
FastAPI uses these to automatically validate and serialize responses to JSON.

```python
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str    # e.g. "dog"
    confidence: float  # e.g. 87.3

class ErrorResponse(BaseModel):
    error: str         # e.g. "Model file not found"
```

**What Pydantic does automatically:**
1. If `confidence` is an integer like `87`, it converts it to `87.0`
2. If a required field is missing, it returns a clear error to the client
3. It serializes the Python object to JSON when FastAPI returns the response

---

## 8. Step 5 — Build The Service Layer

**File:** `backend/services/prediction_service.py`

The service layer sits between the API (HTTP handling) and the repository (AI logic).
Its only job: coordinate the call and guarantee cleanup of temp files.

```python
import os
from repositories.model_repository import model_repo

def process_audio_prediction(file_path: str, model_name: str):
    try:
        prediction, confidence = model_repo.predict(file_path, model_name)
        return str(prediction), float(confidence)

    finally:
        # The 'finally' block ALWAYS runs, even if an exception occurred above.
        # This guarantees the temporary audio file is deleted from disk.
        if os.path.exists(file_path):
            os.remove(file_path)
```

**Why `try / finally` instead of `try / except`?**

`try/except` only runs cleanup code when an error happens.
`try/finally` runs cleanup code **always** — whether it succeeded or failed.

If the AI crashes mid-prediction, we still want to delete the temp file.
Otherwise the `uploads/` folder slowly fills up with thousands of orphaned audio files.

---

## 9. Step 6 — Build The API Endpoint

**File:** `backend/api/endpoints/predict.py`

```python
from fastapi import APIRouter, UploadFile, File, Form
router = APIRouter()
```

> `APIRouter` groups related routes. We define the route here and register it in
> `main.py` under the `/api` prefix. This keeps each feature in its own file.

```python
@router.post('/predict', response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(...),
    model_name: str = Form("svm_dual_modal")
):
```

**Decoding each part:**

| Code | Meaning |
|---|---|
| `@router.post('/predict')` | Register this as the handler for `POST /predict` (becomes `/api/predict`) |
| `response_model=PredictionResponse` | FastAPI validates return value matches this schema |
| `UploadFile = File(...)` | Expect a file upload. `...` means REQUIRED. |
| `str = Form("svm_dual_modal")` | Expect a text field. The string is the default value. |
| `async def` | Asynchronous: can handle many requests simultaneously |

```python
    # Save uploaded file temporarily to disk
    file_path = os.path.join(UPLOADS_DIR, audio.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
```

> `"wb"` = "write binary". Audio files are binary data, not text.
> `shutil.copyfileobj` streams the upload directly to disk without loading it all into RAM.

---

## 10. Step 7 — Build The Server

**File:** `backend/main.py`

```python
app = FastAPI(title="Environmental Sound Classification API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

**What is CORS?**

CORS (Cross-Origin Resource Sharing) is a browser security rule. If JavaScript at
`http://localhost:5000` tries to call an API at a different origin, the browser blocks it.
Setting `allow_origins=["*"]` tells the browser: "this API allows requests from anywhere."

```python
app.include_router(predict_router, prefix='/api')
```

> Mounts the router from `predict.py` under `/api`.
> The route `POST /predict` inside the router becomes `POST /api/predict` globally.

```python
@app.get("/{filename:path}")
def serve_static(filename: str):
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
```

> `{filename:path}` is a **path parameter** that matches any URL including slashes.
> Requests for `/style.css`, `/script.js` etc. are all caught here and served from
> the `frontend/` folder. This is how we serve the entire frontend from one FastAPI server.

```python
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        model_repo.train_pipeline()
    else:
        uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)
```

> `if __name__ == '__main__'` only runs when you execute `python main.py` directly.
> If another file imports `main.py`, this block is skipped.
>
> `sys.argv` is the list of command-line arguments.
> Running `python main.py --train` makes `sys.argv = ["main.py", "--train"]`.

---

## 11. Step 8 — Build The Frontend

**File:** `frontend/script.js`

#### Drawing the Real Waveform on Canvas

When the user selects a file, the browser reads it using the Web Audio API and
draws the actual mathematical waveform on the HTML canvas element.

```javascript
fileInput.addEventListener('change', async () => {
    // AudioContext is the browser's built-in audio processing engine
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    const file = fileInput.files[0];
    const arrayBuffer = await file.arrayBuffer();  // Read file as raw bytes

    // Decode the WAV bytes into raw PCM samples (actual amplitude numbers)
    audioContext.decodeAudioData(arrayBuffer, (buffer) => {
        drawWaveform(buffer);
    });
});
```

```javascript
function drawWaveform(buffer) {
    const rawData = buffer.getChannelData(0);  // Get mono channel as Float32Array
    const samples = canvas.clientWidth;        // One column of pixels per sample

    const blockSize = Math.floor(rawData.length / samples);  // Samples per pixel

    for (let i = 0; i < samples; i++) {
        // Find the min and max amplitude in this pixel's chunk
        // This represents the waveform height at this x position
        let min = 1.0, max = -1.0;
        for (let j = 0; j < blockSize; j++) {
            const datum = rawData[(i * blockSize) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        // Draw a vertical rectangle from min to max amplitude at pixel x=i
        ctx.fillRect(i, centerY + min * heightMultiplier, 1, (max - min) * heightMultiplier);
    }
}
```

#### Calling the Prediction API and Showing Real Confidence

```javascript
const formData = new FormData();
formData.append('audio', file);           // Attach the .wav file
formData.append('model_name', modelName); // Attach the selected model name

const response = await fetch('/api/predict', { method: 'POST', body: formData });
const data     = await response.json();   // Parse {"prediction": "dog", "confidence": 87.3}

// Use the REAL confidence number from the backend (not a random fake number)
const realConf = data.confidence.toFixed(1);   // Round to 1 decimal place
confBar.style.width = realConf + '%';          // Animate the progress bar
confValue.innerText = realConf + '%';          // Display the number
```

---

## 12. The Full Request Lifecycle

Here is exactly what happens from the moment you click "Execute Analysis":

```
BROWSER                            FASTAPI SERVER                    AI BRAIN
───────                            ──────────────                    ────────
1. User clicks "Execute Analysis"
2. JS builds FormData
   { audio: file.wav,
     model_name: "svm_dual_modal" }
3. fetch POST /api/predict ──────────────────────────────────►
                                   4. predict() in predict.py receives request
                                   5. Saves file → uploads/filename.wav
                                   6. Calls process_audio_prediction()
                                   7. Calls model_repo.predict() ──────────────►
                                                                   8. Lazy-loads model if first time
                                                                   9. extract_mel_spectrogram() → (1,3,224,224) tensor
                                                                   10. extract_recurrence_plot() → (1,3,224,224) tensor
                                                                   11. ResNet-18 → embedding_mel (512,)
                                                                   11. ResNet-18 → embedding_rp  (512,)
                                                                   12. np.concatenate → fused (1024,)
                                                                   13. clf.predict([fused]) → "dog"
                                                                   14. clf.predict_proba → 87.3%
                                                                  ◄──────────── return ("dog", 87.3)
                                   15. Deletes uploads/filename.wav
                                   16. Returns JSON response:
◄──────────────────────────────────    {"prediction":"dog","confidence":87.3}
17. JS reads data.prediction → "dog"
18. JS reads data.confidence → 87.3
19. Updates DOM:
    predictionText.innerText = "DOG"
    confBar.style.width = "87.3%"
    confValue.innerText = "87.3%"
```

---

## 13. ML Concepts Explained Simply

### What is ResNet-18?

ResNet-18 is a deep neural network with 18 layers, trained by Microsoft on 1.2 million
images. It is extremely good at recognizing visual patterns.

When we feed it a spectrogram or recurrence plot, it extracts 512 numbers (features)
that describe the visual content. This is **Transfer Learning** — using a model trained
on one task (image classification) as a feature extractor for a different task (sound).

### What is an SVM?

A Support Vector Machine finds the best mathematical "dividing line" (hyperplane)
between classes in high-dimensional space.

After ResNet extracts 1024 features from an audio clip, the SVM is like a sorting
machine that says: "given these 1024 measurements, which of 50 sound categories
does this most likely belong to?"

### What is Softmax?

Neural networks output raw "logit" scores — any number, positive or negative.
Softmax converts them into probabilities that sum to 1:

```
Raw logits: [-2.1,  8.4,  3.7,  1.2]
Softmax:    [0.001, 0.83, 0.12, 0.04]   ← all sum to 1.0
```

The highest value (0.83 → 83%) becomes the confidence score.

### What is `predict_proba`?

By default, scikit-learn classifiers only return the winning class name.
`predict_proba` returns a probability for EVERY class, e.g.:
```python
{"dog": 0.73, "cat": 0.15, "rain": 0.08, ...}
```
We take the maximum (0.73 → 73%) as the confidence score.
SVM requires `probability=True` during training to enable this feature.

### Why does the SVM need `probability=True` during training?

By default, SVMs are binary classifiers that only output a class label, not
a probability. Setting `probability=True` makes scikit-learn use an extra
technique (Platt scaling) during training to calibrate the SVM outputs into
proper probability values. This is why we always train with this flag set.

---

## 14. Running the Project

### First time setup (with pre-trained model files)

```bash
# 1. Copy your trained model files into backend/models/:
#    resnet_svm_classifier.pkl
#    random_forest.pkl
#    mlp_neural_net.pkl
#    sresnet_model.pth
#    category_mapping.pkl

# 2. Start the server
cd backend
python main.py
```

Open your browser: **http://127.0.0.1:5000**

### To re-train the scikit-learn models (SVM, RF, MLP)

```bash
cd backend
python main.py --train
```

> Note: SResNet training requires the separate Google Colab notebook (too slow for CPU).

### Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: librosa` | Library not installed | `pip install librosa` |
| `FileNotFoundError: model not found` | `.pkl` missing from `models/` | Copy the model files |
| `No space left on device` | C: drive full | Free space or point `HF_HOME` to another drive |
| `422 Unprocessable Entity` | Wrong form field names in JS | Check `formData.append('audio', ...)` matches `audio: UploadFile` in predict.py |
| Port 5000 already in use | Old server still running | Kill the old process: `taskkill /f /im python.exe` |
