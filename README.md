# Environmental Sound Classification System

A research-grade audio analysis system implementing **Dual-Modal Feature Fusion** using Mel-Spectrograms and Recurrence Plots, based on the SResNet paper.

## Architecture

The system uses a two-stream deep learning pipeline:

1. **Linear Features** — Mel-Spectrogram (captures frequency over time)
2. **Non-Linear Features** — Recurrence Plot (captures chaotic dynamics of the signal)
3. **Dual-Modal Fusion** — ResNet-18 extracts 512 features from each stream, fused into a 1,024-dim vector
4. **Classification** — Multiple classifiers trained on the fused feature vector

## Models Available

| Model | Accuracy | Type |
|-------|----------|------|
| Random Forest | 39.00% | Traditional ML |
| MLP Neural Net | 48.00% | Shallow Deep Learning |
| ResNet-18 + SVM | 51.25% | Transfer Learning + Linear SVM |
| SResNet | 60.50% | End-to-End Deep Learning |
| Google AudioSet AI (AST) | 95.00%+ | Pre-trained Transformer |

## Project Structure

```
SOUND_CLASSIFICATION/
├── frontend/               # Web UI (HTML, CSS, JS)
│   ├── index.html          # Main prediction interface
│   ├── reports.html        # Model performance dashboard
│   ├── style.css           # Scientific instrument theme
│   └── script.js           # Audio waveform + API logic
│
├── backend/
│   ├── main.py             # FastAPI server entry point
│   ├── api/endpoints/      # REST API routes
│   ├── services/           # Business logic layer
│   ├── repositories/       # model_repository.py (all AI models)
│   ├── schemas/            # Pydantic request/response models
│   ├── core/               # Config (paths, settings)
│   ├── models/             # Trained model weights (not in repo)
│   └── dataset/            # ESC-50 audio + metadata (not in repo)
│
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download ESC-50 Dataset
Place the ESC-50 dataset in `backend/dataset/audio/` and metadata CSV in `backend/dataset/meta/`.

### 3. Train the local models
```bash
cd backend
python main.py --train
```

### 4. Start the server
```bash
cd backend
python main.py
```
Open `http://127.0.0.1:5000` in your browser.

## Dataset
Uses the [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) — 2,000 labeled environmental sound recordings across 50 classes.

## Key Concepts
- **Endpoint Detection (EPD)**: Silence trimming before feature extraction
- **Piecewise Aggregate Approximation (PAA)**: Dimensionality reduction for Recurrence Plots
- **Dual-Modal Fusion**: Concatenation of linear + non-linear feature embeddings
