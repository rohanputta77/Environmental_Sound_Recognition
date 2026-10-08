import os
import sys
import warnings
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

warnings.filterwarnings('ignore')

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
# Our extra libraries (like transformers) were installed into a "libs" folder
# on the D: drive because the C: drive was full.
# We tell Python to look there when importing packages.
libs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Tell HuggingFace (the company that provides the AST model) to save its
# downloaded AI weights to the D: drive, not C: drive.
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hf_cache')

# Try to import the HuggingFace transformer library.
# If it is not installed, we simply skip it (AST model won't work, but others will).
try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
except ImportError:
    pass

from core.config import AUDIO_DIR, META_CSV, MODEL_SAVE_PATH


# ─── SWIGLU ACTIVATION ────────────────────────────────────────────────────────
# SwiGLU is a modern "gate" used inside the SResNet model's fusion layer.
# Think of it as a smart filter: it decides which parts of the 1024-dim
# fused feature vector are important before passing them to the final classifier.
# It is used by Google's PaLM and Meta's LLaMA — so it's state-of-the-art!
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)  # Main path
        self.linear2 = nn.Linear(in_features, out_features)  # Gate path
        self.swish = nn.SiLU()                               # Smooth activation function

    def forward(self, x):
        # The gate (linear2) decides how much of the main signal (linear1) passes through
        return self.linear1(x) * self.swish(self.linear2(x))


# ─── SRESNET ARCHITECTURE ─────────────────────────────────────────────────────
# This is the main custom deep learning model we built, based on the SResNet paper.
#
# It has TWO separate ResNet-18 networks running in parallel:
#   - stream_mel  → processes the Mel-Spectrogram image (LINEAR features)
#   - stream_rp   → processes the Recurrence Plot image  (NON-LINEAR features)
#
# After both streams run, their 512-dim outputs are concatenated → 1024-dim vector.
# This is "Dual-Modal Feature Fusion".
# Then SwiGLU compresses it to 512 dims, and a final Linear layer gives 50 class scores.
class DualStreamSResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        # Stream 1: For Mel-Spectrogram (linear frequency features)
        self.stream_mel = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_mel.fc = nn.Identity()  # Remove ResNet's final classification head
                                            # so we get raw 512-dim features instead

        # Stream 2: For Recurrence Plot (non-linear dynamics features)
        self.stream_rp = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_rp.fc = nn.Identity()  # Same: get raw 512-dim features

        # Fusion layer: takes 512 + 512 = 1024 dims and compresses to 512
        self.fusion = SwiGLU(1024, 512)

        # Final classifier: maps 512 features to 50 sound class scores
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, mel, rp):
        # Step 1: Extract features from each stream
        f_mel = self.stream_mel(mel)           # shape: (1, 512)
        f_rp  = self.stream_rp(rp)            # shape: (1, 512)

        # Step 2: Fuse (concatenate) them side by side
        fused = torch.cat([f_mel, f_rp], dim=1)  # shape: (1, 1024)

        # Step 3: Apply the SwiGLU gate to compress and filter
        gated_features = self.fusion(fused)    # shape: (1, 512)

        # Step 4: Classify into one of 50 sound categories
        return self.classifier(gated_features) # shape: (1, 50)


# ─── MODEL REPOSITORY ─────────────────────────────────────────────────────────
# This class is the central "model manager" for the whole project.
# It stores all 5 AI models and knows how to:
#   1. Load each model from disk (lazily — only when first needed)
#   2. Extract Dual-Modal features (Mel-Spectrogram + Recurrence Plot)
#   3. Run predictions with any model
#   4. Train the 3 scikit-learn classifiers (SVM, RF, MLP)
class ModelRepository:
    def __init__(self):
        # All models start as None. They are loaded from disk only when
        # a prediction request first asks for them (this is called "lazy loading").
        # This saves RAM on startup — especially important if C: drive is almost full.
        self.extractor  = None  # Shared ResNet-18 feature extractor (for SVM/RF/MLP)
        self.clf        = None  # SVM classifier
        self.rf_clf     = None  # Random Forest classifier
        self.mlp_clf    = None  # MLP Neural Network classifier
        self.sresnet    = None  # Our custom DualStreamSResNet model
        self.ast_extractor = None  # HuggingFace AST audio preprocessor
        self.ast_model     = None  # HuggingFace AST Transformer model
        self.id_to_cat     = None  # Dictionary: {0: "dog", 1: "rain", ...}

    # ── LOADERS ──────────────────────────────────────────────────────────────
    # Each _load_* method loads one model from disk into RAM.
    # They are called automatically before the first prediction.

    def _load_classifier(self):
        """Load the trained SVM from disk."""
        if os.path.exists(MODEL_SAVE_PATH):
            self.clf = joblib.load(MODEL_SAVE_PATH)

    def _load_rf_classifier(self):
        """Load the trained Random Forest from disk."""
        rf_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'random_forest.pkl')
        if os.path.exists(rf_path):
            self.rf_clf = joblib.load(rf_path)

    def _load_mlp_classifier(self):
        """Load the trained MLP Neural Net from disk."""
        mlp_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'mlp_neural_net.pkl')
        if os.path.exists(mlp_path):
            self.mlp_clf = joblib.load(mlp_path)

    def _load_ast(self):
        """Download (or load from cache) the pre-trained Google AudioSet Transformer."""
        print("Downloading/Loading AST Pretrained Model from HuggingFace...")
        self.ast_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.ast_model     = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.ast_model.eval()  # Set to evaluation mode (disables training-only dropout layers)

    def _load_sresnet(self):
        """Load our custom-trained SResNet model and the category label mapping."""
        sresnet_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'sresnet_model.pth')
        mapping_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'category_mapping.pkl')

        if os.path.exists(sresnet_path) and os.path.exists(mapping_path):
            # Load the category-to-ID mapping and reverse it to get ID-to-category
            cat_to_id    = joblib.load(mapping_path)
            self.id_to_cat = {v: k for k, v in cat_to_id.items()}  # {0: "dog", 1: "rain", ...}

            # Create an empty SResNet model and fill it with our trained weights
            self.sresnet = DualStreamSResNet(num_classes=50)
            self.sresnet.load_state_dict(torch.load(sresnet_path, map_location=torch.device('cpu')))
            self.sresnet.eval()  # Set to evaluation mode

    def _build_feature_extractor(self):
        """
        Returns a shared ResNet-18 with its final layer removed.
        This is used by SVM, Random Forest and MLP to extract 512-dim embeddings.
        We only build it once and reuse it every time (cached in self.extractor).
        """
        if self.extractor is None:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = torch.nn.Identity()  # Remove the final 1000-class head
            resnet.eval()
            self.extractor = resnet
        return self.extractor

    # ── FEATURE EXTRACTION ───────────────────────────────────────────────────

    def extract_mel_spectrogram(self, file_path):
        """
        LINEAR FEATURE: Mel-Spectrogram
        Converts a raw audio waveform into a 2D image that shows
        how much energy is present at each frequency over time.
        This captures the "what note is being played and when" information.

        Output: a (1, 3, 224, 224) PyTorch tensor — like a 224x224 RGB image.
        """
        # Load audio at 22050 Hz sample rate and fix length to exactly 5 seconds
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.util.fix_length(y, size=5 * sr)

        # Compute the Mel-Spectrogram (a frequency heatmap over time)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=224)

        # Convert power to decibels (log scale, matches human hearing perception)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert to a PyTorch tensor and resize to 224x224 (ResNet's required input size)
        tensor_img = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_img = torch.nn.functional.interpolate(tensor_img, size=(224, 224), mode='bilinear', align_corners=False)

        # ResNet expects 3 color channels (RGB), so we repeat the single channel 3 times
        tensor_img = tensor_img.repeat(1, 3, 1, 1)
        return tensor_img

    def extract_recurrence_plot(self, file_path):
        """
        NON-LINEAR FEATURE: Recurrence Plot (RP)
        Unlike the Mel-Spectrogram, this captures hidden chaotic patterns
        and self-similarities in the sound signal over time.

        Steps:
          1. Endpoint Detection (EPD): Trim leading/trailing silence
          2. PAA (Piecewise Aggregate Approximation): Compress signal to 224 points
          3. Build distance matrix: RP[i][j] = |signal[i] - signal[j]|
          4. Normalize to [0, 1] range so it can be treated as an image

        Output: a (1, 3, 224, 224) PyTorch tensor — like a 224x224 RGB image.
        """
        y, sr = librosa.load(file_path, sr=22050)

        # Step 1 — Endpoint Detection: remove silence from start and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        y_fixed = librosa.util.fix_length(y_trimmed, size=5 * sr)

        # Step 2 — PAA: compress the signal from ~110,250 samples down to 224 points
        chunk_size = len(y_fixed) // 224
        y_paa = y_fixed[:chunk_size * 224].reshape(224, chunk_size).mean(axis=1)

        # Step 3 — Build the Recurrence Plot (distance matrix)
        # RP[i][j] = absolute difference between the signal at time i and time j
        y_paa_col  = y_paa.reshape(-1, 1)   # column vector (224, 1)
        y_paa_row  = y_paa.reshape(1, -1)   # row vector    (1, 224)
        rp_matrix  = np.abs(y_paa_col - y_paa_row)  # results in a (224, 224) matrix

        # Step 4 — Normalize values to [0, 1] range (so it acts like a grayscale image)
        rp_matrix = (rp_matrix - np.min(rp_matrix)) / (np.max(rp_matrix) - np.min(rp_matrix) + 1e-8)

        # Convert to PyTorch tensor and repeat to 3 channels for ResNet
        tensor_rp = torch.tensor(rp_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_rp = tensor_rp.repeat(1, 3, 1, 1)
        return tensor_rp

    # ── PREDICTION ───────────────────────────────────────────────────────────

    def predict(self, file_path, model_name="svm_dual_modal"):
        """
        Main prediction method. Routes to the correct model based on model_name.
        Returns a tuple: (predicted_label, confidence_score)
          - predicted_label: e.g. "dog bark"
          - confidence_score: float from 0.0 to 100.0
        """
        valid_models = ["svm_dual_modal", "sresnet", "random_forest", "mlp_nn", "ast_pretrained"]
        if model_name not in valid_models:
            raise ValueError(f"Unknown model: '{model_name}'. Choose from {valid_models}")

        # ── Case 1: Google AudioSet Transformer (AST) ──
        if model_name == "ast_pretrained":
            # Load the model on first use (lazy loading)
            if self.ast_model is None:
                self._load_ast()

            # AST requires 16000 Hz audio (not 22050 Hz like our other models)
            y, sr = librosa.load(file_path, sr=16000)

            # Preprocess audio into the format the AST model expects
            inputs = self.ast_extractor(y, sampling_rate=16000, return_tensors="pt")

            with torch.no_grad():  # Disable gradient calculation (we are NOT training)
                logits = self.ast_model(**inputs).logits      # Raw scores for all 527 AudioSet classes
                probs  = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities (sum = 1)
                conf_score = probs.max().item() * 100.0       # Highest probability as a percentage

            predicted_class_idx = logits.argmax(-1).item()    # Index of the winning class
            label = self.ast_model.config.id2label[predicted_class_idx]
            return label, conf_score

        # ── Case 2: Custom SResNet (End-to-End Deep Learning) ──
        if model_name == "sresnet":
            if self.sresnet is None:
                self._load_sresnet()
                if self.sresnet is None:
                    raise FileNotFoundError("SResNet model files not found. Run: python main.py --train")

            # Extract both feature types
            tensor_mel = self.extract_mel_spectrogram(file_path)
            tensor_rp  = self.extract_recurrence_plot(file_path)

            with torch.no_grad():
                outputs    = self.sresnet(tensor_mel, tensor_rp)   # Forward pass through DualStreamSResNet
                probs      = torch.nn.functional.softmax(outputs, dim=1)
                conf_score = probs.max().item() * 100.0
                _, predicted = torch.max(outputs.data, 1)
                pred_id    = predicted.item()

            return self.id_to_cat[pred_id], conf_score

        # ── Case 3: SVM, Random Forest, or MLP ──
        # These three all use the same pipeline:
        #   Extract features → Fuse → Run scikit-learn classifier

        # Load the correct classifier (lazy loading)
        if model_name == "random_forest":
            if self.rf_clf is None:
                self._load_rf_classifier()
                if self.rf_clf is None:
                    raise FileNotFoundError("Random Forest model not found. Run: python main.py --train")
            clf_to_use = self.rf_clf

        elif model_name == "mlp_nn":
            if self.mlp_clf is None:
                self._load_mlp_classifier()
                if self.mlp_clf is None:
                    raise FileNotFoundError("MLP Neural Net model not found. Run: python main.py --train")
            clf_to_use = self.mlp_clf

        else:  # svm_dual_modal
            if self.clf is None:
                self._load_classifier()
                if self.clf is None:
                    raise FileNotFoundError("SVM model not found. Run: python main.py --train")
            clf_to_use = self.clf

        # Use the shared ResNet-18 backbone to extract raw feature embeddings
        extractor = self._build_feature_extractor()

        # Extract both feature types (Dual-Modal)
        tensor_mel = self.extract_mel_spectrogram(file_path)
        tensor_rp  = self.extract_recurrence_plot(file_path)

        with torch.no_grad():
            # Pass each image through ResNet-18 → get 512-dim embedding vectors
            embedding_mel = extractor(tensor_mel).squeeze().numpy()  # (512,)
            embedding_rp  = extractor(tensor_rp).squeeze().numpy()   # (512,)

        # Feature Fusion: join both embeddings end-to-end → 1024-dim vector
        embedding_fused = np.concatenate([embedding_mel, embedding_rp])  # (1024,)

        # Run the scikit-learn classifier on the fused features
        prediction = clf_to_use.predict([embedding_fused])[0]

        # Get confidence using predict_proba (probability per class)
        try:
            probs      = clf_to_use.predict_proba([embedding_fused])[0]
            conf_score = np.max(probs) * 100.0  # Highest probability as percentage
        except AttributeError:
            # Some classifiers don't support predict_proba, so default to 100%
            conf_score = 100.0

        return prediction, conf_score

    # ── TRAINING ─────────────────────────────────────────────────────────────

    def train_pipeline(self):
        """
        Trains all 3 scikit-learn classifiers (SVM, Random Forest, MLP).
        Run this with:  python main.py --train

        Steps:
          1. Load the ESC-50 metadata CSV (list of all audio files + labels)
          2. For each audio file: extract Dual-Modal features
          3. Concatenate (fuse) features into 1024-dim vectors
          4. Split into 80% train / 20% test
          5. Train SVM, Random Forest, and MLP
          6. Print accuracy for each model
          7. Save all models to the "models/" folder
        """
        print(f"Loading metadata from {META_CSV}...")
        df = pd.read_csv(META_CSV)
        print(f"Found {len(df)} audio files. Starting feature extraction...")

        extractor = self._build_feature_extractor()
        features  = []  # Will hold one 1024-dim vector per audio file
        labels    = []  # Will hold the category label for each audio file

        for idx, row in df.iterrows():
            filename = row['filename']
            category = row['category']
            file_path = os.path.join(AUDIO_DIR, filename)

            # Skip files that don't exist on disk
            if not os.path.exists(file_path):
                continue

            try:
                # Extract Dual-Modal features
                tensor_mel = self.extract_mel_spectrogram(file_path)
                tensor_rp  = self.extract_recurrence_plot(file_path)

                with torch.no_grad():
                    embedding_mel = extractor(tensor_mel).squeeze().numpy()
                    embedding_rp  = extractor(tensor_rp).squeeze().numpy()

                # Fuse (concatenate) the two embeddings
                embedding_fused = np.concatenate([embedding_mel, embedding_rp])

                features.append(embedding_fused)
                labels.append(category)

                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} files...")

            except Exception as e:
                print(f"  Skipped {filename}: {e}")

        # Convert lists to numpy arrays (required by scikit-learn)
        X = np.array(features)  # shape: (2000, 1024)
        y = np.array(labels)    # shape: (2000,)

        print("\nFeature extraction complete. Splitting dataset 80/20 train/test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ── Train SVM ──
        print("Training SVM...")
        clf = SVC(kernel='linear', probability=True)  # probability=True enables predict_proba
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  SVM accuracy: {acc * 100:.2f}%")
        joblib.dump(clf, MODEL_SAVE_PATH)
        self.clf = clf

        # ── Train Random Forest ──
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"  Random Forest accuracy: {rf_acc * 100:.2f}%")
        rf_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'random_forest.pkl')
        joblib.dump(rf, rf_path)
        self.rf_clf = rf

        # ── Train MLP Neural Network ──
        print("Training MLP Neural Net...")
        mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
        print(f"  MLP accuracy: {mlp_acc * 100:.2f}%")
        mlp_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'mlp_neural_net.pkl')
        joblib.dump(mlp, mlp_path)
        self.mlp_clf = mlp

        print(f"\nAll models saved to: {os.path.dirname(MODEL_SAVE_PATH)}")


# ─── SINGLETON INSTANCE ───────────────────────────────────────────────────────
# We create ONE shared instance of ModelRepository for the entire application.
# This means models are only loaded into RAM once, not on every request.
model_repo = ModelRepository()
