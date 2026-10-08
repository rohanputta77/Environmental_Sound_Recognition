# SResNet Colab Training Guide

To achieve 80-90% accuracy, we will train the **SResNet** model (Dual-Stream ResNet with SwiGLU Fusion) end-to-end on a free Google Colab GPU. 

### Step 1: Prepare your data
1. Zip your local `dataset` folder (which contains the `audio/` folder and `meta/esc50.csv`). 
2. Upload that `dataset.zip` to your Google Drive.

### Step 2: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to the top menu: **Runtime** -> **Change runtime type** -> Select **T4 GPU** -> Save.

### Step 3: Run the Code
Copy and paste the following Python code into the first cell of your Colab notebook and hit **Run**. 

It will ask you to connect to your Google Drive, extract the dataset, train the AI for 15 epochs, and save the final `sresnet_model.pth` file for you to download!

```python
import os
import zipfile
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. MOUNT GOOGLE DRIVE & EXTRACT DATASET
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

# UPDATE THIS PATH to where you uploaded dataset.zip in your Drive
ZIP_PATH = '/content/drive/MyDrive/dataset.zip' 
EXTRACT_DIR = '/content/dataset'

print("Extracting dataset...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/')
print("Dataset extracted!")

AUDIO_DIR = os.path.join(EXTRACT_DIR, 'audio')
META_CSV = os.path.join(EXTRACT_DIR, 'meta', 'esc50.csv')

# ==========================================
# 2. DEFINE THE DATASET & PREPROCESSING
# ==========================================
class ESC50Dataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.df = pd.read_csv(csv_file)
        # Create a mapping from string category to integer (0-49)
        self.categories = self.df['category'].unique().tolist()
        self.cat_to_id = {cat: i for i, cat in enumerate(self.categories)}
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)
        
    def extract_mel(self, y, sr):
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=224)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        t = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        return t.repeat(3, 1, 1) # RGB shape
        
    def extract_rp(self, y, sr):
        y_paa = y[:(len(y)//224)*224].reshape(224, -1).mean(axis=1)
        rp_matrix = np.abs(y_paa.reshape(-1, 1) - y_paa.reshape(1, -1))
        rp_matrix = (rp_matrix - np.min(rp_matrix)) / (np.max(rp_matrix) - np.min(rp_matrix) + 1e-8)
        t = torch.tensor(rp_matrix, dtype=torch.float32).unsqueeze(0)
        return t.repeat(3, 1, 1) # RGB shape

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label = self.cat_to_id[row['category']]
        
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = librosa.util.fix_length(y, size=5 * sr)
        
        tensor_mel = self.extract_mel(y, sr)
        tensor_rp = self.extract_rp(y, sr)
        
        return tensor_mel, tensor_rp, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. DEFINE THE SRESNET MODEL (Dual-Modal + SwiGLU)
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.linear1(x) * self.swish(self.linear2(x))

class DualStreamSResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Stream 1: Linear Features (Mel Spectrogram)
        self.stream_mel = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_mel.fc = nn.Identity()
        
        # Stream 2: Nonlinear Features (Recurrence Plot)
        self.stream_rp = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stream_rp.fc = nn.Identity()
        
        # SwiGLU Fusion Layer (512 + 512 = 1024 dimensions -> 512)
        self.fusion = SwiGLU(1024, 512)
        
        # Final Classification Head
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, mel, rp):
        f_mel = self.stream_mel(mel)
        f_rp = self.stream_rp(rp)
        
        # Concatenate and pass through SwiGLU gating mechanism
        fused = torch.cat([f_mel, f_rp], dim=1)
        gated_features = self.fusion(fused)
        
        return self.classifier(gated_features)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("Preparing DataLoader...")
dataset = ESC50Dataset(META_CSV, AUDIO_DIR)

# 80/20 Train Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DualStreamSResNet(num_classes=50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    for mel, rp, labels in tqdm(train_loader, desc="Training"):
        mel, rp, labels = mel.to(device), rp.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(mel, rp)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, rp, labels in test_loader:
            mel, rp, labels = mel.to(device), rp.to(device), labels.to(device)
            outputs = model(mel, rp)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Loss: {running_loss/len(train_loader):.4f} | Test Accuracy: {100 * correct / total:.2f}%")

# Save the final model
torch.save(model.state_dict(), '/content/sresnet_model.pth')
import joblib
joblib.dump(dataset.cat_to_id, '/content/category_mapping.pkl')
print("\nTraining Complete! Download 'sresnet_model.pth' and 'category_mapping.pkl' from the files tab.")
```
