import os
import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Constants
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 150 # Max frames for MFCC (approx 3 seconds)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

def extract_features(file_path):
    """
    Extract MFCC features from an audio file.
    Pad or truncate the features to a fixed length.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Pad or truncate
        if mfccs.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LEN]
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

class AudioDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = self.labels[idx]
        features = extract_features(file_path)
        
        if features is None:
            # Return zero tensor if failed
            features = np.zeros((N_MFCC, MAX_LEN))
            
        # Reshape to (1, N_MFCC, MAX_LEN) for 2D CNN
        features = features[np.newaxis, ...]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flat_size = 64 * (N_MFCC // 8) * (MAX_LEN // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the threshold where FPR == FNR
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_index]
    return eer

def load_data(data_dir):
    """
    Load data assuming ASVspoof-style protocol file or folder structure.
    For this example, we assume folders `genuine` and `spoof` inside the data_dir,
    or we look for a protocol file.
    """
    files = []
    labels = []
    
    # Simple folder based loading
    genuine_dir = os.path.join(data_dir, 'genuine')
    spoof_dir = os.path.join(data_dir, 'spoof')
    
    if os.path.exists(genuine_dir) and os.path.exists(spoof_dir):
        for f in glob.glob(os.path.join(genuine_dir, '*.wav')) + glob.glob(os.path.join(genuine_dir, '*.flac')):
            files.append(f)
            labels.append(1) # Genuine = 1
        for f in glob.glob(os.path.join(spoof_dir, '*.wav')) + glob.glob(os.path.join(spoof_dir, '*.flac')):
            files.append(f)
            labels.append(0) # Deepfake = 0
    else:
        # If the structure is different, you'll need to adapt this parsing
        print(f"Warning: Expected '{genuine_dir}' and '{spoof_dir}' not found.")
        print("Looking for all audio files in the directory for demonstration.")
        # Mock logic just to have something runnable if dataset isn't structured properly yet
        all_files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True) + \
                    glob.glob(os.path.join(data_dir, '**', '*.flac'), recursive=True)
        for f in all_files:
            files.append(f)
            # Pseudo-label based on filename if 'spoof' or 'fake' in it
            if 'spoof' in f.lower() or 'fake' in f.lower():
                labels.append(0)
            else:
                labels.append(1)
                
    return files, labels

def train():
    data_dir = 'dataset/LA_norm/train' # Adjusted as per problem statement
    print(f"Scanning {data_dir} for audio files...")
    
    files, labels = load_data(data_dir)
    if not files:
        print("No files found. Please ensure the dataset is placed correctly.")
        print("Expected structure: dataset/LA_norm/train/genuine/ and dataset/LA_norm/train/spoof/")
        return
        
    print(f"Found {len(files)} files.")
    
    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    # Class weights to handle imbalance
    num_fake = y_train.count(0)
    num_genuine = y_train.count(1)
    pos_weight = num_fake / num_genuine if num_genuine > 0 else 1.0
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AudioCNN().to(device)
    criterion = nn.BCELoss() # Or use BCEWithLogitsLoss if no sigmoid at the end
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Handle scalar output for batch size 1
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
                
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        predictions = (all_outputs > 0.5).astype(int)
        
        acc = accuracy_score(all_targets, predictions)
        f1 = f1_score(all_targets, predictions)
        eer = compute_eer(all_targets, all_outputs)
        
        print(f"Validation - Accuracy: {acc:.4f}, F1: {f1:.4f}, EER: {eer:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'deepfake_audio_model.pth')
            print("Saved best model.")
            
    # Final evaluation on the test set
    print("\n--- Final Performance Report ---")
    model.load_state_dict(torch.load('deepfake_audio_model.pth'))
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            
    predictions = (np.array(all_outputs) > 0.5).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, predictions))
    print(f"Accuracy: {accuracy_score(all_targets, predictions):.4f}")
    print(f"F1 Score: {f1_score(all_targets, predictions):.4f}")
    print(f"EER: {compute_eer(all_targets, all_outputs):.4f}")

if __name__ == '__main__':
    train()
