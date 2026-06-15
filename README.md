# Deepfake Audio Detection

This project provides a complete pipeline for identifying deepfake (AI-generated) audio versus genuine (human) audio. It uses a Convolutional Neural Network (CNN) built with PyTorch, trained on Mel-frequency cepstral coefficients (MFCC) extracted from audio clips.

## Features
- **Data Preprocessing**: Extracts MFCCs using `librosa` and pads/truncates to a fixed length.
- **Model Architecture**: A 2D CNN with 3 convolutional layers followed by fully connected layers, optimized for spectrogram/MFCC inputs.
- **Training Pipeline**: Handles class imbalance, tracks validation loss, and saves the best model based on F1-score.
- **Inference Script**: Predicts new audio files via command-line.
- **Streamlit Web App**: A user-friendly web interface for uploading audio and viewing predictions.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Create a virtual environment (recommended) and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
The project is configured to use **The Fake-or-Real Dataset** (from Kaggle).
- Download the dataset and place the `LA_norm/train` folder in a `dataset/` directory in the root of this project.
- Expected structure:
  ```text
  dataset/LA_norm/train/
  ├── genuine/
  └── spoof/
  ```
*(Note: If your dataset has a different structure or uses a protocol text file, adjust the `load_data` function in `train_pipeline.py`.)*

### 3. Training the Model
Run the training script to train the model and save `deepfake_audio_model.pth`:
```bash
python train_pipeline.py
```
This script will output the training progress, accuracy, EER (Equal Error Rate), F1 score, and confusion matrix.

### 4. Command-Line Inference
Test the trained model on a single audio file:
```bash
python predict.py path/to/audio.wav
```

### 5. Running the Streamlit App
Launch the interactive web app:
```bash
streamlit run app.py
```

## Methodology & Architecture
1. **Feature Extraction**: Audio files are resampled to 16kHz. We extract 40 MFCC features over a fixed window (approx. 3 seconds).
2. **Architecture**: 
   - Conv2D (1->16) -> ReLU -> MaxPool2D
   - Conv2D (16->32) -> ReLU -> MaxPool2D
   - Conv2D (32->64) -> ReLU -> MaxPool2D
   - Flatten -> Linear(128) -> Dropout(0.5) -> Linear(1) -> Sigmoid
3. **Metrics**: Evaluated using Overall Accuracy, Equal Error Rate (EER), F1 Score, and Confusion Matrix to ensure robust performance across both Genuine and Deepfake classes.
