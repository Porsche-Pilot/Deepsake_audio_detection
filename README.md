# Deepfake Audio Detection

This project provides a complete pipeline for identifying deepfake (AI-generated) audio versus genuine (human) audio. It uses a Convolutional Neural Network (CNN) built with PyTorch, trained on Mel-frequency cepstral coefficients (MFCC) extracted from audio clips.

## Features
- **Data Preprocessing**: Extracts MFCCs and pads/truncates them to a fixed length.
- **Model Architecture**: A 2D CNN with 3 convolutional layers followed by fully connected layers, optimized for spectrogram/MFCC inputs.
- **Kaggle Training Pipeline**: An executable Jupyter Notebook (`notebook.ipynb`) designed to train seamlessly on Kaggle's free GPUs using The Fake-or-Real Dataset.
- **Pre-trained Model**: Includes the fully trained model (`deepfake_audio_model.pth`) ready for inference.
- **Streamlit Web App**: A beautiful, premium corporate web interface for uploading audio and viewing real-time deepfake predictions.
- **Command-Line Inference**: Predicts new audio files via `predict.py`.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Web App (Recommended)
Launch the interactive web app to test the pre-trained model on any audio file:
```bash
streamlit run app.py
```
This will open the "Audio Integrity Verification System" in your browser. Upload any `.wav` or `.mp3` file to see if it is classified as a "Genuine" human voice or an "AI Deepfake".

### 3. Command-Line Inference
You can also test the trained model on a single audio file via the terminal:
```bash
python predict.py path/to/audio.wav
```

## Model Training (Kaggle)
The model was trained using the `notebook.ipynb` file directly on Kaggle. To retrain the model yourself:
1. Upload `notebook.ipynb` to a new Kaggle Notebook.
2. Add "The Fake-or-Real Dataset" to the notebook environment.
3. Turn on the GPU accelerator and click "Run All".
4. The notebook will calculate Error Metrics (Accuracy, F1, EER, Confusion Matrix) and output the `deepfake_audio_model.pth` file.

## Methodology & Architecture
1. **Feature Extraction**: Audio files are resampled to 16kHz. We extract 40 MFCC features over a fixed window (approx. 3 seconds).
2. **Architecture**: 
   - Conv2D (1->16) -> ReLU -> MaxPool2D
   - Conv2D (16->32) -> ReLU -> MaxPool2D
   - Conv2D (32->64) -> ReLU -> MaxPool2D
   - Flatten -> Linear(128) -> Dropout(0.5) -> Linear(1) -> Sigmoid
3. **Metrics**: Evaluated using Overall Accuracy, Equal Error Rate (EER), F1 Score, and Confusion Matrix to ensure robust performance across both Genuine and Deepfake classes.
