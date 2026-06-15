import torch
import librosa
import numpy as np
import argparse
import sys
from train_pipeline import AudioCNN, extract_features

def predict_audio(file_path, model_path='deepfake_audio_model.pth'):
    """
    Predict whether an audio file is Genuine or Deepfake.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AudioCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model and 'deepfake_audio_model.pth' exists.")
        sys.exit(1)
        
    # Extract features
    features = extract_features(file_path)
    if features is None:
        print("Failed to extract features from the audio file.")
        sys.exit(1)
        
    # Reshape for model input: (Batch, Channel, Height, Width)
    features = features[np.newaxis, np.newaxis, ...]
    inputs = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(inputs).item()
        
    # Genuine = 1, Deepfake = 0
    confidence = output if output >= 0.5 else 1 - output
    label = "Genuine (Human)" if output >= 0.5 else "Deepfake (AI-Generated)"
    
    print(f"File: {file_path}")
    print(f"Prediction: {label}")
    print(f"Confidence Score: {confidence:.4f}")
    
    return label, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Inference")
    parser.add_argument('file_path', type=str, help='Path to the audio file (.wav, .flac, etc.)')
    parser.add_argument('--model', type=str, default='deepfake_audio_model.pth', help='Path to the trained model weights')
    
    args = parser.parse_args()
    predict_audio(args.file_path, args.model)
