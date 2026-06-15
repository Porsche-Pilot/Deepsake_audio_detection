import os
import numpy as np
import scipy.io.wavfile as wav

def generate_mock_audio(filename, is_genuine, duration=2.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if is_genuine:
        # Generate a clean sine wave
        tone = np.sin(2 * np.pi * 440 * t)
    else:
        # Generate noisy audio
        tone = np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.5, t.shape)
        
    # Normalize to 16-bit PCM range
    audio_data = np.int16(tone / np.max(np.abs(tone)) * 32767)
    wav.write(filename, sample_rate, audio_data)

def main():
    base_dir = 'dataset/LA_norm/train'
    genuine_dir = os.path.join(base_dir, 'genuine')
    spoof_dir = os.path.join(base_dir, 'spoof')
    
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)
    
    print("Generating mock dataset...")
    # Generate 10 genuine files
    for i in range(10):
        generate_mock_audio(os.path.join(genuine_dir, f'gen_{i}.wav'), is_genuine=True)
        
    # Generate 10 spoof files
    for i in range(10):
        generate_mock_audio(os.path.join(spoof_dir, f'spoof_{i}.wav'), is_genuine=False)
        
    print("Mock dataset created successfully!")

if __name__ == "__main__":
    main()
