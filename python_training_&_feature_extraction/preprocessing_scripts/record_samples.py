# Before run this code , run this command on your terminal
# sudo apt-get install libportaudio2
# pip install sounddevice scipy

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# Exact specs for your C++ Ring Buffer
SAMPLE_RATE = 16000  
DURATION = 2.0       # seconds
CHANNELS = 1         # Mono

def record_audio(filename):
    print(f"\n🎤 Get ready... Recording {DURATION} seconds for '{filename}'")
    print("🔴 RECORDING...")
    
    # Capture the audio as a int16 numpy array
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), 
                        samplerate=SAMPLE_RATE, 
                        channels=CHANNELS, 
                        dtype=np.int16)
    
    # Wait until the recording is finished
    sd.wait()  
    print("✅ Done!")

    # Save to disk
    write(filename, SAMPLE_RATE, audio_data)
    print(f"💾 Saved as {filename}\n")

if __name__ == "__main__":
    print("=== Wake Word Sample Recorder ===")
    while True:
        name = input("Enter filename (e.g., noise_1.wav) or 'q' to quit: ")
        if name.lower() == 'q':
            break
        if not name.endswith('.wav'):
            name += '.wav'
            
        record_audio(name)