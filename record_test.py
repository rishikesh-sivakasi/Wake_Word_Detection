import sys
import wave

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("Missing required libraries. Please install them first by running:")
    print("pip install sounddevice numpy")
    sys.exit(1)

def record_audio(filename, duration=15, samplerate=16000):
    print("=" * 50)
    print(f"🎙️  Recording for {duration} seconds...")
    print("Please speak normally, add some background noise, and say 'hey saya' a few times.")
    print("=" * 50)
    
    # Record audio (mono, 16-bit, 16000 Hz)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Block execution until recording finishes
    
    print("\nRecording finished! Saving to file...")
    
    # Save to WAV file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit PCM (2 bytes)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
        
    print(f"✅ Saved test audio to: {filename}")
    print("\nYou can now run the inference script to test your model:")
    print(f"TF_USE_LEGACY_KERAS=1 PYTHONPATH=. python -m multilingual_kws.run inference --keywords \"hey_saya\" --modelpaths \"hey_saya_model\" --wav \"{filename}\" --visualizer True")

if __name__ == "__main__":
    output_filename = "test_audio_stream.wav"
    record_time = 15 # default to 15 seconds
    
    if len(sys.argv) > 1:
        try:
            record_time = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid number of seconds.")
            sys.exit(1)
            
    record_audio(output_filename, duration=record_time)
