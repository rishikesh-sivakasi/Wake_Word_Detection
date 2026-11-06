import os
import sys
import time
import numpy as np

# Set environment variable before importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TF logs

try:
    import tensorflow as tf
    import sounddevice as sd
except ImportError:
    print("Missing libraries. Ensure tensorflow and sounddevice are installed.")
    sys.exit(1)

# Add repo to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from multilingual_kws.embedding import input_data

def live_test(model_path, keyword="alexa", threshold=0.82):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    model_settings = input_data.standard_microspeech_model_settings(label_count=3)
    
    sample_rate = 16000
    clip_duration_samples = sample_rate * 1 # 1 second window
    step_samples = sample_rate // 2 # 0.5 sec step
    
    # We will keep a 1-second rolling buffer
    audio_buffer = np.zeros(clip_duration_samples, dtype=np.float32)
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status:
            pass
            
        # indata is shape (frames, 1) and dtype float32
        new_data = indata[:, 0]
        
        # Shift buffer and append
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = new_data

    print("\n" + "="*50)
    print(f"🎤 LIVE LISTENING FOR: '{keyword}'")
    print("Speak into your microphone!")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    # We capture audio using sounddevice stream
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=step_samples, callback=audio_callback):
        try:
            while True:
                time.sleep(0.5)
                # Take current 1 second buffer
                audio_tensor = tf.convert_to_tensor(audio_buffer, dtype=tf.float32)
                
                # Compute spectrogram exactly as the model expects
                spectrogram = input_data.to_micro_spectrogram(model_settings, audio_tensor)
                
                # Reshape to (1, 49, 40, 1)
                spectrogram = tf.expand_dims(spectrogram, axis=0)
                spectrogram = tf.expand_dims(spectrogram, axis=-1)
                
                # Predict
                prediction = model.predict(spectrogram, verbose=0)[0]
                
                # Index 0: Silence, Index 1: Unknown, Index 2: Keyword
                keyword_prob = prediction[2]
                unknown_prob = prediction[1]
                
                if keyword_prob >= threshold:
                    print(f" '{keyword}' DETECTED! (Confidence: {keyword_prob*100:.1f}%)")
                elif unknown_prob >= threshold:
                    # Just to show it's hearing *something* (optional feedback)
                    print(f"   (heard something else...)     ", end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live microphone streaming inference for Multilingual KWS model"
    )
    parser.add_argument(
        "--model", "-m",
        default="alexa_model",
        help="Path to trained model directory (default: 'alexa_model')",
    )
    parser.add_argument(
        "--keyword", "-k",
        default="alexa",
        help="Target keyword label to listen for (default: 'alexa')",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.82,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.82)",
    )
    args = parser.parse_args()

    live_test(args.model, keyword=args.keyword, threshold=args.threshold)
