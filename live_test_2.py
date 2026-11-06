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


def live_test(
    model_path,
    keyword="alexa",
    threshold=0.80,
    hop_sec=0.15,
    cooldown_sec=2.0,
):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    model_settings = input_data.standard_microspeech_model_settings(label_count=3) 

    sample_rate = 16000
    clip_duration_samples = sample_rate * 1  # 1 second window
    step_samples = int(sample_rate * hop_sec)

    # We will keep a 1-second rolling buffer
    audio_buffer = np.zeros(clip_duration_samples, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status:
            pass

        # indata is shape (frames, 1) and dtype float32
        new_data = indata[:, 0]

        # Shift buffer and append atomically to prevent race condition
        new_buffer = np.empty_like(audio_buffer)
        new_buffer[:-frames] = audio_buffer[frames:]
        new_buffer[-frames:] = new_data
        audio_buffer = new_buffer

    print("\n" + "="*50)
    print(f"🎤 LIVE LISTENING FOR: '{keyword}'")
    print(f"Hop: {hop_sec}s | Threshold: {threshold:.2f} | Cooldown: {cooldown_sec}s")
    print("Speak into your microphone!")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")

    last_trigger_time = 0.0

    # We capture audio using sounddevice stream
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=step_samples, callback=audio_callback):
        try:
            while True:
                time.sleep(hop_sec)

                now = time.time()
                in_cooldown = (now - last_trigger_time) < cooldown_sec

                if in_cooldown:
                    remaining = cooldown_sec - (now - last_trigger_time)
                    print(f"   (cooldown {remaining:.1f}s...)     ", end="\r")
                    continue

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
                raw_keyword_prob = prediction[2]
                unknown_prob = prediction[1]

                if raw_keyword_prob >= threshold:
                    print(f"\n '{keyword}' DETECTED! (Confidence: {raw_keyword_prob*100:.1f}%)")
                    last_trigger_time = now
                elif unknown_prob >= threshold:
                    # Just to show it's hearing *something* (optional feedback)
                    print(f"   (heard something else...)     ", end="\r")
                else:
                    # Show live debug so you can see what's happening
                    print(f"   ░ raw:{raw_keyword_prob*100:5.1f}%     ", end="\r")
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
        default=0.80,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.80)",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=0.15,
        help="Sliding window hop step in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown duration after trigger in seconds (default: 2.0)",
    )
    args = parser.parse_args()

    live_test(
        args.model,
        keyword=args.keyword,
        threshold=args.threshold,
        hop_sec=args.hop,
        cooldown_sec=args.cooldown,
    )