"""
augment_wake_word.py
====================
Takes your 10 real "Hey Saya" recordings and augments them
into 300 samples ready for training.

HOW TO RUN:
  python3 augment_wake_word.py

STEP 1: Put your 10 real recordings in:
  /home/arz-1029/wake_word_detection/python_training_&_feature_extraction/hey_saya_samples/

STEP 2: Run this script:
  python3 augment_wake_word.py

STEP 3: Check the output folder:
  /home/arz-1029/wake_word_detection/python_training_&_feature_extraction/dataset/train/positive/
  Should contain 290+ WAV files
"""

import os
import numpy as np
import librosa
import soundfile as sf
import random

# ══════════════════════════════════════════════════════════════════
# PATHS — matching your project structure exactly
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = "/home/arz-1029/pt_task/python_training_&_feature_extraction"
INPUT_DIR  = os.path.join(BASE_DIR, "dataset/hey_saya_samples/")        # your 10 real recordings
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "train", "positive")  # augmented output

TARGET_COUNT = 300
SAMPLE_RATE  = 16000
DURATION     = 2.0   # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# AUDIO HELPERS
# ══════════════════════════════════════════════════════════════════
def load_wav(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target   = int(SAMPLE_RATE * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


def save_wav(audio, path):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


# ══════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def pitch_shift(audio, semitones):
    return librosa.effects.pitch_shift(
        audio, sr=SAMPLE_RATE, n_steps=semitones
    ).astype(np.float32)


def time_stretch(audio, rate):
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target    = int(SAMPLE_RATE * DURATION)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    else:
        stretched = stretched[:target]
    return stretched.astype(np.float32)


def volume_change(audio, factor):
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def add_gaussian_noise(audio, level):
    noise = np.random.randn(len(audio)).astype(np.float32) * level
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def time_shift(audio, shift_ms):
    shift_samples = int(SAMPLE_RATE * shift_ms / 1000)
    if shift_samples > 0:
        shifted = np.pad(audio, (shift_samples, 0))[:len(audio)]
    else:
        shifted = np.pad(audio, (0, abs(shift_samples)))
        shifted = shifted[:len(audio)]
    return shifted.astype(np.float32)


def simulate_distance(audio, factor):
    quieter  = audio * factor
    smoothed = np.convolve(quieter, np.ones(3) / 3, mode='same').astype(np.float32)
    return np.clip(smoothed, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# ALL AUGMENTATION COMBINATIONS
# ══════════════════════════════════════════════════════════════════
def get_augmentations():
    aug = []

    # Pitch shifts (6)
    for s in [-3, -2, -1, +1, +2, +3]:
        aug.append((f"pitch_{s:+d}", lambda a, s=s: pitch_shift(a, s)))

    # Time stretch (4)
    for r in [0.85, 0.90, 1.10, 1.15]:
        aug.append((f"speed_{r:.2f}", lambda a, r=r: time_stretch(a, r)))

    # Volume (4)
    for v in [0.5, 0.7, 1.3, 1.5]:
        aug.append((f"vol_{v:.1f}", lambda a, v=v: volume_change(a, v)))

    # Gaussian noise (3)
    for l in [0.003, 0.008, 0.015]:
        aug.append((f"noise_{l:.3f}", lambda a, l=l: add_gaussian_noise(a, l)))

    # Time shift (4)
    for ms in [-150, -75, +75, +150]:
        aug.append((f"shift_{ms}ms", lambda a, ms=ms: time_shift(a, ms)))

    # Distance simulation (2)
    for d in [0.4, 0.65]:
        aug.append((f"dist_{d:.2f}", lambda a, d=d: simulate_distance(a, d)))

    # Combinations (6)
    aug.append(("combo_p1_n1",   lambda a: add_gaussian_noise(pitch_shift(a, +1), 0.005)))
    aug.append(("combo_pm1_n2",  lambda a: add_gaussian_noise(pitch_shift(a, -1), 0.008)))
    aug.append(("combo_slow_q",  lambda a: volume_change(time_stretch(a, 0.88), 0.6)))
    aug.append(("combo_fast_l",  lambda a: volume_change(time_stretch(a, 1.12), 1.4)))
    aug.append(("combo_p2_dist", lambda a: simulate_distance(pitch_shift(a, -2), 0.5)))
    aug.append(("combo_p2_n1",   lambda a: add_gaussian_noise(pitch_shift(a, +2), 0.008)))

    # #Pitch Shift + Time Shift
    # aug.append(("combo_p2_shift100",  lambda a: time_shift(pitch_shift(a, +2), 100)))
    # aug.append(("combo_m2_shiftm100", lambda a: time_shift(pitch_shift(a, -2), -100)))
    
    # #Pitch Shift + Time Stretch (Speed)
    # aug.append(("combo_p1_fast",      lambda a: time_stretch(pitch_shift(a, +1), 1.1)))
    # aug.append(("combo_m1_slow",      lambda a: time_stretch(pitch_shift(a, -1), 0.9)))
    
    # #Time Shift + Gaussian Noise
    # aug.append(("combo_shift50_noise", lambda a: add_gaussian_noise(time_shift(a, 50), 0.01)))
    
    # #The "Extreme" Combo (Pitch + Speed + Time Shift + Noise)
    # aug.append(("combo_extreme_1",    lambda a: add_gaussian_noise(time_shift(time_stretch(pitch_shift(a, +2), 1.15), 80), 0.005)))

    # ==========================================
    # NEW: MASSIVE COMBINATION GRID (81 samples)
    # ==========================================
    # By nesting these loops, we multiply the variations:
    # 3 pitches × 3 speeds × 3 noise levels × 3 shifts = 81 extreme combos!
    
    # pitches = [-2, 0, +2]             # Lower, normal, higher
    # speeds  = [0.85, 1.0, 1.15]       # Slow, normal, fast
    # noises  = [0.0, 0.005, 0.015]     # Clean, light noise, heavy noise
    # shifts  = [-100, 0, +100]         # Early, centered, late
    
    # for p in pitches:
    #     for s in speeds:
    #         for n in noises:
    #             for sh in shifts:
    #                 # Skip the "do nothing" combination because that's just the original file
    #                 if p == 0 and s == 1.0 and n == 0.0 and sh == 0:
    #                     continue
                        
    #                 name = f"grid_p{p}_s{s}_n{n}_sh{sh}"
                    
    #                 # We wrap the lambda to trap the specific variables in the loop
    #                 def make_combo(audio, p=p, s=s, n=n, sh=sh):
    #                     temp = audio
    #                     if p != 0:   temp = pitch_shift(temp, p)
    #                     if s != 1.0: temp = time_stretch(temp, s)
    #                     if sh != 0:  temp = time_shift(temp, sh)
    #                     if n != 0.0: temp = add_gaussian_noise(temp, n)
    #                     return temp
                        
    #                 aug.append((name, make_combo))

    return aug  # Will now return ~108 augmentations per original!

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Wake Word Augmentation")
    print("=" * 55)
    print(f"  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Check input folder exists
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder not found: {INPUT_DIR}")
        print(f"Create the folder and put your 10 WAV files inside it.")
        return

    # Load all real recordings
    wav_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"ERROR: No WAV files found in {INPUT_DIR}")
        return

    print(f"Found {len(wav_files)} real recordings:")
    originals = []
    for path in wav_files:
        try:
            audio = load_wav(path)
            originals.append((os.path.basename(path), audio))
            print(f"  ✅ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ❌ {os.path.basename(path)} — {e}")

    if len(originals) == 0:
        print("ERROR: Could not load any recordings.")
        return

    augmentations = get_augmentations()
    expected      = len(originals) * (1 + len(augmentations))
    print(f"\n{len(originals)} originals × {1 + len(augmentations)} versions")
    print(f"Expected output: ~{expected} samples")
    print()

    saved = 0

    # Save originals first
    for name, audio in originals:
        save_wav(audio, os.path.join(OUTPUT_DIR, f"orig_{name}"))
        saved += 1

    # Apply all augmentations
    print("Generating augmented samples...")
    for name, audio in originals:
        base = name.replace(".wav", "")
        for aug_name, aug_fn in augmentations:
            try:
                augmented = aug_fn(audio)
                save_wav(augmented, os.path.join(OUTPUT_DIR, f"{base}_{aug_name}.wav"))
                saved += 1
            except Exception:
                pass

    print()
    print("=" * 55)
    print(f"  Done! Saved {saved} samples to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 55)
    print()
    print("Next step: python3 train_embedding.py")


if __name__ == "__main__":
    main()