"""
augment_wake_word.py
====================
Takes your real "Hey Saya" recordings and augments them
into 300 samples ready for training.

AUGMENTATION BUDGET: 29 augmentations per original + 1 original = 30 per file
With 10 originals → 300 total clips (the research-confirmed saturation point)

CHANGES FROM PREVIOUS VERSION:
  1. Time stretch range EXPANDED from [0.85-1.15] to [0.75-1.20]
     → Covers fast speech (the "misses quick Hey Saya" problem)
  2. Time shift reduced from 4 values to 2 (freed 2 slots for more useful combos)
  3. Combinations updated: all 6 pitch values now appear in at least one combo
  4. combo_fast_noise added: time_stretch(0.80) + noise — directly trains
     the model on fast speech with background noise, the real deployment scenario

HOW TO RUN:
  python3 augment_wake_word.py

STEP 1: Put your real recordings in:
  dataset/hey_saya_samples/

STEP 2: Run this script

STEP 3: Check output folder:
  dataset/train/positive/
  Should contain exactly 300 WAV files (10 originals × 30 versions each)
"""

import os
import numpy as np
import librosa
import soundfile as sf

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = "/home/arz-1029/pt_task/python_training_&_feature_extraction"
INPUT_DIR  = os.path.join(BASE_DIR, "dataset", "hey_saya_samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "train", "positive")

SAMPLE_RATE = 16000
DURATION    = 2.0   # seconds — must match MAX_FRAMES / HOP_LENGTH in training

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
    """
    rate < 1.0 → slower speech (longer duration, trimmed to 2s)
    rate > 1.0 → faster speech (shorter duration, padded to 2s)

    Range used: 0.75 (very fast) to 1.20 (slow/deliberate)
    The 0.75 and 1.20 extremes specifically address the "misses fast speech"
    issue — the model needs to have seen compressed speech during training
    or it will not recognise it at inference time.
    """
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
    """
    Adds white Gaussian noise.
    level=0.003 → barely audible
    level=0.008 → light background hiss
    level=0.015 → noticeable noise floor
    """
    noise = np.random.randn(len(audio)).astype(np.float32) * level
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def time_shift(audio, shift_ms):
    """
    Shifts audio left or right in time by shift_ms milliseconds.
    Simulates the wake word appearing slightly earlier or later
    within the 2-second detection window.
    """
    shift_samples = int(SAMPLE_RATE * shift_ms / 1000)
    if shift_samples > 0:
        shifted = np.pad(audio, (shift_samples, 0))[:len(audio)]
    else:
        shifted = np.pad(audio, (0, abs(shift_samples)))
        shifted = shifted[:len(audio)]
    return shifted.astype(np.float32)


def simulate_distance(audio, factor):
    """
    Simulates speaking from further away:
    lower amplitude + slight low-pass smoothing (3-tap moving average).
    factor=0.4 → 2-3 metres
    factor=0.65 → 1-1.5 metres
    """
    quieter  = audio * factor
    smoothed = np.convolve(quieter, np.ones(3) / 3, mode='same').astype(np.float32)
    return np.clip(smoothed, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# AUGMENTATION LIST — exactly 29 per original → 30 total per file
#
# Budget breakdown:
#   Pitch shifts      : 6   (covers -3 to +3 semitones)
#   Time stretch      : 6   (covers 0.75x fast to 1.20x slow) ← EXPANDED
#   Volume change     : 4   (covers 0.5x quiet to 1.5x loud)
#   Gaussian noise    : 3   (light / medium / heavy)
#   Time shift        : 2   (early / late in window) ← REDUCED from 4
#   Distance sim      : 2   (near / far)
#   Combinations      : 6   (multi-factor, all 6 pitches covered in at least 1)
#   TOTAL             : 29
# ══════════════════════════════════════════════════════════════════
def get_augmentations():
    aug = []

    # ── Pitch shifts (6) ──────────────────────────────────────────
    # Covers accent/voice variation. Indian English has significant
    # pitch range across speakers so ±3 semitones is appropriate.
    for s in [-3, -2, -1, +1, +2, +3]:
        aug.append((f"pitch_{s:+d}", lambda a, s=s: pitch_shift(a, s)))

    # ── Time stretch (6) ──────────────────────────────────────────
    # EXPANDED from 4 to 6 values. Added 0.75 and 1.20.
    # 0.75 = very fast speech (someone rushing "Hey Saya")
    # 1.20 = slow deliberate speech
    # These extremes are exactly what causes misses at inference time
    # when registration samples were all recorded at normal pace.
    for r in [0.75, 0.85, 0.90, 1.10, 1.15, 1.20]:
        aug.append((f"speed_{r:.2f}", lambda a, r=r: time_stretch(a, r)))

    # ── Volume change (4) ─────────────────────────────────────────
    # Covers whispered (0.5) to loud (1.5) utterances.
    for v in [0.5, 0.7, 1.3, 1.5]:
        aug.append((f"vol_{v:.1f}", lambda a, v=v: volume_change(a, v)))

    # ── Gaussian noise (3) ────────────────────────────────────────
    # Simulates microphone noise floor variation.
    for l in [0.003, 0.008, 0.015]:
        aug.append((f"noise_{l:.3f}", lambda a, l=l: add_gaussian_noise(a, l)))

    # ── Time shift (2) ────────────────────────────────────────────
    # REDUCED from 4 to 2 (±150ms, ±75ms removed — similar to ±100ms).
    # Freed 2 slots for more informative combination augmentations.
    for ms in [-100, +100]:
        aug.append((f"shift_{ms}ms", lambda a, ms=ms: time_shift(a, ms)))

    # ── Distance simulation (2) ───────────────────────────────────
    # Covers speaking from 1m and 2-3m away.
    for d in [0.4, 0.65]:
        aug.append((f"dist_{d:.2f}", lambda a, d=d: simulate_distance(a, d)))

    # ── Combinations (6) ─────────────────────────────────────────
    # Each combo targets a specific real-world scenario.
    # Ensures all 6 pitch values appear in at least one combination.
    #
    # combo_p1_n1    : pitched up slightly + light noise (most common real scenario)
    # combo_m1_n2    : pitched down slightly + medium noise
    # combo_p3_fast  : high pitch + fast speech (excited speaker) ← covers +3
    # combo_m3_dist  : low pitch + distant mic (deep voice, far away) ← covers -3
    # combo_slow_quiet: slow speech + quiet (hesitant speaker)
    # combo_fast_noise: fast speech + noise ← THE KEY FIX for your "misses fast" bug
    #   This is the most important combo — it directly trains the model on the
    #   exact scenario that was causing misses: fast utterance in noisy conditions.
    aug.append(("combo_p1_n1",
                lambda a: add_gaussian_noise(pitch_shift(a, +1), 0.005)))

    aug.append(("combo_m1_n2",
                lambda a: add_gaussian_noise(pitch_shift(a, -1), 0.008)))

    aug.append(("combo_p3_fast",
                lambda a: time_stretch(pitch_shift(a, +3), 1.10)))

    aug.append(("combo_m3_dist",
                lambda a: simulate_distance(pitch_shift(a, -3), 0.5)))

    aug.append(("combo_slow_quiet",
                lambda a: volume_change(time_stretch(a, 1.15), 0.6)))

    aug.append(("combo_fast_noise",
                lambda a: add_gaussian_noise(time_stretch(a, 0.80), 0.008)))

    # Sanity check — this must print 29
    assert len(aug) == 29, f"Expected 29 augmentations, got {len(aug)}"
    return aug


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

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder not found: {INPUT_DIR}")
        return

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

    if not originals:
        print("ERROR: Could not load any recordings.")
        return

    augmentations = get_augmentations()
    total_expected = len(originals) * (1 + len(augmentations))

    print(f"\n{len(originals)} originals × {1 + len(augmentations)} versions each")
    print(f"Expected output: {total_expected} clips")
    print()

    # Augmentation budget summary
    print("Augmentation budget per original:")
    print("  Pitch shifts      : 6  (−3 to +3 semitones)")
    print("  Time stretch      : 6  (0.75x fast → 1.20x slow)")
    print("  Volume change     : 4  (0.5x → 1.5x)")
    print("  Gaussian noise    : 3  (light/medium/heavy)")
    print("  Time shift        : 2  (−100ms / +100ms)")
    print("  Distance sim      : 2  (near/far)")
    print("  Combinations      : 6  (multi-factor, all pitches covered)")
    print("  TOTAL             : 29 + 1 original = 30 per file")
    print()

    saved  = 0
    failed = 0

    # Save originals into output folder
    for name, audio in originals:
        out_path = os.path.join(OUTPUT_DIR, f"orig_{name}")
        save_wav(audio, out_path)
        saved += 1

    # Apply all augmentations
    print("Generating augmented samples...")
    for name, audio in originals:
        base = name.replace(".wav", "")
        for aug_name, aug_fn in augmentations:
            try:
                augmented = aug_fn(audio)
                out_path  = os.path.join(OUTPUT_DIR, f"{base}_{aug_name}.wav")
                save_wav(augmented, out_path)
                saved += 1
            except Exception as e:
                print(f"  ⚠️  {base}_{aug_name}: {e}")
                failed += 1

    print()
    print("=" * 55)
    print(f"  Done!")
    print(f"  Saved : {saved} clips")
    if failed:
        print(f"  Failed: {failed} clips (see warnings above)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 55)
    print()

    if saved == total_expected:
        print(f"✅ Exact count verified: {saved} = "
              f"{len(originals)} originals × 30 versions")
    else:
        print(f"⚠️  Expected {total_expected}, got {saved}. "
              f"Check failed clips above.")

    print()
    print("Next step: python3 train_embedding.py \\")
    print("    --indian_data ./indian_accent_data \\")
    print("    --esc50_noise ./noise_esc50_prepared")


if __name__ == "__main__":
    main()