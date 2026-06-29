"""
prepare_esc50_noise.py
=======================
Prepares the ESC-50 environmental noise dataset for use as the
"noise prototype" training data in your wake word system.

WHAT THIS SCRIPT DOES:
  1. Reads meta/esc50.csv to get category labels for all 2,000 clips
  2. Converts each 5-second, 44.1kHz WAV → 16kHz mono
  3. Splits each 5-second clip into TWO non-overlapping 2-second chunks
     → doubles your noise data for free (2,000 clips → ~4,000 chunks)
  4. Organises chunks into category folders matching ESC-50's labels

WHY CHUNK INSTEAD OF SILENCE-DETECT:
  Unlike speech, noise has no "spoken moment" to detect — the whole
  clip IS the signal. So instead of cropping to silence boundaries
  (which made sense for Common Voice), we simply slice the 5-second
  clip into two 2-second windows. Both halves are valid noise samples.

FOLDER OUTPUT STRUCTURE:
  noise_esc50_prepared/
    ├── dog/
    │    ├── clip_001_a.wav   ← first half of original
    │    ├── clip_001_b.wav   ← second half of original
    │    └── ...
    ├── rain/
    ├── engine/
    ├── footsteps/
    └── ... (50 categories total)

HOW TO RUN:
  pip install pydub pandas tqdm
  sudo apt-get install ffmpeg     ← if not already installed

  python3 prepare_esc50_noise.py \
      --esc50  /path/to/ESC-50-master \
      --output ./noise_esc50_prepared

AFTER RUNNING:
  Point your training script and register_wake_word.py's NOISE_DIR
  at the categories most relevant to your robot's environment, or
  use the whole folder for maximum noise diversity.
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

try:
    from pydub import AudioSegment
except ImportError:
    print("ERROR: pydub not installed.")
    print("Run: pip install pydub")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
TARGET_SAMPLE_RATE = 16000   # Must match your MFCC pipeline
CHUNK_DURATION_MS  = 2000    # 2 seconds — matches your detection window


# ══════════════════════════════════════════════════════════════════
# CORE CONVERSION
# ══════════════════════════════════════════════════════════════════
def process_clip(wav_path: str, out_dir: str, base_name: str) -> int:
    """
    Converts one ESC-50 clip to 16kHz mono, then splits into
    two non-overlapping 2-second chunks.
    Returns the number of chunks successfully written (0, 1, or 2).
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)

        total_len = len(audio)
        written   = 0

        # Chunk A — first 2 seconds
        if total_len >= CHUNK_DURATION_MS:
            chunk_a = audio[:CHUNK_DURATION_MS]
            out_a   = os.path.join(out_dir, f"{base_name}_a.wav")
            chunk_a.export(out_a, format="wav")
            written += 1

        # Chunk B — second 2 seconds (only if there's enough audio left)
        if total_len >= CHUNK_DURATION_MS * 2:
            chunk_b = audio[CHUNK_DURATION_MS: CHUNK_DURATION_MS * 2]
            out_b   = os.path.join(out_dir, f"{base_name}_b.wav")
            chunk_b.export(out_b, format="wav")
            written += 1
        elif total_len > CHUNK_DURATION_MS:
            # Less than a full second chunk left — pad remainder to 2s
            remainder = audio[CHUNK_DURATION_MS:]
            pad_ms    = CHUNK_DURATION_MS - len(remainder)
            if pad_ms > 0 and len(remainder) > 500:  # only keep if >0.5s of real audio
                padded = remainder + AudioSegment.silent(
                    duration=pad_ms, frame_rate=TARGET_SAMPLE_RATE
                )
                out_b = os.path.join(out_dir, f"{base_name}_b.wav")
                padded.export(out_b, format="wav")
                written += 1

        return written

    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Prepare ESC-50 noise dataset for wake word training")
    parser.add_argument("--esc50",  required=True, help="Path to ESC-50-master folder")
    parser.add_argument("--output", default="./noise_esc50_prepared", help="Output folder")
    args = parser.parse_args()

    meta_path  = os.path.join(args.esc50, "meta", "esc50.csv")
    audio_dir  = os.path.join(args.esc50, "audio")

    if not os.path.exists(meta_path):
        print(f"ERROR: meta/esc50.csv not found at {meta_path}")
        sys.exit(1)
    if not os.path.exists(audio_dir):
        print(f"ERROR: audio/ folder not found at {audio_dir}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  ESC-50 Noise Preparation")
    print(f"  Input  : {args.esc50}")
    print(f"  Output : {args.output}")
    print(f"{'='*55}\n")

    df = pd.read_csv(meta_path)
    print(f"Loaded {len(df)} clips across {df['category'].nunique()} categories\n")

    total_written = 0
    total_failed   = 0
    category_counts = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        filename = row["filename"]
        category = row["category"]

        wav_path = os.path.join(audio_dir, filename)
        if not os.path.exists(wav_path):
            total_failed += 1
            continue

        cat_out_dir = os.path.join(args.output, category)
        os.makedirs(cat_out_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        n_written = process_clip(wav_path, cat_out_dir, base_name)

        total_written += n_written
        category_counts[category] = category_counts.get(category, 0) + n_written

        if n_written == 0:
            total_failed += 1

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PREPARATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Total chunks written : {total_written:,}")
    print(f"  Failed clips         : {total_failed}")
    print(f"  Categories           : {len(category_counts)}")
    print(f"\n  Top 10 categories by chunk count:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cat:20s} : {count}")

    print(f"\nNext steps:")
    print(f"  1. Use {args.output}/ as part of your noise training data")
    print(f"  2. For register_wake_word.py NOISE_DIR, pick a representative")
    print(f"     subset (e.g. fan, traffic, engine, footsteps, rain, chatter)")
    print(f"     rather than all 50 categories, to keep the noise prototype focused")
    print(f"     on sounds your robot will actually encounter.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
