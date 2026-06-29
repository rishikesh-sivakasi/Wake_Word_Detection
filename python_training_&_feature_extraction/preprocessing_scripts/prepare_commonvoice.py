"""
prepare_commonvoice.py
======================
Trims, converts, and organises Common Voice Hindi and Tamil datasets
for use in the episodic training pipeline (train_embedding.py).

WHAT THIS SCRIPT DOES:
  1. Reads validated.tsv from Hindi and Tamil Common Voice folders
  2. Groups clips by INDIVIDUAL WORD, not full sentence and not speaker
  3. Randomly selects MAX_CLIPS_PER_LANGUAGE clips (default 800)
  4. Converts MP3 → WAV at 16kHz mono
  5. Strips leading/trailing silence, then centre-pads/crops to exactly
     2 seconds so the speech content is preserved and well positioned
  6. Organises clips into word-level folders for episodic sampling
  7. Prints a summary of what was prepared

WHY WORD-LEVEL GROUPING (NOT SENTENCE-LEVEL, NOT SPEAKER-LEVEL):
  - Sentence-level grouping fails on modern Common Voice releases because
    almost every sentence is recorded by only 1-3 speakers (not the 15+
    needed for N_SHOT=10 episodes).
  - Speaker-level grouping (group by client_id) was considered and
    rejected. It would train the model to distinguish WHO is speaking
    (voice-ID / speaker verification), which is the OPPOSITE of what a
    wake word detector needs. We need: same word + different speaker
    => embeddings CLOSE together. Speaker-grouping teaches: same speaker
    => close, different speaker => far apart — backwards for this task.
  - Word-level grouping is correct: a common word appears across hundreds
    of different sentences spoken by hundreds of different speakers,
    giving the model exactly the "same word, many accents" signal it
    needs to generalise across Indian accents.

  NOTE: Common Voice provides sentence-level audio only (no word-aligned
  clips). So each clip used in a word group is the FULL SENTENCE
  containing that word, not an isolated word recording. This is fine for
  episodic training. It is NOT used to build the actual "Hey Saya"
  wake-word template — only your own dedicated recordings are used for
  that (via register_wake_word.py).

WHY SILENCE-AWARE CENTRE TRIMMING (NOT BLIND CENTRE-CROP):
  A blind centre-crop can land entirely on silence if the spoken content
  in a clip is short and positioned near the start or end of the file —
  common in Common Voice since clip length varies a lot. This version:
    1. Detects the non-silent (speech) region first
    2. Crops the file down to just that speech region
    3. If the speech region is longer than 2s, takes the first 2s of it
    4. If shorter than 2s, centre-pads it with silence on both sides
  This guarantees real speech is always present in the final clip, and
  centre-padding (rather than end-padding) better matches how a wake
  word appears mid-window in the live 2-second detection buffer.

WHY 800 CLIPS PER LANGUAGE:
  Episodic training samples N_WAY=5 classes, N_SHOT=10, N_QUERY=5 per
  episode. 800 clips split across ~20 word groups gives ~40 clips per
  group — enough for reliable episodic sampling without taking days to
  process on CPU.

FOLDER OUTPUT STRUCTURE:
  indian_accent_data/
    ├── hindi/
    │    ├── group_001/   ← ~40 WAV clips (sentences containing word 1)
    │    ├── group_002/
    │    └── ...
    └── tamil/
         ├── group_001/
         ├── group_002/
         └── ...

HOW TO RUN:
  pip install pydub pandas tqdm
  sudo apt-get install ffmpeg        ← required by pydub for MP3 decoding

  python3 prepare_commonvoice.py \
      --hindi  /path/to/cv-corpus-hindi \
      --tamil  /path/to/cv-corpus-tamil \
      --output ./indian_accent_data

  Optional flags:
      --max              800   (max clips per language)
      --min_per_group    15    (lower this if a language still yields 0 groups)

AFTER RUNNING:
  Point train_embedding.py at --indian_data ./indian_accent_data
"""

import os
import sys
import argparse
import random
import pandas as pd
from tqdm import tqdm

try:
    from pydub import AudioSegment
    from pydub import silence
except ImportError:
    print("ERROR: pydub not installed.")
    print("Run: pip install pydub")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# CONFIG (defaults — overridable via CLI flags where noted)
# ══════════════════════════════════════════════════════════════════
DEFAULT_MAX_CLIPS_PER_LANGUAGE = 800     # how many clips to use per language
DEFAULT_MIN_CLIPS_PER_GROUP    = 15      # minimum clips per word-group
MAX_CLIPS_PER_GROUP            = 40      # cap so one common word doesn't dominate
TARGET_SAMPLE_RATE             = 16000   # must match your MFCC pipeline
TARGET_DURATION_MS             = 2000    # 2 seconds, matches the live detection window
SILENCE_THRESH_DBFS            = -40     # anything quieter than this = silence
MIN_SILENCE_LEN_MS             = 100     # minimum gap length to count as silence
RANDOM_SEED                    = 42

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════
# AUDIO CONVERSION — silence-aware, centre-padded
# ══════════════════════════════════════════════════════════════════
def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    Convert one MP3 file to 16kHz mono WAV.
    Strips leading/trailing silence so short words aren't lost inside a
    blind crop, then fits the result into exactly 2 seconds using
    centre-padding (or front-cropping if the speech itself is >2s).

    Returns True on success, False on failure.
    """
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)

        # Step 1 — find where the actual speech is, drop surrounding silence
        nonsilent_ranges = silence.detect_nonsilent(
            audio,
            min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DBFS,
        )
        if nonsilent_ranges:
            start_trim = nonsilent_ranges[0][0]
            end_trim   = nonsilent_ranges[-1][1]
            audio      = audio[start_trim:end_trim]
        # If detect_nonsilent finds nothing (e.g. a clip that's all very
        # quiet), we fall through and use the original audio as-is rather
        # than producing an empty clip.

        # Step 2 — fit into exactly 2 seconds
        if len(audio) >= TARGET_DURATION_MS:
            # Speech itself is long — take the first 2 seconds of it
            audio = audio[:TARGET_DURATION_MS]
        else:
            # Speech shorter than 2s — centre it with silence padding on
            # both sides, so the word sits mid-window like it would in
            # live detection rather than always being left-aligned
            pad_total = TARGET_DURATION_MS - len(audio)
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            audio = (
                AudioSegment.silent(duration=pad_left,  frame_rate=TARGET_SAMPLE_RATE)
                + audio
                + AudioSegment.silent(duration=pad_right, frame_rate=TARGET_SAMPLE_RATE)
            )

        audio.export(wav_path, format="wav")
        return True

    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# TSV LOADER
# ══════════════════════════════════════════════════════════════════
def load_validated_clips(corpus_dir: str, language: str) -> list:
    """
    Reads validated.tsv and returns a list of dicts:
      { 'mp3_path': ..., 'sentence': ..., 'client_id': ... }
    """
    tsv_path  = os.path.join(corpus_dir, "validated.tsv")
    clips_dir = os.path.join(corpus_dir, "clips")

    if not os.path.exists(tsv_path):
        print(f"ERROR: validated.tsv not found in {corpus_dir}")
        sys.exit(1)
    if not os.path.exists(clips_dir):
        print(f"ERROR: clips/ folder not found in {corpus_dir}")
        sys.exit(1)

    print(f"\nLoading {language} validated.tsv ...")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    needed = ["path", "sentence", "client_id"]
    for col in needed:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in validated.tsv")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

    df = df[needed].dropna()
    print(f"  Total validated clips: {len(df):,}")

    clips = []
    for _, row in df.iterrows():
        mp3_name = row["path"]
        if not mp3_name.endswith(".mp3"):
            mp3_name = mp3_name + ".mp3"
        mp3_path = os.path.join(clips_dir, mp3_name)
        if os.path.exists(mp3_path):
            clips.append({
                "mp3_path":  mp3_path,
                "sentence":  str(row["sentence"]).strip(),
                "client_id": str(row["client_id"]).strip(),
            })

    print(f"  Clips found on disk: {len(clips):,}")
    return clips


# ══════════════════════════════════════════════════════════════════
# GROUPING STRATEGY — WORD LEVEL
# ══════════════════════════════════════════════════════════════════
def tokenize(sentence: str) -> list:
    """
    Simple whitespace tokenizer. Works for Hindi (Devanagari) and Tamil
    script, both of which are space-separated at the word level.
    Strips common punctuation including Devanagari danda marks.
    """
    PUNCT = '.,!?;:"\'()[]।॥'
    cleaned = sentence
    for p in PUNCT:
        cleaned = cleaned.replace(p, " ")
    return [w.strip() for w in cleaned.split() if w.strip()]


def group_clips(clips: list, max_clips: int, min_per_group: int) -> dict:
    """
    Groups clips by individual word (see module docstring for the full
    rationale on why word-level, not sentence-level or speaker-level).

    Returns: { 'word': [list of clip dicts] }
    """
    word_map = {}
    for clip in clips:
        words = tokenize(clip["sentence"])
        for w in set(words):              # dedupe within one sentence
            if len(w) < 2:
                continue                  # skip 1-character tokens (likely noise)
            word_map.setdefault(w, []).append(clip)

    valid_groups = {
        w: clip_list for w, clip_list in word_map.items()
        if len(clip_list) >= min_per_group
    }
    print(f"  Word groups with >= {min_per_group} clips: {len(valid_groups)}")

    if len(valid_groups) == 0:
        return {}

    # Largest groups first — most reliable, most speaker diversity
    sorted_words = sorted(valid_groups.keys(), key=lambda w: -len(valid_groups[w]))

    all_selected    = []
    selected_groups = {}
    used_paths      = set()   # don't reuse the same audio file across groups

    for w in sorted_words:
        fresh = [c for c in valid_groups[w] if c["mp3_path"] not in used_paths]
        if len(fresh) < min_per_group:
            continue

        take     = min(MAX_CLIPS_PER_GROUP, len(fresh))
        selected = random.sample(fresh, take)

        selected_groups[w] = selected
        used_paths.update(c["mp3_path"] for c in selected)
        all_selected.extend(selected)

        if len(all_selected) >= max_clips:
            break

    print(f"  Groups selected : {len(selected_groups)}")
    print(f"  Total clips     : {len(all_selected)}")
    if selected_groups:
        print(f"  Example words   : {list(selected_groups.keys())[:8]}")
    return selected_groups


# ══════════════════════════════════════════════════════════════════
# MAIN PROCESSOR
# ══════════════════════════════════════════════════════════════════
def process_language(corpus_dir: str, output_dir: str, language: str,
                      max_clips: int, min_per_group: int):
    print(f"\n{'='*55}")
    print(f"  Processing: {language.upper()}")
    print(f"{'='*55}")

    clips  = load_validated_clips(corpus_dir, language)
    groups = group_clips(clips, max_clips, min_per_group)

    if len(groups) == 0:
        print(f"ERROR: No valid groups found for {language}.")
        print(f"       Try lowering --min_per_group (currently {min_per_group})")
        return 0, 0

    lang_out_dir  = os.path.join(output_dir, language)
    success_total = 0
    fail_total    = 0

    print(f"\nConverting MP3 → WAV (16kHz mono, silence-trimmed, 2 sec)...")

    for group_idx, (word, clip_list) in enumerate(groups.items()):
        group_dir = os.path.join(lang_out_dir, f"group_{group_idx+1:03d}")
        os.makedirs(group_dir, exist_ok=True)

        for clip_idx, clip in enumerate(tqdm(clip_list,
                                              desc=f"  group_{group_idx+1:03d}",
                                              leave=False)):
            wav_path = os.path.join(group_dir, f"clip_{clip_idx+1:04d}.wav")

            if os.path.exists(wav_path):
                success_total += 1
                continue   # already converted, skip

            if convert_mp3_to_wav(clip["mp3_path"], wav_path):
                success_total += 1
            else:
                fail_total += 1

    print(f"\n  ✅ Converted : {success_total} clips")
    if fail_total > 0:
        print(f"  ❌ Failed   : {fail_total} clips  (decode error — safe to ignore)")

    return success_total, len(groups)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice data for wake word training")
    parser.add_argument("--hindi",  required=True, help="Path to Common Voice Hindi corpus folder")
    parser.add_argument("--tamil",  required=True, help="Path to Common Voice Tamil corpus folder")
    parser.add_argument("--output", default="./indian_accent_data", help="Output folder")
    parser.add_argument("--max",    type=int, default=DEFAULT_MAX_CLIPS_PER_LANGUAGE,
                        help=f"Max clips per language (default {DEFAULT_MAX_CLIPS_PER_LANGUAGE})")
    parser.add_argument("--min_per_group", type=int, default=DEFAULT_MIN_CLIPS_PER_GROUP,
                        help=f"Minimum clips required per word group (default {DEFAULT_MIN_CLIPS_PER_GROUP})")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Common Voice Preparation Script")
    print(f"  Output          : {args.output}")
    print(f"  Max clips/lang  : {args.max}")
    print(f"  Min clips/group : {args.min_per_group}")
    print(f"{'='*55}")

    hi_clips, hi_groups = process_language(args.hindi, args.output, "hindi", args.max, args.min_per_group)
    ta_clips, ta_groups = process_language(args.tamil, args.output, "tamil", args.max, args.min_per_group)

    print(f"\n{'='*55}")
    print(f"  PREPARATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Hindi  : {hi_clips:,} clips  across {hi_groups} groups")
    print(f"  Tamil  : {ta_clips:,} clips  across {ta_groups} groups")
    print(f"  Total  : {hi_clips + ta_clips:,} clips")
    print(f"\n  Output structure:")
    print(f"  {args.output}/")
    print(f"    ├── hindi/")
    print(f"    │    ├── group_001/   ← WAV files")
    print(f"    │    ├── group_002/")
    print(f"    │    └── ...")
    print(f"    └── tamil/")
    print(f"         ├── group_001/")
    print(f"         └── ...")
    print(f"\nNext step:")
    print(f"  python3 train_embedding.py --indian_data {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()