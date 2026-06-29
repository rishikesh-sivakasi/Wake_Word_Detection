"""
move_hindi_tamil_to_negatives.py
==================================
Flattens your existing Hindi/Tamil word-group folders (nested as
<lang>/<group>/clip_*.wav, the structure build_indian_group_map() needs)
into the flat folder load_wav_files() expects for hard negatives:

    dataset/train/negative/<lang>_<group>_<original_filename>.wav

EDIT THE TWO PATHS BELOW to match your actual setup, then run:
    python3 move_hindi_tamil_to_negatives.py

This COPIES (doesn't delete) the originals by default -- set MOVE=True
near the top if you want to relocate instead of duplicate.
"""

import os
import shutil

# ---- EDIT THESE TWO PATHS ----
SRC_ROOT = "./indian_accent_data"          # parent folder containing hindi/ and tamil/
DST_DIR  = "./dataset/train/negative"       # matches HARD_NEGATIVE_DIR in train_embedding.py
LANGS    = ["hindi", "tamil"]                # add "bengali" etc. here if you have it
MOVE     = False                             # False = copy (keeps originals), True = move
# -------------------------------

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    count = 0
    skipped_langs = []

    for lang in LANGS:
        lang_dir = os.path.join(SRC_ROOT, lang)
        if not os.path.isdir(lang_dir):
            skipped_langs.append(lang_dir)
            continue

        for group in sorted(os.listdir(lang_dir)):
            group_dir = os.path.join(lang_dir, group)
            if not os.path.isdir(group_dir):
                continue
            for f in sorted(os.listdir(group_dir)):
                if not f.lower().endswith(".wav"):
                    continue
                src_path = os.path.join(group_dir, f)
                dst_name = f"{lang}_{group}_{f}"
                dst_path = os.path.join(DST_DIR, dst_name)

                if MOVE:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                count += 1

    print(f"{'Moved' if MOVE else 'Copied'} {count} files into {DST_DIR}")
    for d in skipped_langs:
        print(f"  (skipped, not found: {d})")

    print(f"\nIMPORTANT: now remove (or move out) {SRC_ROOT}/hindi and {SRC_ROOT}/tamil")
    print(f"if they're still inside the folder you pass as --indian_data --")
    print(f"otherwise build_indian_group_map() will pick them up AGAIN for the")
    print(f"accent episode, which defeats the point of moving them to negatives.")


if __name__ == "__main__":
    main()