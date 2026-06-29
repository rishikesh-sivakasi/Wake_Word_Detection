"""
check.py
===================
Scans the word clips produced by prepare_svarah.py for likely alignment
problems, so you don't have to randomly sample 35,910 files by ear.
Flags clips that are unusually short/long, near-silent, or look clipped.

This is a SCREEN, not a verdict -- a flagged clip might be totally fine
(some real words ARE short or quiet), and an unflagged clip could still
have a subtly wrong boundary. Use it to prioritize what to listen to,
not as a final correctness check.

RUN:
    python3 check.py --root ./indian_accent_data/svarah_english
"""

import os
import glob
import argparse
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                         help="e.g. ./indian_accent_data/svarah_english")
    parser.add_argument("--short_ms", type=float, default=150,
                         help="Flag clips shorter than this")
    parser.add_argument("--long_ms", type=float, default=1500,
                         help="Flag clips longer than this (likely bled into a neighbor word)")
    parser.add_argument("--silence_peak", type=float, default=0.02,
                         help="Flag clips whose peak amplitude is below this")
    parser.add_argument("--print_limit", type=int, default=50,
                         help="How many flagged clips to print to console")
    parser.add_argument("--out_file", default="svarah_qa_flagged.txt")
    args = parser.parse_args()

    word_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    flagged = []
    n_scanned = 0

    for word_dir in word_dirs:
        word = os.path.basename(word_dir)
        for wav_path in sorted(glob.glob(os.path.join(word_dir, "*.wav"))):
            try:
                audio, sr = sf.read(wav_path)
            except Exception as e:
                flagged.append((wav_path, 0, f"unreadable file: {e}"))
                continue

            n_scanned += 1
            if len(audio) == 0:
                flagged.append((wav_path, 0, "empty file"))
                continue

            dur_ms = len(audio) / sr * 1000
            peak = float(np.max(np.abs(audio)))

            reasons = []
            if dur_ms < args.short_ms:
                reasons.append(f"very short ({dur_ms:.0f}ms)")
            if dur_ms > args.long_ms:
                reasons.append(f"unusually long ({dur_ms:.0f}ms) -- may include part of a neighbor word")
            if peak < args.silence_peak:
                reasons.append(f"near-silent (peak={peak:.4f})")
            if peak >= 0.999:
                reasons.append("clipped/distorted (peak saturated)")

            if reasons:
                flagged.append((wav_path, dur_ms, "; ".join(reasons)))

    print(f"Scanned {n_scanned} clips across {len(word_dirs)} word folders")
    print(f"Flagged {len(flagged)} clips ({len(flagged) / max(n_scanned, 1) * 100:.1f}%) for manual review\n")

    for path, dur, reason in flagged[:args.print_limit]:
        print(f"  [{reason}]  {path}")
    if len(flagged) > args.print_limit:
        print(f"  ... and {len(flagged) - args.print_limit} more (see {args.out_file})")

    with open(args.out_file, "w") as f:
        for path, dur, reason in flagged:
            f.write(f"{path}\t{dur:.0f}ms\t{reason}\n")
    print(f"\nFull flagged list written to {args.out_file}")
    print(f"Listen to a random ~20-30 of these (and ~10-20 UNflagged clips too, as a sanity check)")
    print(f"before trusting the dataset for training.")


if __name__ == "__main__":
    main()