"""
remove_long_outliers.py
========================
Deletes clips flagged as "unusually long" by qa_check_svarah.py.
A real single English word doesn't take 1.5+ seconds at any normal
speaking pace -- these durations alone are enough evidence of an
alignment failure (boundary bled into a neighboring word or a pause),
no listening needed before removing them.

Leaves "near-silent" and "clipped/distorted" flags alone -- those need
an actual listen, since both can be legitimate real speech.

RUN (dry run first, shows what WOULD be deleted without deleting):
    python3 remove_long_outliers.py --flagged_file svarah_qa_flagged.txt --dry_run

Then for real:
    python3 remove_long_outliers.py --flagged_file svarah_qa_flagged.txt
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged_file", default="svarah_qa_flagged.txt")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    n_deleted = 0
    n_missing = 0

    with open(args.flagged_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            path, dur, reason = parts
            if "unusually long" not in reason:
                continue

            if not os.path.exists(path):
                n_missing += 1
                continue

            if args.dry_run:
                print(f"would delete: {path}  ({dur})")
            else:
                os.remove(path)
            n_deleted += 1

    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{verb} {n_deleted} long-outlier clips.")
    if n_missing:
        print(f"({n_missing} listed files were already missing -- fine, just means "
              f"they were cleaned up already or paths moved.)")


if __name__ == "__main__":
    main()