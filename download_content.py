#!/usr/bin/env python3
"""
download_content.py - Automated asset downloader for Multilingual KWS

Downloads and extracts:
  1. Multilingual Embedding Model (multilingual_context_73_0.8011)
  2. Unknown Words Dataset (unknown_files.tar.gz)
  3. Google Speech Commands v0.02 (_background_noise_)
  4. (Optional) Multilingual Spoken Words Corpus microset (mswc_microset)

Usage:
  python download_content.py [--target-dir content] [--include-mswc]
"""

import os
import sys
import argparse
import tarfile
import urllib.request
from pathlib import Path


ASSETS = {
    "embedding_model": {
        "name": "Pretrained Embedding Model (multilingual_context_73_0.8011)",
        "url": "https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz",
        "archive_name": "multilingual_context_73_0.8011.tar.gz",
        "extract_to": "",  # extracts directly to root of target_dir
        "check_path": "multilingual_context_73_0.8011/saved_model.pb",
        "wildcard": None,
    },
    "unknown_files": {
        "name": "Unknown Words Dataset",
        "url": "https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz",
        "archive_name": "unknown_files.tar.gz",
        "extract_to": "unknown_files",
        "check_path": "unknown_files/unknown_words",
        "wildcard": None,
    },
    "background_noise": {
        "name": "Google Speech Commands v0.02 (_background_noise_)",
        "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "archive_name": "speech_commands_v0.02.tar.gz",
        "extract_to": "speech_commands",
        "check_path": "speech_commands/_background_noise_",
        "filter_prefix": "_background_noise_",
    },
    "mswc_microset": {
        "name": "Multilingual Spoken Words Corpus (MSWC Microset)",
        "url": "https://mswc.mlcommons-storage.org/mswc_microset.tar.gz",
        "archive_name": "mswc_microset.tar.gz",
        "extract_to": "mswc_microset",
        "check_path": "mswc_microset",
        "wildcard": None,
        "optional": True,
    },
}


class DownloadProgressBar:
    def __init__(self, description: str):
        self.description = description
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            downloaded = block_num * block_size
            sys.stdout.write(f"\r  Downloading {self.description}: {downloaded / (1024 * 1024):.1f} MB")
            sys.stdout.flush()
            return

        percent = int(block_num * block_size * 100 / total_size)
        percent = min(percent, 100)
        if percent != self.last_percent:
            self.last_percent = percent
            bar_len = 30
            filled_len = int(bar_len * percent // 100)
            bar = "=" * filled_len + "-" * (bar_len - filled_len)
            downloaded_mb = (block_num * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{bar}] {percent:3d}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()


def download_and_extract(key: str, info: dict, target_dir: Path):
    dest_dir = target_dir / info["extract_to"] if info["extract_to"] else target_dir
    check_target = target_dir / info["check_path"]

    if check_target.exists():
        print(f"✓ {info['name']} already exists at: {check_target}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / info["archive_name"]

    print(f"\n[-] Fetching {info['name']}...")
    print(f"    Source URL: {info['url']}")

    progress_bar = DownloadProgressBar(info["archive_name"])
    try:
        urllib.request.urlretrieve(info["url"], archive_path, reporthook=progress_bar)
        print()  # newline after progress bar
    except Exception as e:
        print(f"\n❌ Error downloading {info['url']}: {e}")
        if archive_path.exists():
            archive_path.unlink()
        sys.exit(1)

    print(f"[-] Extracting {archive_path.name} to {dest_dir}...")
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            filter_prefix = info.get("filter_prefix")
            if filter_prefix:
                # Extract only members matching prefix (saves gigabytes of unneeded files)
                members = [m for m in tar.getmembers() if m.name.startswith(filter_prefix)]
                tar.extractall(path=dest_dir, members=members)
            else:
                tar.extractall(path=dest_dir)
        print(f"✓ Extracted successfully.")
    except Exception as e:
        print(f"❌ Error extracting {archive_path}: {e}")
        sys.exit(1)
    finally:
        if archive_path.exists():
            archive_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download and configure assets for Multilingual KWS.")
    parser.add_argument(
        "--target-dir",
        default="content",
        help="Target folder to install dataset assets (default: 'content')",
    )
    parser.add_argument(
        "--include-mswc",
        action="store_true",
        help="Also download the optional MSWC microset archive (~612 MB)",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    print("=" * 65)
    print(" Multilingual Keyword Spotting (KWS) Content Asset Downloader")
    print(f" Destination Directory: {target_dir}")
    print("=" * 65)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Core required assets
    core_keys = ["embedding_model", "unknown_files", "background_noise"]
    if args.include_mswc:
        core_keys.append("mswc_microset")

    for key in core_keys:
        download_and_extract(key, ASSETS[key], target_dir)

    print("\n" + "=" * 65)
    print("✅ All required assets are verified and ready in:")
    print(f"   {target_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()

