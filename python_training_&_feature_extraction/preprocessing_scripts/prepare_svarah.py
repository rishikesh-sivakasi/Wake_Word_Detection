"""
prepare_svarah.py
==================
Downloads AI4Bharat's Svarah (Indian-accented English speech) from
HuggingFace, force-aligns each utterance's transcript to its audio using
torchaudio's MMS forced aligner, and writes out individual WORD-LEVEL clips
in exactly the folder structure build_indian_group_map() expects:

    <out_dir>/svarah_english/<word>/clip_XXXXXX.wav

Why this is needed at all: Svarah ships as full SENTENCES with a
transcript, not isolated words like Speech Commands. train_embedding.py's
"Indian accent episode" needs many DIFFERENT SPEAKERS saying the SAME WORD,
grouped into folders. Forced alignment is what extracts individual word
boundaries out of sentence-level audio so we can build those folders.

WHAT THIS DATA ACTUALLY GIVES YOU: Indian speakers reading English
sentences in their natural accent. This is the right input for teaching
"same English word, different Indian accent -> close embeddings" --
which is what should transfer to "Hey Saya". It is NOT a Hindi/Tamil
native-language dataset (Svarah's "primary_language"/"native_place_state"
fields describe the SPEAKER's background, not the language being spoken --
they speak English throughout).

CAVEATS, READ BEFORE RUNNING:
  - Svarah only ships a single "test" split (6,656 utterances, ~9.6 hours,
    117 speakers). There's no train/val split provided -- this script just
    extracts words; YOU still need to hold out some words/speakers for
    validation the same way the rest of the pipeline does.
  - Forced alignment is not perfect. Spot-check a handful of output clips
    by ear before trusting the whole batch -- play clip_000010.wav and
    confirm it actually contains the word the folder is named after.
  - Common short function words (a, the, is, to...) will dominate clip
    counts. --max_words_per_class caps this so a handful of words don't
    drown out everything else; you may still want to manually thin out
    folders for words you don't care about (Speech Commands-style content
    words generalize better than articles/prepositions).
  - This downloads a real model checkpoint (MMS forced aligner, a few
    hundred MB) and the dataset itself (~1GB) on first run -- needs an
    unrestricted internet connection, run it on your own machine, not in
    a locked-down sandbox.

INSTALL:
    pip install torch torchaudio datasets soundfile --break-system-packages

RUN:
    python3 prepare_svarah.py --out_dir ./indian_accent_data
    # then: python3 train_embedding.py --indian_data ./indian_accent_data
"""

import os
import re
import argparse

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset

SAMPLE_RATE = 16000
LANG_FOLDER_NAME = "svarah_english"  # sibling of any existing hindi/ tamil/ folders


def normalize_text(text):
    """Lowercase, strip to the character set the MMS aligner supports
    (a-z and apostrophe), split into words."""
    text = re.sub(r"[^a-zA-Z' ]", " ", text.lower())
    return [w for w in text.split() if w]


def main():
    parser = argparse.ArgumentParser(description="Prepare Svarah for the Indian accent episode")
    parser.add_argument("--out_dir", required=True,
                         help="Output root. Pass this same path as --indian_data "
                              "to train_embedding.py.")
    parser.add_argument("--min_word_len_ms", type=int, default=200,
                         help="Skip aligned words shorter than this -- usually alignment noise.")
    parser.add_argument("--max_words_per_class", type=int, default=60,
                         help="Cap clips per word so common function words don't dominate.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N utterances (useful for a quick test run).")
    args = parser.parse_args()

    lang_dir = os.path.join(args.out_dir, LANG_FOLDER_NAME)
    os.makedirs(lang_dir, exist_ok=True)

    print("Loading Svarah from HuggingFace (downloads ~1GB on first run)...")
    dataset = load_dataset("ai4bharat/Svarah", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset)} utterances.")

    print("Loading MMS forced-alignment model (torchaudio)...")
    bundle = torchaudio.pipelines.MMS_FA
    align_model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model = align_model.to(device)

    word_counts = {}
    n_clips = 0
    n_failed = 0

    for i, sample in enumerate(dataset):
        audio = sample["audio_filepath"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.unsqueeze(0)  # (1, num_samples)

        words = normalize_text(sample["text"])
        if not words:
            continue

        try:
            tokens = tokenizer(words)
            with torch.inference_mode():
                emission, _ = align_model(wav.to(device))
                token_spans = aligner(emission[0], tokens)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  [{i}] alignment failed, skipping: {e}")
            continue

        # Convert emission-frame indices back to raw sample indices
        ratio = wav.shape[1] / emission.shape[1]

        for word, spans in zip(words, token_spans):
            if not spans:
                continue
            if word_counts.get(word, 0) >= args.max_words_per_class:
                continue

            start = int(spans[0].start * ratio)
            end = int(spans[-1].end * ratio)
            dur_ms = (end - start) / SAMPLE_RATE * 1000
            if dur_ms < args.min_word_len_ms:
                continue

            # # --- START NEW PADDING LOGIC ---
            # TARGET_SAMPLES = SAMPLE_RATE * 2  # 32,000 samples (2 seconds)
            # raw_clip = wav[0, start:end]
            
            # if raw_clip.shape[0] < TARGET_SAMPLES:
            #     # Pad with silence to reach exactly 2 seconds
            #     pad_total = TARGET_SAMPLES - raw_clip.shape[0]
            #     pad_left = pad_total // 2
            #     pad_right = pad_total - pad_left
            #     clip = torch.nn.functional.pad(raw_clip, (pad_left, pad_right)).numpy()
            # else:
            #     # Truncate if somehow longer than 2 seconds
            #     clip = raw_clip[:TARGET_SAMPLES].numpy()
            # # --- END NEW PADDING LOGIC ---

            clip = wav[0, start:end].numpy()
            
            word_dir = os.path.join(lang_dir, word)
            os.makedirs(word_dir, exist_ok=True)
            out_path = os.path.join(word_dir, f"clip_{n_clips:06d}.wav")
            sf.write(out_path, clip, SAMPLE_RATE)

            word_counts[word] = word_counts.get(word, 0) + 1
            n_clips += 1

        if i % 200 == 0:
            print(f"  Processed {i}/{len(dataset)} utterances, "
                  f"{n_clips} word clips so far, {n_failed} alignment failures")

    qualifying = {w: c for w, c in word_counts.items() if c >= 15}
    print(f"\nDone. Wrote {n_clips} word clips across {len(word_counts)} distinct words "
          f"to {lang_dir}")
    print(f"{len(qualifying)} words have >= 15 clips "
          f"(train_embedding.py's INDIAN_MIN_GROUP_SIZE) and will actually be used.")
    print("\nTop 10 most common words:")
    for w, c in sorted(word_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {w}: {c}")
    print(f"\nSpot-check a few clips before training, e.g.:")
    print(f"  play {lang_dir}/<some_word>/clip_000010.wav")


if __name__ == "__main__":
    main()