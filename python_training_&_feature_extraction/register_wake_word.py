"""
register_wake_word.py
=====================
Creates a wake word template bank from audio samples.

HOW IT WORKS:
  - Loads all WAV files from hey_saya_samples/ and noise_samples/
  - Computes an embedding for each sample using the trained model
  - Averages all Hey Saya embeddings → one prototype vector (L2-normalized)
  - Averages all Noise embeddings    → one prototype vector (L2-normalized)
  - Saves both as a (2, 128) matrix → templates.npy
    Row 0 = Hey Saya prototype
    Row 1 = Noise prototype

WHY PROTOTYPE (AVERAGED) INSTEAD OF ALL 10 RAW EMBEDDINGS:
  The embedding model was trained with Prototypical Network episodic
  learning. Its inference logic IS "compute prototype → compare".
  Averaging the 10 registration embeddings before saving is consistent
  with exactly what the model was trained to do at query time.
  Storing 10 raw embeddings and doing top-K at runtime is a different
  (and slightly inconsistent) inference strategy for this model type.

CHANGING THE WAKE WORD LATER:
  Record 10 new samples of the new word → run this script again.
  The embedding model NEVER needs to be retrained.
  Only templates.npy changes.

HOW TO RUN:
  1. Put your 10 Hey Saya WAV files into: ./hey_saya_samples/
  2. Put your noise/background WAV files into: ./noise_samples/
  3. Run: python3 register_wake_word.py
"""

import os
import sys
import numpy as np
import librosa
import librosa.filters
import torch
from embedding_model import WakeWordEmbedder

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
SAYA_DIR       = "./hey_saya_samples"
NOISE_DIR      = "./noise_samples"
MODEL_PATH     = "embedding_model.pth"
TEMPLATES_SAVE = "templates.npy"      # shape: (2, 128) — Row 0=Saya, Row 1=Noise

SAMPLE_RATE    = 16000
N_FFT          = 512
HOP_LENGTH     = 160
N_MELS         = 40
N_MFCC         = 40
MAX_FRAMES     = 196
EMBEDDING_DIM  = 128

# ══════════════════════════════════════════════════════════════════
# PASTE YOUR VALUES FROM global_stats.txt (output of train_embedding.py)
# ══════════════════════════════════════════════════════════════════
GLOBAL_MEAN = -1.9617609977722168
GLOBAL_STD = 12.306929588317871

DEVICE = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
# MFCC EXTRACTION — exactly matches mfcc.cpp
# ══════════════════════════════════════════════════════════════════
_mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32)
_n           = np.arange(N_MELS)
_k           = np.arange(N_MFCC)[:, np.newaxis]
_dct         = np.cos(np.pi / N_MELS * (_n + 0.5) * _k)
_dct[0]     *= 1.0 / np.sqrt(2)
_dct        *= np.sqrt(2.0 / N_MELS)
_dct         = _dct.astype(np.float32)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Exact same pipeline as mfcc.cpp"""
    # Pre-emphasis
    emphasized = np.copy(audio)
    for i in range(len(audio) - 1, 0, -1):
        emphasized[i] = audio[i] - 0.97 * audio[i - 1]

    # AGC
    max_val = np.max(np.abs(emphasized))
    if max_val > 0.15:
        gain = np.clip(0.80 / max_val, 1.0, 8.0)
        emphasized = np.clip(emphasized * gain, -1.0, 1.0)

    S_power = np.abs(librosa.stft(emphasized, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   window='hamming', center=False)) ** 2.0
    mel_S   = _mel_filters @ S_power
    log_mel = np.log(mel_S + 1e-6)
    mfcc    = _dct @ log_mel

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════
def load_model():
    model      = WakeWordEmbedder(embedding_dim=EMBEDDING_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  ✅ Model loaded from {MODEL_PATH}")
    return model


# ══════════════════════════════════════════════════════════════════
# EMBEDDING HELPER
# ══════════════════════════════════════════════════════════════════
def get_embedding(model, wav_path: str) -> np.ndarray:
    """Returns L2-normalized 128-dim embedding for one WAV file."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    target_len = SAMPLE_RATE * 2
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc   = extract_mfcc(audio)
    mfcc   = (mfcc - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)   # (1,1,40,196)
    with torch.no_grad():
        emb = model(tensor)   # (1,128) — already L2-normalized inside model
    return emb.squeeze().numpy()   # (128,)


# ══════════════════════════════════════════════════════════════════
# BUILD PROTOTYPE
# Computes individual embeddings for all WAVs in a folder,
# averages them, then re-normalizes → one prototype vector.
# This is consistent with how Prototypical Networks work at inference.
# ══════════════════════════════════════════════════════════════════
def build_prototype(model, folder: str, label: str) -> tuple[np.ndarray, list]:
    """
    Returns:
      prototype  : (128,) L2-normalized mean embedding
      embeddings : list of individual (128,) embeddings (kept for quality check)
    """
    wav_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"  ERROR: No WAV files found in {folder}")
        sys.exit(1)

    print(f"\n  [{label}] Found {len(wav_files)} samples:")
    embeddings = []
    for wav_path in wav_files:
        try:
            emb = get_embedding(model, wav_path)
            embeddings.append(emb)
            print(f"    ✅ {os.path.basename(wav_path)}")
        except Exception as e:
            print(f"    ❌ {os.path.basename(wav_path)} — {e}")

    if len(embeddings) == 0:
        print(f"  ERROR: All {label} files failed.")
        sys.exit(1)

    # Average all embeddings → prototype, then L2-normalize
    # This is the exact operation Prototypical Networks use at test time
    mean_emb   = np.mean(embeddings, axis=0)
    norm       = np.linalg.norm(mean_emb) + 1e-8
    prototype  = (mean_emb / norm).astype(np.float32)

    return prototype, embeddings


# ══════════════════════════════════════════════════════════════════
# QUALITY CHECK — pairwise cosine similarity matrix
# All samples of the same word should score > 0.60 against each other.
# A low-scoring sample means that recording is inconsistent — re-record it.
# ══════════════════════════════════════════════════════════════════
def quality_check(embeddings: list, label: str):
    n = len(embeddings)
    if n < 2:
        print(f"  (Need at least 2 samples for quality check)")
        return

    print(f"\n  [{label}] Pairwise similarity matrix (aim for > 0.60):\n")
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = float(np.dot(embeddings[i], embeddings[j]))

    # Header
    print("       " + "  ".join([f"S{i+1:02d}" for i in range(n)]))
    for i in range(n):
        row = f"  S{i+1:02d}  "
        for j in range(n):
            row += "   -- " if i == j else f"  {sim[i,j]:.2f}"
        print(row)

    # Per-sample average
    print(f"\n  Per-sample average similarity:")
    outliers = []
    for i in range(n):
        others = [sim[i, j] for j in range(n) if j != i]
        avg    = np.mean(others)
        bar    = "█" * int(avg * 20)
        flag   = ""
        if avg < 0.50:
            flag = "  ⚠️  LOW — consider re-recording"
            outliers.append(i + 1)
        print(f"    Sample {i+1:02d}: {avg:.4f}  {bar}{flag}")

    overall = np.mean([sim[i, j] for i in range(n) for j in range(n) if i != j])
    print(f"\n  Overall average pairwise similarity: {overall:.4f}", end="  ")
    if   overall < 0.50: print("⚠️  Low — try recording more consistently.")
    elif overall < 0.70: print("✅  Acceptable.")
    else:                print("✅  Excellent consistency.")

    if outliers:
        print(f"\n  ⚠️  Samples {outliers} scored low — re-record and run again.")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    for folder, label in [(SAYA_DIR, "Hey Saya"), (NOISE_DIR, "Noise")]:
        if not os.path.exists(folder):
            print(f"ERROR: Folder not found: {folder}")
            sys.exit(1)

    print("Loading model...")
    model = load_model()

    # ── Build prototypes ──────────────────────────────────────────
    saya_proto,  saya_embs  = build_prototype(model, SAYA_DIR,  "Hey Saya")
    noise_proto, noise_embs = build_prototype(model, NOISE_DIR, "Noise")

    # ── Save (2, 128) matrix — Row 0 = Saya, Row 1 = Noise ────────
    template_matrix = np.vstack([saya_proto, noise_proto]).astype(np.float32)
    np.save(TEMPLATES_SAVE, template_matrix)

    print(f"\n{'='*55}")
    print(f"  ✅  Wake word registered!")
    print(f"  Template shape : {template_matrix.shape}  (Row 0=Saya, Row 1=Noise)")
    print(f"  Saved to       : {TEMPLATES_SAVE}")
    print(f"{'='*55}")

    # ── Quality checks ────────────────────────────────────────────
    quality_check(saya_embs,  "Hey Saya")
    quality_check(noise_embs, "Noise")

    # ── Separation check — how far apart are the two prototypes? ──
    separation = float(np.dot(saya_proto, noise_proto))
    print(f"\n  Saya ↔ Noise prototype similarity: {separation:.4f}")
    if separation > 0.40:
        print("  ⚠️  Prototypes are too close — noise samples may sound too similar")
        print("     to Hey Saya. Add more varied background noise and re-register.")
    elif separation > 0.20:
        print("  ✅  Acceptable separation.")
    else:
        print("  ✅  Excellent separation — noise filter will be effective.")

    print(f"\nNext steps:")
    print(f"  1. Copy {TEMPLATES_SAVE} to your Raspberry Pi")
    print(f"  2. Run: ./wake_word_detector {TEMPLATES_SAVE} 0.75")


if __name__ == "__main__":
    main()