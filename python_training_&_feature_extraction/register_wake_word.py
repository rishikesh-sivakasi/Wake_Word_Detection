"""
register_wake_word.py
=====================
Creates a wake word template bank from 10 audio samples.

WHAT CHANGED FROM PREVIOUS VERSION:
  OLD: average all 10 embeddings into 1 vector → template.npy  (shape: 128,)
  NEW: store ALL 10 embeddings as a matrix    → templates.npy  (shape: N, 128)

WHY THIS IS BETTER:
  Averaging loses information. If you say "Hey Saya" fast sometimes
  and slow other times, the average sits between both and matches neither
  perfectly. Keeping all embeddings means at least one will always
  closely match however you say it in real life.

HOW TO RUN:
  1. Put your 10 WAV files into: ./hey_saya_samples/
  2. Run: python3 register_wake_word.py

CHANGING THE WAKE WORD LATER:
  Record 10 new samples of the new word → run this script again.
  The embedding model never changes. Only templates.npy changes.
"""

import os
import sys
import numpy as np
import librosa
import torch

from embedding_model import WakeWordEmbedder

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
SAMPLES_DIR    = "./hey_saya_samples"
MODEL_PATH     = "embedding_model.pth"
TEMPLATES_SAVE = "templates.npy"       # shape: (N_samples, 128)

SAMPLE_RATE    = 16000
N_FFT          = 512
HOP_LENGTH     = 160
N_MELS         = 40
N_MFCC         = 40
MAX_FRAMES     = 196
EMBEDDING_DIM  = 128

# Must match TOP_K in wake_word_detector.cpp
TOP_K = 3

# ══════════════════════════════════════════════════════════════════
# PASTE YOUR VALUES FROM global_stats.txt (from train_embedding.py)
# ══════════════════════════════════════════════════════════════════
GLOBAL_MEAN = -1.9461231231689453  # ← replace after training
GLOBAL_STD  = 12.231356620788574  # ← replace after training

DEVICE = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
# MFCC EXTRACTION — exactly matches mfcc.cpp + dataset_extractor.cpp
# ══════════════════════════════════════════════════════════════════
import librosa.filters

mel_filters_matrix = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32
)
_n = np.arange(N_MELS)
_k = np.arange(N_MFCC)[:, np.newaxis]
dct_matrix = np.cos(np.pi / N_MELS * (_n + 0.5) * _k)
dct_matrix[0] *= 1.0 / np.sqrt(2)
dct_matrix *= np.sqrt(2.0 / N_MELS)
dct_matrix = dct_matrix.astype(np.float32)


def apply_pre_emphasis(audio, coeff=0.97):
    emphasized = np.copy(audio)
    for i in range(len(audio) - 1, 0, -1):
        emphasized[i] = audio[i] - coeff * audio[i - 1]
    return emphasized


def apply_agc(audio, threshold=0.15, target=0.80, max_gain=8.0):
    max_val = np.max(np.abs(audio))
    if max_val > threshold:
        gain = np.clip(target / max_val, 1.0, max_gain)
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio


def extract_mfcc_cpp_match(audio: np.ndarray) -> np.ndarray:
    """Exact same pipeline as dataset_extractor.cpp"""
    audio   = apply_pre_emphasis(audio)
    audio   = apply_agc(audio)
    S_power = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   window='hamming', center=False)) ** 2.0
    mel_S   = mel_filters_matrix @ S_power
    log_mel = np.log(mel_S + 1e-6)
    mfcc    = dct_matrix @ log_mel
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)


def load_model():
    model      = WakeWordEmbedder(embedding_dim=EMBEDDING_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")
    return model


def get_embedding(model, wav_path):
    """Return L2-normalized 128-dim embedding for one WAV file."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    target_len = SAMPLE_RATE * 2
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc   = extract_mfcc_cpp_match(audio)
    mfcc   = (mfcc - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)  # (1,1,40,196)
    with torch.no_grad():
        emb = model(tensor)   # (1,128) — L2 normalized inside model
    return emb.squeeze().numpy()   # (128,)


# ══════════════════════════════════════════════════════════════════
# MAIN REGISTRATION
# ══════════════════════════════════════════════════════════════════
def register(samples_dir):

    if not os.path.exists(samples_dir):
        print(f"ERROR: Folder not found: {samples_dir}")
        sys.exit(1)

    wav_files = sorted([
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.lower().endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"ERROR: No WAV files found in {samples_dir}")
        sys.exit(1)

    if len(wav_files) < 5:
        print(f"⚠️  Only {len(wav_files)} samples — recommend at least 10.")

    if TOP_K > len(wav_files):
        print(f"⚠️  TOP_K={TOP_K} but only {len(wav_files)} samples.")
        print(f"   Reduce TOP_K in this file and in wake_word_detector.cpp")

    print(f"\nFound {len(wav_files)} samples:")
    for f in wav_files:
        print(f"  {os.path.basename(f)}")

    # ── Compute embeddings ────────────────────────────────────────
    model      = load_model()
    embeddings = []
    good_files = []

    print(f"\nComputing embeddings...")
    for wav_path in wav_files:
        try:
            emb = get_embedding(model, wav_path)
            embeddings.append(emb)
            good_files.append(wav_path)
            print(f"  ✅ {os.path.basename(wav_path)}")
        except Exception as e:
            print(f"  ❌ {os.path.basename(wav_path)} — {e}")

    if len(embeddings) == 0:
        print("ERROR: All files failed.")
        sys.exit(1)

    # ── Save ALL embeddings as matrix (N, 128) — NO averaging ─────
    template_bank = np.array(embeddings, dtype=np.float32)
    np.save(TEMPLATES_SAVE, template_bank)

    print(f"\n{'='*55}")
    print(f"  ✅  Wake word registered!")
    print(f"  Samples stored : {len(embeddings)} / {len(wav_files)}")
    print(f"  Template shape : {template_bank.shape}  ← (N_samples, {EMBEDDING_DIM})")
    print(f"  Saved to       : {TEMPLATES_SAVE}")
    print(f"  At detection   : top-{TOP_K} average similarity used")
    print(f"{'='*55}")

    # ── Quality check — pairwise similarity matrix ─────────────────
    # All samples should score reasonably similar to each other.
    # An outlier means that recording is inconsistent — re-record it.
    n = len(embeddings)
    print(f"\nPairwise similarity matrix (aim for > 0.60 on most pairs):\n")

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = float(np.dot(embeddings[i], embeddings[j]))

    # Print header
    print("       " + "  ".join([f"S{i+1:02d}" for i in range(n)]))
    for i in range(n):
        row = f"  S{i+1:02d}  "
        for j in range(n):
            row += "   -- " if i == j else f"  {sim_matrix[i,j]:.2f}"
        print(row)

    # Per-sample average similarity
    print(f"\nPer-sample average similarity to all other samples:")
    outliers = []
    for i in range(n):
        others = [sim_matrix[i,j] for j in range(n) if j != i]
        avg    = np.mean(others)
        bar    = "█" * int(avg * 20)
        flag   = ""
        if avg < 0.50:
            flag = "  ⚠️  LOW — consider re-recording"
            outliers.append(i + 1)
        print(f"  Sample {i+1:02d}: {avg:.4f}  {bar}{flag}")

    overall = np.mean([sim_matrix[i,j] for i in range(n)
                       for j in range(n) if i != j])
    print(f"\n  Overall average pairwise similarity: {overall:.4f}")

    if overall < 0.50:
        print("  ⚠️  Low — try recording more consistently.")
    elif overall < 0.70:
        print("  ✅  Acceptable.")
    else:
        print("  ✅  Excellent consistency.")

    if outliers:
        print(f"\n  ⚠️  Samples {outliers} scored low.")
        print(f"     Re-record those files and run this script again.")

    print(f"\nNext steps:")
    print(f"  1. Copy {TEMPLATES_SAVE} to your Raspberry Pi")
    print(f"  2. Run: ./wake_word_detector {TEMPLATES_SAVE} 0.75")


if __name__ == "__main__":
    register(SAMPLES_DIR)