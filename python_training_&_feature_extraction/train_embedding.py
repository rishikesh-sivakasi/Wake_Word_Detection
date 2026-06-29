"""
train_embedding.py — FINAL VERSION WITH INDIAN ACCENT DATA
============================================================
Trains the embedding model using FOUR data sources:
  1. Google Speech Commands       (general speech separation)
  2. Indian accent word groups    (Common Voice Hindi + Tamil,
                                    grouped by word — see prepare_commonvoice.py)
  3. Your Hey Saya samples        (teaches exact wake word boundary)
  4. Your negative samples        (teaches what to reject)

Episode mix per training step:
  35% Hey Saya episodes            — pushes Hey Saya far from everything else
  35% Indian accent episodes       — teaches "same word, different Indian
                                      accent" should map to close embeddings
  30% Speech Commands episodes     — general English speech separation

WHY ADD INDIAN ACCENT EPISODES:
  Speech Commands alone is English, single-accent (mostly US speakers).
  Training only on that does not teach the model that a Tamil speaker
  and a Hindi speaker saying the same word should land close together
  in embedding space. Indian accent episodes sample word-groups built
  from Common Voice Hindi/Tamil (via prepare_commonvoice.py) where each
  group already contains the same word spoken by many different
  speakers/accents — exactly the signal needed for accent-robust
  "Hey Saya" detection across Indian speakers.

HOW TO RUN:
  python3 train_embedding.py --indian_data ./indian_accent_data

BEFORE RUNNING:
  1. Run augment_wake_word.py first to create 300 positive samples
  2. Make sure your negative folder has 200+ clips
  3. Speech Commands downloads automatically if not present
  4. Run prepare_commonvoice.py first to build ./indian_accent_data
     (hindi/group_NNN/ and tamil/group_NNN/ folders of WAV clips)

AFTER RUNNING:
  1. Open global_stats.txt
  2. Copy GLOBAL_MEAN and GLOBAL_STD into:
     - register_wake_word.py
     - wake_word_detector.cpp
  3. Run: python3 register_wake_word.py
"""

import os
import random
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import urllib.request
import tarfile

from embedding_model import WakeWordEmbedder

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
# NOTE: Update BASE_DIR to match your actual project root, or run this
# script from your project root and leave it as "." (current directory).
BASE_DIR = os.environ.get("WAKE_WORD_BASE_DIR", ".")

SPEECH_COMMANDS_DIR = os.path.join(BASE_DIR, "speech_commands")
HEY_SAYA_DIR        = os.path.join(BASE_DIR, "dataset", "train", "positive")
HARD_NEGATIVE_DIR   = os.path.join(BASE_DIR, "dataset", "train", "negative")
MODEL_SAVE          = os.path.join(BASE_DIR, "embedding_model.pth")
STATS_FILE          = os.path.join(BASE_DIR, "global_stats.txt")

# ══════════════════════════════════════════════════════════════════
# AUDIO CONFIG
# ══════════════════════════════════════════════════════════════════
SAMPLE_RATE   = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 40
N_MFCC        = 40
MAX_FRAMES    = 196
EMBEDDING_DIM = 128

# ══════════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ══════════════════════════════════════════════════════════════════
N_SHOT             = 10    # support samples per class
N_QUERY            = 5     # query samples per class per episode
N_WAY              = 5     # classes per episode
EPISODES_PER_EPOCH = 200   # reduced from 500 — CPU-only training budget
VAL_EPISODES       = 60    # reduced from 100 — keeps validation fast
EPOCHS             = 40    # reduced from 80 — fits a 2-week CPU-only timeline
LR                 = 1e-3
PATIENCE           = 10    # slightly reduced to match lower EPOCHS

# Episode mix — must sum to 1.0
HEY_SAYA_EPISODE_RATIO  = 0.35    # forces Hey Saya far from everything else
INDIAN_EPISODE_RATIO    = 0.35    # teaches accent-invariant word recognition
# remaining (1 - HEY_SAYA_EPISODE_RATIO - INDIAN_EPISODE_RATIO) = 0.30
# goes to plain Speech Commands episodes

# Minimum clips required in an Indian word-group to use it for an episode
INDIAN_MIN_GROUP_SIZE = N_SHOT + N_QUERY   # = 15

WORDS_TO_USE = [
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog",
    "happy", "house", "marvin", "sheila", "tree", "wow"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# MFCC — exact match to mfcc.cpp + dataset_extractor.cpp
# ══════════════════════════════════════════════════════════════════
import librosa.filters as lf

_mel = lf.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32)
_n   = np.arange(N_MELS)
_k   = np.arange(N_MFCC)[:, np.newaxis]
_dct = np.cos(np.pi / N_MELS * (_n + 0.5) * _k)
_dct[0] *= 1.0 / np.sqrt(2)
_dct    *= np.sqrt(2.0 / N_MELS)
_dct     = _dct.astype(np.float32)


def pre_emphasis(audio, coeff=0.97):
    out = np.copy(audio)
    for i in range(len(audio) - 1, 0, -1):
        out[i] = audio[i] - coeff * audio[i - 1]
    return out


def agc(audio, threshold=0.15, target=0.80, max_gain=8.0):
    mv = np.max(np.abs(audio))
    if mv > threshold:
        gain  = np.clip(target / mv, 1.0, max_gain)
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    audio   = pre_emphasis(audio)
    audio   = agc(audio)
    S_power = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   window='hamming', center=False)) ** 2.0
    mel_S   = _mel @ S_power
    log_mel = np.log(mel_S + 1e-6)
    mfcc    = _dct @ log_mel
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0,0),(0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target   = SAMPLE_RATE * 2
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio


# ══════════════════════════════════════════════════════════════════
# DOWNLOAD SPEECH COMMANDS
# ══════════════════════════════════════════════════════════════════
def download_speech_commands():
    url     = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive = "speech_commands_v0.02.tar.gz"
    if os.path.exists(SPEECH_COMMANDS_DIR):
        print(f"Speech Commands already at {SPEECH_COMMANDS_DIR}")
        return
    print("Downloading Google Speech Commands (~2.3 GB)...")
    def progress(b, bs, total):
        print(f"\r  {min(b*bs*100/total, 100):.1f}%", end="", flush=True)
    urllib.request.urlretrieve(url, archive, reporthook=progress)
    print("\nExtracting...")
    os.makedirs(SPEECH_COMMANDS_DIR, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(SPEECH_COMMANDS_DIR)
    os.remove(archive)
    print("Done.")


# ══════════════════════════════════════════════════════════════════
# LOAD DATASETS
# ══════════════════════════════════════════════════════════════════
def load_wav_files(folder, label):
    if not os.path.exists(folder):
        print(f"WARNING: {label} folder not found: {folder}")
        return []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]
    print(f"  {label:<25}: {len(files)} files")
    return files


def build_word_file_map():
    word_files = {}
    for word in WORDS_TO_USE:
        d     = os.path.join(SPEECH_COMMANDS_DIR, word)
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.endswith(".wav")] if os.path.isdir(d) else []
        if len(files) >= N_SHOT + N_QUERY:
            word_files[word] = files
    print(f"  {'Speech Commands':<25}: {len(word_files)} words loaded")
    return word_files


def build_indian_group_map(indian_data_dir):
    """
    Loads word-groups produced by prepare_commonvoice.py.

    Expected structure:
      indian_data_dir/
        hindi/group_001/clip_0001.wav, clip_0002.wav, ...
        hindi/group_002/...
        tamil/group_001/...
        tamil/group_002/...

    Returns: { 'hindi_group_001': [wav paths], 'tamil_group_001': [...], ... }
    Groups smaller than INDIAN_MIN_GROUP_SIZE are skipped, since they
    cannot supply a full N_SHOT + N_QUERY episode on their own.
    """
    indian_groups = {}

    if not indian_data_dir or not os.path.exists(indian_data_dir):
        print(f"  {'Indian accent data':<25}: not found, skipping "
              f"(pass --indian_data to enable)")
        return indian_groups

    for lang in sorted(os.listdir(indian_data_dir)):
        lang_dir = os.path.join(indian_data_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for group_name in sorted(os.listdir(lang_dir)):
            group_dir = os.path.join(lang_dir, group_name)
            if not os.path.isdir(group_dir):
                continue
            files = [os.path.join(group_dir, f) for f in os.listdir(group_dir)
                     if f.endswith(".wav")]
            if len(files) >= INDIAN_MIN_GROUP_SIZE:
                key = f"{lang}_{group_name}"
                indian_groups[key] = files

    print(f"  {'Indian accent data':<25}: {len(indian_groups)} word groups loaded "
          f"(>= {INDIAN_MIN_GROUP_SIZE} clips each)")
    return indian_groups


# ══════════════════════════════════════════════════════════════════
# GLOBAL STATS — computed from ALL data
# ══════════════════════════════════════════════════════════════════
def compute_global_stats(word_files, indian_groups, hey_saya_files, neg_files, n_samples=600):
    print("\nComputing GLOBAL_MEAN and GLOBAL_STD...")

    all_files  = [f for fs in word_files.values() for f in fs]
    all_files += [f for fs in indian_groups.values() for f in fs]
    all_files += hey_saya_files
    all_files += neg_files
    sampled    = random.sample(all_files, min(n_samples, len(all_files)))

    all_mfccs = []
    for fp in tqdm(sampled, desc="  Sampling"):
        try:
            mfcc = extract_mfcc(load_audio(fp))
            all_mfccs.append(mfcc.flatten())
        except Exception:
            continue

    stacked = np.concatenate(all_mfccs)
    mean    = float(np.mean(stacked))
    std     = float(np.std(stacked))

    print(f"\n  GLOBAL_MEAN = {mean:.8f}")
    print(f"  GLOBAL_STD  = {std:.8f}")
    print(f"\n  ⚠️  Copy these two numbers into:")
    print(f"     register_wake_word.py  → GLOBAL_MEAN, GLOBAL_STD")
    print(f"     wake_word_detector.cpp → GLOBAL_MEAN, GLOBAL_STD")

    with open(STATS_FILE, "w") as f:
        f.write(f"GLOBAL_MEAN = {mean}\nGLOBAL_STD = {std}\n")
    print(f"  Saved to: {STATS_FILE}\n")

    return mean, std


# ══════════════════════════════════════════════════════════════════
# EPISODIC DATASET
#
# Three types of episodes:
#
# TYPE 1 — Hey Saya episode (35% of training):
#   Class 0      = Hey Saya (10 support + 5 query from your 300 samples)
#   Class 1      = Hard negatives (10 support + 5 query), when available
#   Class 2..N-1 = Remaining slots filled with random Speech Commands words
#   This forces: Hey Saya embedding MUST be far from all other sounds,
#   including specifically your curated ambient/negative recordings —
#   not just generic words.
#
# TYPE 2 — Indian accent episode (35% of training):
#   Class 0..4 = 5 random Indian word-groups (Hindi/Tamil mixed)
#   Each group contains the SAME word spoken by many different
#   speakers/accents. This teaches: same word + different accent
#   => embeddings should be CLOSE together. This is the key fix for
#   generalising "Hey Saya" detection across Indian accents.
#
# TYPE 3 — Speech Commands episode (30% of training):
#   Class 0..4 = 5 random Speech Commands words
#   This teaches: general speech sound separation
# ══════════════════════════════════════════════════════════════════
class EpisodicDataset(Dataset):

    def __init__(self, word_files, indian_groups, hey_saya_files, neg_files,
                 global_mean, global_std, n_episodes):
        self.word_files     = word_files
        self.words          = list(word_files.keys())
        self.indian_groups  = indian_groups
        self.indian_keys    = list(indian_groups.keys())
        self.hey_saya       = hey_saya_files
        self.negatives      = neg_files
        self.mean           = global_mean
        self.std            = global_std
        self.n_episodes     = n_episodes

        self.can_hey_saya  = len(hey_saya_files) >= (N_SHOT + N_QUERY)
        if not self.can_hey_saya:
            print(f"WARNING: Need {N_SHOT+N_QUERY} Hey Saya files, "
                  f"have {len(hey_saya_files)}.")

        # FIX: previously self.negatives was stored but never read anywhere
        # in episode construction — your 200+ curated "hard negative" clips
        # only ever influenced the global mean/std normalization stats, not
        # the actual training signal. The model was never explicitly taught
        # "this is noise, don't fire on it." make_hey_saya_episode() below
        # now spends one class slot on hard negatives whenever there are
        # enough of them.
        self.can_negatives = len(neg_files) >= (N_SHOT + N_QUERY)
        if not self.can_negatives:
            print(f"WARNING: Need {N_SHOT+N_QUERY} hard negative files, "
                  f"have {len(neg_files)}. Hey Saya episodes will not "
                  f"include an explicit negative class until you have more.")

        self.can_indian = len(self.indian_keys) >= N_WAY
        if not self.can_indian:
            print(f"WARNING: Need {N_WAY} Indian accent word-groups, "
                  f"have {len(self.indian_keys)}. Indian episodes will be "
                  f"skipped (falls back to Speech Commands episodes).")

    def __len__(self):
        return self.n_episodes

    def load_mfcc(self, path):
        try:
            audio = load_audio(path)
            mfcc  = extract_mfcc(audio)
            mfcc  = (mfcc - self.mean) / (self.std + 1e-8)
            return torch.tensor(mfcc).unsqueeze(0)   # (1, 40, 196)
        except Exception:
            return torch.zeros(1, N_MFCC, MAX_FRAMES)

    def _episode_from_groups(self, group_lookup, group_keys, n_way):
        """
        Shared helper: builds an N-way episode from any dict of
        {class_key: [file paths]}. Used by both the Indian-accent
        episode and the Speech-Commands episode since the logic is
        identical — only the source dict differs.
        """
        episode_keys = random.sample(group_keys, n_way)
        all_support  = []
        all_queries  = []
        all_labels   = []

        for class_idx, key in enumerate(episode_keys):
            chosen  = random.sample(group_lookup[key], N_SHOT + N_QUERY)
            support = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
            all_support.append(torch.stack(support))
            for f in chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)

        combined    = list(zip(all_queries, all_labels))
        random.shuffle(combined)
        all_queries, all_labels = zip(*combined)

        return (torch.stack(all_support),
                torch.stack(list(all_queries)),
                torch.tensor(list(all_labels)))

    def make_hey_saya_episode(self):
        """
        Episode where Hey Saya is class 0.

        FIX: class 1 is now reserved for hard negatives whenever we have
        enough of them (self.can_negatives). Previously every "other class"
        slot was filled from Speech Commands words only, so your curated
        negative recordings never appeared as a training class — the model
        had no direct signal pushing "Hey Saya" away from your specific
        negative/ambient-noise recordings, only away from generic words
        like "yes"/"dog"/"stop". This is the main reason the registered
        Saya/Noise prototypes in templates.npy ended up only ~0.20 apart
        in cosine similarity (0.80 similarity) instead of being clearly
        separated.
        """
        # Hey Saya: 10 support + 5 query
        chosen           = random.sample(self.hey_saya, N_SHOT + N_QUERY)
        hs_support       = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
        hs_query_files   = chosen[N_SHOT:]

        all_support  = [torch.stack(hs_support)]
        all_queries  = []
        all_labels   = []

        # Hey Saya queries (class 0)
        for f in hs_query_files:
            all_queries.append(self.load_mfcc(f))
            all_labels.append(0)

        class_idx      = 1
        remaining_ways = N_WAY - 1

        # Hard-negative class (class 1), when we have enough clips
        if self.can_negatives:
            neg_chosen  = random.sample(self.negatives, N_SHOT + N_QUERY)
            neg_support = [self.load_mfcc(f) for f in neg_chosen[:N_SHOT]]
            all_support.append(torch.stack(neg_support))
            for f in neg_chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)
            class_idx      += 1
            remaining_ways -= 1

        # Remaining classes: random Speech Commands words
        other_words = random.sample(self.words, remaining_ways)
        for word in other_words:
            chosen  = random.sample(self.word_files[word], N_SHOT + N_QUERY)
            support = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
            all_support.append(torch.stack(support))
            for f in chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)
            class_idx += 1

        # Shuffle queries
        combined    = list(zip(all_queries, all_labels))
        random.shuffle(combined)
        all_queries, all_labels = zip(*combined)

        return (torch.stack(all_support),
                torch.stack(list(all_queries)),
                torch.tensor(list(all_labels)))

    def make_indian_accent_episode(self):
        """
        Episode using only Indian accent word-groups (Hindi + Tamil mixed).
        Each class = one word, support/query drawn from many different
        speakers/accents saying that same word. Teaches the model that
        accent variation within the same word should NOT separate the
        embeddings.
        """
        return self._episode_from_groups(self.indian_groups, self.indian_keys, N_WAY)

    def make_speech_commands_episode(self):
        """Standard episode using only Speech Commands words."""
        return self._episode_from_groups(self.word_files, self.words, N_WAY)

    def __getitem__(self, idx):
        if idx >= self.n_episodes:
            raise IndexError

        r = random.random()

        if self.can_hey_saya and r < HEY_SAYA_EPISODE_RATIO:
            return self.make_hey_saya_episode()

        elif self.can_indian and r < (HEY_SAYA_EPISODE_RATIO + INDIAN_EPISODE_RATIO):
            return self.make_indian_accent_episode()

        else:
            return self.make_speech_commands_episode()


# ══════════════════════════════════════════════════════════════════
# PROTOTYPICAL LOSS
# ══════════════════════════════════════════════════════════════════
def prototypical_loss(model, support, queries, labels):
    n_way, n_shot = support.shape[0], support.shape[1]

    support_flat = support.view(n_way * n_shot, *support.shape[2:]).to(DEVICE)
    support_embs = model(support_flat).view(n_way, n_shot, EMBEDDING_DIM)
    prototypes   = F.normalize(support_embs.mean(dim=1), p=2, dim=1)

    query_embs   = model(queries.to(DEVICE))
    logits       = (query_embs @ prototypes.T) * 10.0
    loss         = F.cross_entropy(logits, labels.to(DEVICE))
    accuracy     = (logits.argmax(dim=1) == labels.to(DEVICE)).float().mean().item()

    return loss, accuracy


# ══════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════
def validate(model, val_words, val_indian_groups, hey_saya_files, neg_files, mean, std):
    model.eval()
    dataset   = EpisodicDataset(val_words, val_indian_groups, hey_saya_files, neg_files,
                                mean, std, VAL_EPISODES)
    total_acc = 0.0
    with torch.no_grad():
        for support, queries, labels in dataset:
            _, acc     = prototypical_loss(model, support, queries, labels)
            total_acc += acc
    model.train()
    return total_acc / VAL_EPISODES


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train the wake word embedding model")
    parser.add_argument("--indian_data", default=None,
                        help="Path to prepared Indian accent data "
                             "(output of prepare_commonvoice.py). "
                             "If omitted, training falls back to "
                             "Speech Commands + Hey Saya only.")
    args = parser.parse_args()

    # ── Step 1: Download Speech Commands ─────────────────────────
    download_speech_commands()

    # ── Step 2: Load all datasets ─────────────────────────────────
    print("\nLoading datasets:")
    word_files     = build_word_file_map()
    indian_groups  = build_indian_group_map(args.indian_data)
    hey_saya_files = load_wav_files(HEY_SAYA_DIR,       "Hey Saya (positive)")
    neg_files      = load_wav_files(HARD_NEGATIVE_DIR,   "Hard negatives")

    print(f"\nDataset summary:")
    print(f"  Speech Commands words : {len(word_files)}")
    print(f"  Indian accent groups  : {len(indian_groups)}")
    print(f"  Hey Saya samples      : {len(hey_saya_files)}")
    print(f"  Hard negative samples : {len(neg_files)}")
    print(f"  Episode mix           : {HEY_SAYA_EPISODE_RATIO:.0%} Hey Saya / "
          f"{INDIAN_EPISODE_RATIO:.0%} Indian accent / "
          f"{1 - HEY_SAYA_EPISODE_RATIO - INDIAN_EPISODE_RATIO:.0%} Speech Commands")

    if not indian_groups:
        print(f"\n  ⚠️  No Indian accent data loaded — training will proceed with")
        print(f"     Speech Commands + Hey Saya only. Pass --indian_data to add it:")
        print(f"     python3 train_embedding.py --indian_data ./indian_accent_data")

    # Check minimum requirements
    if len(word_files) < N_WAY:
        print(f"\nERROR: Need {N_WAY} Speech Commands words. Check {SPEECH_COMMANDS_DIR}")
        return

    if len(hey_saya_files) < N_SHOT + N_QUERY:
        print(f"\nERROR: Need at least {N_SHOT + N_QUERY} Hey Saya samples.")
        print(f"  Run augment_wake_word.py first to create 300 samples.")
        return

    # ── Step 3: Compute normalization stats ───────────────────────
    mean, std = compute_global_stats(word_files, indian_groups, hey_saya_files, neg_files)

    # ── Step 4: Split Speech Commands into train/val ──────────────
    all_words = list(word_files.keys())
    random.shuffle(all_words)
    split       = int(len(all_words) * 0.8)
    train_words = {w: word_files[w] for w in all_words[:split]}
    val_words   = {w: word_files[w] for w in all_words[split:]}

    print(f"\nTrain words ({len(train_words)}): {list(train_words.keys())}")
    print(f"Val words   ({len(val_words)}):   {list(val_words.keys())}")

    # ── Step 4b: Split Indian accent groups into train/val ─────────
    train_indian_groups = {}
    val_indian_groups   = {}
    if indian_groups:
        indian_keys = list(indian_groups.keys())
        random.shuffle(indian_keys)
        i_split              = int(len(indian_keys) * 0.8)
        train_indian_groups  = {k: indian_groups[k] for k in indian_keys[:i_split]}
        val_indian_groups    = {k: indian_groups[k] for k in indian_keys[i_split:]}

        print(f"Train Indian groups ({len(train_indian_groups)})")
        print(f"Val Indian groups   ({len(val_indian_groups)})")

        if len(val_indian_groups) < N_WAY:
            print(f"  ⚠️  Val Indian groups below N_WAY={N_WAY} — "
                  f"validation will fall back to Speech Commands episodes only.")

    # ── Step 4c: Split Hey Saya and hard-negative samples into train/val ──
    # FIX: previously hey_saya_files/neg_files were passed unsplit to BOTH
    # the training dataset and validate() — meaning validation accuracy on
    # Hey-Saya-type episodes was measured partly on samples the model had
    # already trained on. Speech Commands and Indian groups were already
    # split correctly; this brings Hey Saya / negatives in line with them.
    hey_saya_shuffled = list(hey_saya_files)
    random.shuffle(hey_saya_shuffled)
    hs_split       = int(len(hey_saya_shuffled) * 0.8)
    train_hey_saya = hey_saya_shuffled[:hs_split]
    val_hey_saya   = hey_saya_shuffled[hs_split:]

    neg_shuffled = list(neg_files)
    random.shuffle(neg_shuffled)
    neg_split  = int(len(neg_shuffled) * 0.8)
    train_neg  = neg_shuffled[:neg_split]
    val_neg    = neg_shuffled[neg_split:]

    print(f"\nTrain Hey Saya ({len(train_hey_saya)}) / Val Hey Saya ({len(val_hey_saya)})")
    print(f"Train negatives ({len(train_neg)}) / Val negatives ({len(val_neg)})")

    if len(val_hey_saya) < N_SHOT + N_QUERY:
        print(f"  ⚠️  Val Hey Saya below {N_SHOT + N_QUERY} clips — record more, "
              f"or validation accuracy on Hey Saya episodes will be unreliable.")
    if len(val_neg) < N_SHOT + N_QUERY:
        print(f"  ⚠️  Val negatives below {N_SHOT + N_QUERY} clips — record more "
              f"hard negatives, or this class will be skipped in validation episodes.")

    train_dataset = EpisodicDataset(
        train_words, train_indian_groups, train_hey_saya, train_neg,
        mean, std, EPISODES_PER_EPOCH
    )

    # ── Step 5: Model + optimizer ─────────────────────────────────
    model     = WakeWordEmbedder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    print(f"\n{'='*60}")
    print(f"  Training — {N_WAY}-way {N_SHOT}-shot episodic")
    print(f"  {HEY_SAYA_EPISODE_RATIO:.0%} Hey Saya / "
          f"{INDIAN_EPISODE_RATIO:.0%} Indian accent / "
          f"{1 - HEY_SAYA_EPISODE_RATIO - INDIAN_EPISODE_RATIO:.0%} Speech Commands")
    print(f"  Target: Val Acc > 80%")
    print(f"{'='*60}\n")

    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        epoch_loss = epoch_acc = 0.0

        pbar = tqdm(train_dataset, desc=f"Epoch {epoch:03d}", leave=False)
        for support, queries, labels in pbar:
            optimizer.zero_grad()
            loss, acc  = prototypical_loss(model, support, queries, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc  += acc
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.0%}")

        epoch_loss /= EPISODES_PER_EPOCH
        epoch_acc  /= EPISODES_PER_EPOCH

        # Validate
        val_acc = validate(model, val_words, val_indian_groups, val_hey_saya, val_neg, mean, std)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:03d} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train: {epoch_acc:.1%} | "
              f"Val: {val_acc:.1%}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "global_mean": mean,
                "global_std":  std,
                "val_acc":     val_acc,
            }, MODEL_SAVE)
            print(f"  ✅ Best model saved (val={val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.1%}")

    if best_val_acc < 0.65:
        print(f"\n  ⚠️  Below 65% — open this file and change:")
        print(f"     EPOCHS   = 120")
        print(f"     PATIENCE = 20")
        print(f"  Then run again.")
    elif best_val_acc < 0.80:
        print(f"\n  ✅  Good — test on Raspberry Pi.")
    else:
        print(f"\n  ✅  Excellent — model is ready.")

    print(f"\n  Next steps:")
    print(f"  1. Open {STATS_FILE}")
    print(f"     Copy GLOBAL_MEAN and GLOBAL_STD into:")
    print(f"     - register_wake_word.py")
    print(f"     - wake_word_detector.cpp")
    print(f"  2. python3 register_wake_word.py")
    print(f"  3. python3 quantize_exporter.py")
    print(f"  4. Copy weights_embedding.h + templates.npy to Pi")
    print(f"  5. Recompile and test")


if __name__ == "__main__":
    main()

#   GLOBAL_MEAN = -1.94761944
#   GLOBAL_STD  = 12.18327618