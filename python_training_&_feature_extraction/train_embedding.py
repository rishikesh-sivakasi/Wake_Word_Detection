"""
train_embedding.py
Trains the wake word embedding model using a 5-source episodic training mix:
Speech Commands, Indian accents, Hey Saya positives, hard negatives, and ESC-50 noise.
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
import glob

from embedding_model import WakeWordEmbedder

# --- PATHS ---
BASE_DIR = os.environ.get("WAKE_WORD_BASE_DIR", ".")

SPEECH_COMMANDS_DIR = os.path.join(BASE_DIR, "speech_commands")
HEY_SAYA_DIR        = os.path.join(BASE_DIR, "dataset", "train", "positive")
HARD_NEGATIVE_DIR   = os.path.join(BASE_DIR, "dataset", "train", "negative")
MODEL_SAVE          = os.path.join(BASE_DIR, "embedding_model.pth")
STATS_FILE          = os.path.join(BASE_DIR, "global_stats.txt")

# --- AUDIO CONFIG ---
SAMPLE_RATE   = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 40
N_MFCC        = 40
MAX_FRAMES    = 196
EMBEDDING_DIM = 128

# --- TRAINING CONFIG ---
N_SHOT             = 10
N_QUERY            = 5
N_WAY              = 5
EPISODES_PER_EPOCH = 200
VAL_EPISODES       = 60
EPOCHS             = 60
LR                 = 1e-3
PATIENCE           = 10

# --- EPISODE RATIOS ---
HEY_SAYA_EPISODE_RATIO  = 0.35
INDIAN_EPISODE_RATIO    = 0.25
NOISE_EPISODE_RATIO     = 0.15   
# Speech Commands receives the remaining 25%

INDIAN_MIN_GROUP_SIZE = N_SHOT + N_QUERY   
ESC50_MIN_GROUP_SIZE  = N_SHOT + N_QUERY   

WORDS_TO_USE = [
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog",
    "happy", "house", "marvin", "sheila", "tree", "wow"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- MFCC FEATURE EXTRACTION ---
import librosa.filters as lf

_mel = lf.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32)
_n   = np.arange(N_MELS)
_k   = np.arange(N_MFCC)[:, np.newaxis]
_dct = np.cos(np.pi / N_MELS * (_n + 0.5) * _k)
_dct[0] *= 1.0 / np.sqrt(2)
_dct    *= np.sqrt(2.0 / N_MELS)
_dct     = _dct.astype(np.float32)


def pre_emphasis(audio, coeff=0.97):
    # Apply standard pre-emphasis filter
    out = np.copy(audio)
    for i in range(len(audio) - 1, 0, -1):
        out[i] = audio[i] - coeff * audio[i - 1]
    return out


def agc(audio, threshold=0.15, target=0.80, max_gain=8.0):
    # Automatic Gain Control to normalize volume
    mv = np.max(np.abs(audio))
    if mv > threshold:
        gain  = np.clip(target / mv, 1.0, max_gain)
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    # Extracts exactly the same features as the C++ edge inference
    audio   = pre_emphasis(audio)
    audio   = agc(audio)
    S_power = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   window='hamming', center=False)) ** 2.0
    mel_S   = _mel @ S_power
    log_mel = np.log(mel_S + 1e-6)
    mfcc    = _dct @ log_mel
    
    # Pad or truncate to ensure strict MAX_FRAMES length
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0,0),(0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)


def load_audio(path):
    # Load and force audio to exactly 2.0 seconds
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target   = SAMPLE_RATE * 2
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio


# --- DOWNLOAD SPEECH COMMANDS ---
def download_speech_commands():
    # Downloads Google Speech Commands dataset if missing
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


# --- DATASET LOADERS ---
def load_wav_files(folder, label):
    # Helper to load all .wav files from a directory
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
    # Maps Speech Commands words to their respective files
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
    """Returns dictionary mapping Indian word groups to file paths."""
    indian_groups = {}

    if not indian_data_dir or not os.path.exists(indian_data_dir):
        print(f"  {'Indian accent data':<25}: not found, skipping ")
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

    print(f"  {'Indian accent data':<25}: {len(indian_groups)} word groups loaded")
    return indian_groups


def build_esc50_group_map(esc50_dir):
    """Returns dictionary mapping ESC-50 noise categories to file paths."""
    esc50_groups = {}

    if not esc50_dir or not os.path.exists(esc50_dir):
        print(f"  {'ESC-50 noise':<25}: not found, skipping ")
        return esc50_groups

    for category in sorted(os.listdir(esc50_dir)):
        cat_dir = os.path.join(esc50_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        files = [
            os.path.join(cat_dir, f)
            for f in os.listdir(cat_dir)
            if f.lower().endswith(".wav")
        ]
        if len(files) >= ESC50_MIN_GROUP_SIZE:
            esc50_groups[category] = files

    print(f"  {'ESC-50 noise':<25}: {len(esc50_groups)} categories loaded")
    if len(esc50_groups) < N_WAY:
        print(f"  ⚠️  ESC-50 has only {len(esc50_groups)} categories. Noise episodes skipped.")

    return esc50_groups


# --- GLOBAL STATS ---
def compute_global_stats(word_files, indian_groups, hey_saya_files,
                         neg_files, esc50_groups, n_samples=600):
    # Calculates mean and std across a random sample of all available datasets
    print("\nComputing GLOBAL_MEAN and GLOBAL_STD...")

    all_files  = [f for fs in word_files.values() for f in fs]
    all_files += [f for fs in indian_groups.values() for f in fs]
    all_files += hey_saya_files
    all_files += neg_files
    all_files += [f for fs in esc50_groups.values() for f in fs]

    sampled = random.sample(all_files, min(n_samples, len(all_files)))

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

    with open(STATS_FILE, "w") as f:
        f.write(f"GLOBAL_MEAN = {mean}\nGLOBAL_STD = {std}\n")
    
    return mean, std


# --- EPISODIC DATASET ---
class EpisodicDataset(Dataset):
    """
    Handles episodic batch generation across 4 types: 
    Hey Saya, Indian Accent, ESC-50 Noise, and Speech Commands.
    """
    def __init__(self, word_files, indian_groups, hey_saya_files, neg_files,
                 esc50_groups, global_mean, global_std, n_episodes):
        self.word_files     = word_files
        self.words          = list(word_files.keys())
        self.indian_groups  = indian_groups
        self.indian_keys    = list(indian_groups.keys())
        self.hey_saya       = hey_saya_files
        self.negatives      = neg_files
        self.esc50_groups   = esc50_groups          
        self.esc50_keys     = list(esc50_groups.keys())  
        self.mean           = global_mean
        self.std            = global_std
        self.n_episodes     = n_episodes

        self.can_hey_saya = len(hey_saya_files) >= (N_SHOT + N_QUERY)
        self.can_negatives = len(neg_files) >= (N_SHOT + N_QUERY)
        self.can_indian = len(self.indian_keys) >= N_WAY
        self.can_noise = len(self.esc50_keys) >= N_WAY

        # Calculate episode distribution ratios dynamically
        if self.can_noise:
            self._hs_ratio     = HEY_SAYA_EPISODE_RATIO
            self._indian_ratio = HEY_SAYA_EPISODE_RATIO + INDIAN_EPISODE_RATIO
            self._noise_ratio  = HEY_SAYA_EPISODE_RATIO + INDIAN_EPISODE_RATIO + NOISE_EPISODE_RATIO
        else:
            remaining = 1.0 - NOISE_EPISODE_RATIO
            hs_scaled     = HEY_SAYA_EPISODE_RATIO / remaining
            indian_scaled = INDIAN_EPISODE_RATIO   / remaining
            self._hs_ratio     = hs_scaled
            self._indian_ratio = hs_scaled + indian_scaled
            self._noise_ratio  = 1.1   

    def __len__(self):
        return self.n_episodes

    def load_mfcc(self, path):
        try:
            audio = load_audio(path)
            mfcc  = extract_mfcc(audio)
            mfcc  = (mfcc - self.mean) / (self.std + 1e-8)
            return torch.tensor(mfcc).unsqueeze(0)  
        except Exception:
            return torch.zeros(1, N_MFCC, MAX_FRAMES)

    def _episode_from_groups(self, group_lookup, group_keys, n_way):
        # Shared helper to generate N-way episodes from grouped data
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
        # Episode anchoring the wake word against random speech and negatives
        chosen         = random.sample(self.hey_saya, N_SHOT + N_QUERY)
        hs_support     = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
        hs_query_files = chosen[N_SHOT:]

        all_support = [torch.stack(hs_support)]
        all_queries = []
        all_labels  = []

        for f in hs_query_files:
            all_queries.append(self.load_mfcc(f))
            all_labels.append(0)

        class_idx      = 1
        remaining_ways = N_WAY - 1

        if self.can_negatives:
            neg_chosen  = random.sample(self.negatives, N_SHOT + N_QUERY)
            neg_support = [self.load_mfcc(f) for f in neg_chosen[:N_SHOT]]
            all_support.append(torch.stack(neg_support))
            for f in neg_chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)
            class_idx      += 1
            remaining_ways -= 1

        other_words = random.sample(self.words, remaining_ways)
        for word in other_words:
            chosen  = random.sample(self.word_files[word], N_SHOT + N_QUERY)
            support = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
            all_support.append(torch.stack(support))
            for f in chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)
            class_idx += 1

        combined    = list(zip(all_queries, all_labels))
        random.shuffle(combined)
        all_queries, all_labels = zip(*combined)

        return (torch.stack(all_support),
                torch.stack(list(all_queries)),
                torch.tensor(list(all_labels)))

    def make_indian_accent_episode(self):
        return self._episode_from_groups(self.indian_groups, self.indian_keys, N_WAY)

    def make_noise_episode(self):
        return self._episode_from_groups(self.esc50_groups, self.esc50_keys, N_WAY)

    def make_speech_commands_episode(self):
        return self._episode_from_groups(self.word_files, self.words, N_WAY)

    def __getitem__(self, idx):
        if idx >= self.n_episodes:
            raise IndexError

        r = random.random()
        if self.can_hey_saya and r < self._hs_ratio:
            return self.make_hey_saya_episode()
        elif self.can_indian and r < self._indian_ratio:
            return self.make_indian_accent_episode()
        elif self.can_noise and r < self._noise_ratio:
            return self.make_noise_episode()
        else:
            return self.make_speech_commands_episode()


# --- PROTOTYPICAL LOSS ---
def prototypical_loss(model, support, queries, labels):
    # Calculates distance-based loss for episodic training
    n_way, n_shot = support.shape[0], support.shape[1]

    support_flat = support.view(n_way * n_shot, *support.shape[2:]).to(DEVICE)
    support_embs = model(support_flat).view(n_way, n_shot, EMBEDDING_DIM)
    prototypes   = F.normalize(support_embs.mean(dim=1), p=2, dim=1)

    query_embs   = model(queries.to(DEVICE))
    logits       = (query_embs @ prototypes.T) * 10.0
    loss         = F.cross_entropy(logits, labels.to(DEVICE))
    accuracy     = (logits.argmax(dim=1) == labels.to(DEVICE)).float().mean().item()

    return loss, accuracy


# --- VALIDATION ---
def validate(model, val_words, val_indian_groups, hey_saya_files,
             neg_files, esc50_groups, mean, std):
    model.eval()
    dataset   = EpisodicDataset(
        val_words, val_indian_groups, hey_saya_files, neg_files,
        esc50_groups, mean, std, VAL_EPISODES
    )
    total_acc = 0.0
    with torch.no_grad():
        for support, queries, labels in dataset:
            _, acc     = prototypical_loss(model, support, queries, labels)
            total_acc += acc
    model.train()
    return total_acc / VAL_EPISODES


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Train the wake word embedding model")
    parser.add_argument("--indian_data", default=None)
    parser.add_argument("--esc50_noise", default=None)
    args = parser.parse_args()

    # Step 1: Download Speech Commands
    download_speech_commands()

    # Step 2: Load all datasets
    print("\nLoading datasets:")
    word_files     = build_word_file_map()
    indian_groups  = build_indian_group_map(args.indian_data)
    hey_saya_files = load_wav_files(HEY_SAYA_DIR,     "Hey Saya (positive)")
    neg_files      = load_wav_files(HARD_NEGATIVE_DIR, "Hard negatives")
    esc50_groups   = build_esc50_group_map(args.esc50_noise)   

    # Check Requirements
    if len(word_files) < N_WAY:
        print(f"\nERROR: Need {N_WAY} Speech Commands words. Check {SPEECH_COMMANDS_DIR}")
        return

    if len(hey_saya_files) < N_SHOT + N_QUERY:
        print(f"\nERROR: Need at least {N_SHOT + N_QUERY} Hey Saya samples.")
        return

    # Step 3: Compute normalization stats
    mean, std = compute_global_stats(
        word_files, indian_groups, hey_saya_files, neg_files, esc50_groups
    )

    # Step 4: Split datasets into train/val
    
    # Split Speech Commands
    all_words = list(word_files.keys())
    random.shuffle(all_words)
    split       = int(len(all_words) * 0.8)
    train_words = {w: word_files[w] for w in all_words[:split]}
    val_words   = {w: word_files[w] for w in all_words[split:]}

    # Split Indian Accents
    train_indian_groups = {}
    val_indian_groups   = {}
    if indian_groups:
        indian_keys = list(indian_groups.keys())
        random.shuffle(indian_keys)
        i_split             = int(len(indian_keys) * 0.8)
        train_indian_groups = {k: indian_groups[k] for k in indian_keys[:i_split]}
        val_indian_groups   = {k: indian_groups[k] for k in indian_keys[i_split:]}
    
    # Share ESC-50 across train and val
    train_esc50_groups = esc50_groups
    val_esc50_groups   = esc50_groups

    # Split Wake Word Samples
    train_hey_saya = glob.glob("dataset/train/positive/*.wav")
    val_hey_saya   = glob.glob("dataset/val/positive/*.wav")

    # Split Negatives
    neg_shuffled = list(neg_files)
    random.shuffle(neg_shuffled)
    neg_split  = int(len(neg_shuffled) * 0.8)
    train_neg  = neg_shuffled[:neg_split]
    val_neg    = neg_shuffled[neg_split:]

    # Step 5: Build Training Dataset
    train_dataset = EpisodicDataset(
        train_words, train_indian_groups, train_hey_saya, train_neg,
        train_esc50_groups, mean, std, EPISODES_PER_EPOCH
    )

    # Step 6: Model & Optimizer Configuration
    model     = WakeWordEmbedder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc     = 0.0
    patience_counter = 0

    # Training Loop
    for epoch in range(1, EPOCHS + 1):

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

        val_acc = validate(
            model, val_words, val_indian_groups,
            val_hey_saya, val_neg, val_esc50_groups, mean, std
        )
        scheduler.step(val_acc)

        print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | Train: {epoch_acc:.1%} | Val: {val_acc:.1%}")

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
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

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
    print(f"     Check: Saya <-> Noise similarity should now be < 0.35")
    print(f"  3. python3 quantize_exporter.py")
    print(f"  4. Copy weights_embedding.h + templates.npy to Pi")
    print(f"  5. Recompile and test")


if __name__ == "__main__":
    main()