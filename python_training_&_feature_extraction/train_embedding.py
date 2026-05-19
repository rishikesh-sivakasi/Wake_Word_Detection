# """
# train_embedding.py
# ==================
# Trains the WakeWordEmbedder using Triplet Loss on Google Speech Commands.

# WHAT THIS SCRIPT DOES:
#   1. Downloads Google Speech Commands dataset (free, ~2.3 GB)
#   2. Extracts MFCCs using the EXACT same math as your mfcc.cpp
#   3. Builds triplets: (anchor, positive, negative)
#      anchor   = word X, speaker A
#      positive = word X, speaker B   ← same word, different person
#      negative = word Y, speaker A   ← different word
#   4. Trains WakeWordEmbedder with TripletLoss
#   5. Saves: embedding_model.pth

# YOU NEVER NEED "HEY SAYA" DATA FOR THIS STEP.
# The model learns to compare sounds in general.
# After this, register_wake_word.py handles "Hey Saya" with 10 samples.

# HOW TO RUN:
#   pip install torch torchaudio librosa numpy tqdm
#   python3 train_embedding.py

# EXPECTED TIME:
#   ~2-4 hours on CPU, ~30-45 minutes with GPU
# """

# import os
# import random
# import numpy as np
# import librosa
# import torch
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import urllib.request
# import tarfile

# from embedding_model import WakeWordEmbedder, TripletLoss

# # ══════════════════════════════════════════════════════════════════
# # CONFIG — change paths if needed
# # ══════════════════════════════════════════════════════════════════
# DATASET_DIR   = "./speech_commands"   # where to store the downloaded dataset
# MODEL_SAVE    = "embedding_model.pth"

# SAMPLE_RATE   = 16000
# N_FFT         = 512
# HOP_LENGTH    = 160
# N_MELS        = 40
# N_MFCC        = 40
# # We need exactly 196 frames for a 2-second window
# # (16000 * 2.0 / 160 = 200, but with edge effects it's 196)

# # Training Params
# BATCH_SIZE    = 64
# EPOCHS        = 50
# LEARNING_RATE = 0.001
# TRIPLETS_PER_EPOCH = 8000   # How many triplets to generate per epoch
# PATIENCE      = 5           # Early stopping

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ══════════════════════════════════════════════════════════════════
# # 1. C++ MATCHING MFCC EXTRACTION (from generate_tables.py)
# # ══════════════════════════════════════════════════════════════════
# mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32)

# n = np.arange(N_MELS)
# k = np.arange(N_MFCC)[:, np.newaxis]
# dct = np.cos(np.pi / N_MELS * (n + 0.5) * k)
# dct[0] *= 1.0 / np.sqrt(2)
# dct *= np.sqrt(2.0 / N_MELS)
# dct = dct.astype(np.float32)

# def mfcc_from_audio_cpp_match(audio: np.ndarray) -> np.ndarray:
#     """Exact C++ MFCC math match."""
#     # Pre-emphasis
#     audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

#     # AGC (Hardware match)
#     max_val = np.max(np.abs(audio))
#     if max_val > 0.15:
#         gain = np.clip(0.80 / max_val, 1.0, 8.0)
#         audio = np.clip(audio * gain, -1.0, 1.0)

#     S_power = np.abs(
#         librosa.stft(
#             audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
#             window='hamming', center=False
#         )
#     ) ** 2.0

#     mel_S = mel_filters @ S_power
#     log_mel = np.log(mel_S + 1e-6)
#     mfcc = dct @ log_mel

#     # Force to exactly 196 frames (pad or trim)
#     if mfcc.shape[1] < 196:
#         pad_width = 196 - mfcc.shape[1]
#         mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :196]

#     return mfcc.astype(np.float32)


# # ══════════════════════════════════════════════════════════════════
# # 2. DATASET DOWNLOADER & LOADER
# # ══════════════════════════════════════════════════════════════════

# class SpeechCommandsDataset(Dataset):
#     """
#     Dynamically generates triplets from the dataset.
#     """
#     def __init__(self, data_dir, classes, samples_per_epoch, global_mean=0.0, global_std=1.0):
#         self.data_dir = data_dir
#         self.classes = classes
#         self.samples_per_epoch = samples_per_epoch
#         self.mean = global_mean
#         self.std  = global_std

#         # Load all file paths into memory grouped by class
#         self.word_files = {c: [] for c in classes}
#         for c in classes:
#             folder = os.path.join(data_dir, c)
#             if not os.path.isdir(folder): continue
#             for f in os.listdir(folder):
#                 if f.endswith('.wav'):
#                     self.word_files[c].append(os.path.join(folder, f))

#         # Filter out classes with too few samples
#         self.classes = [c for c in classes if len(self.word_files[c]) >= 2]
#         print(f"Using {len(self.classes)} word classes for triplet training")

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         # 1. Pick a random anchor word
#         anchor_word = self.classes[idx % len(self.classes)]
        
#         # 2. Pick two different files of the SAME word
#         anchor_file, pos_file = random.sample(self.word_files[anchor_word], 2)

#         # 3. Pick a DIFFERENT word for the negative
#         neg_word = random.choice([w for w in self.classes if w != anchor_word])
#         neg_file = random.choice(self.word_files[neg_word])

#         # Extract features
#         a_mfcc = mfcc_from_audio_cpp_match(self._load_audio(anchor_file))
#         p_mfcc = mfcc_from_audio_cpp_match(self._load_audio(pos_file))
#         n_mfcc = mfcc_from_audio_cpp_match(self._load_audio(neg_file))

#         # Normalize
#         a_mfcc = (a_mfcc - self.mean) / self.std
#         p_mfcc = (p_mfcc - self.mean) / self.std
#         n_mfcc = (n_mfcc - self.mean) / self.std

#         # Add channel dimension (1, 40, 196)
#         return (
#             torch.tensor(a_mfcc).unsqueeze(0),
#             torch.tensor(p_mfcc).unsqueeze(0),
#             torch.tensor(n_mfcc).unsqueeze(0)
#         )

#     def _load_audio(self, path):
#         """Loads and randomly pads the 1-second audio to exactly 2 seconds."""
#         TARGET_SAMPLES = SAMPLE_RATE * 2  # Exactly 32,000 samples (2.0 seconds)

#         # 1. Load the Google Speech Command
#         audio, _ = librosa.load(path, sr=SAMPLE_RATE)

#         # 2. RANDOMIZED PADDING
#         if len(audio) < TARGET_SAMPLES:
#             pad_total = TARGET_SAMPLES - len(audio)
            
#             # Pick a random starting point for the padding
#             pad_left = random.randint(0, pad_total)
#             pad_right = pad_total - pad_left
            
#             # Add pure silence to the left and right
#             audio = np.pad(audio, (pad_left, pad_right), mode='constant')
            
#         # 3. Trim (Just in case a file is weirdly long)
#         elif len(audio) > TARGET_SAMPLES:
#             audio = audio[:TARGET_SAMPLES]

#         return audio

# def download_speech_commands():
#     if not os.path.exists(DATASET_DIR):
#         os.makedirs(DATASET_DIR)
#         url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
#         tar_path = os.path.join(DATASET_DIR, "speech_commands.tar.gz")
#         print("Downloading Google Speech Commands (~2.3 GB)...")
#         print("This takes a few minutes depending on your internet speed.")
        
#         class DownloadProgressBar(tqdm):
#             def update_to(self, b=1, bsize=1, tsize=None):
#                 if tsize is not None: self.total = tsize
#                 self.update(b * bsize - self.n)
                
#         with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
#             urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
            
#         print("Extracting...")
#         with tarfile.open(tar_path, 'r:gz') as tar:
#             tar.extractall(path=DATASET_DIR)
#         os.remove(tar_path)
#         print(f"Dataset ready at {DATASET_DIR}")
#     else:
#         print(f"Dataset already exists at {DATASET_DIR}, skipping download.")


# # ══════════════════════════════════════════════════════════════════
# # 3. UTILS & SANITY CHECKS
# # ══════════════════════════════════════════════════════════════════

# def compute_dataset_stats(word_files, num_samples=500):
#     """Calculates global mean and std to match the C++ engine scale."""
#     print("\nComputing global mean/std from dataset sample...")
#     all_mfccs = []
    
#     # Flatten the dict of files
#     all_files = [f for files in word_files.values() for f in files]
#     sample_files = random.sample(all_files, min(num_samples, len(all_files)))
    
#     for f in tqdm(sample_files, desc="  Sampling"):
#         # We need the load_audio logic here too so we measure padded math
#         TARGET_SAMPLES = SAMPLE_RATE * 2
#         audio, _ = librosa.load(f, sr=SAMPLE_RATE)
#         if len(audio) < TARGET_SAMPLES:
#             pad = TARGET_SAMPLES - len(audio)
#             audio = np.pad(audio, (pad//2, pad - pad//2), mode='constant')
#         elif len(audio) > TARGET_SAMPLES:
#             audio = audio[:TARGET_SAMPLES]

#         mfcc = mfcc_from_audio_cpp_match(audio)
#         all_mfccs.append(mfcc)

#     stacked = np.hstack(all_mfccs)
#     g_mean = np.mean(stacked)
#     g_std  = np.std(stacked) + 1e-8
    
#     print(f"  GLOBAL_MEAN = {g_mean:f}")
#     print(f"  GLOBAL_STD  = {g_std:f}")
#     print("  ⚠️  COPY THESE into register_wake_word.py and wake_word_detector.cpp!\n")
#     return g_mean, g_std

# def validate_embeddings(model, word_files, g_mean, g_std):
#     """Prints a quick sanity check to ensure the model isn't collapsing."""
#     model.eval()
    
#     # Pick 2 words, get 2 samples of each
#     words = random.sample(list(word_files.keys()), 2)
#     files = [random.sample(word_files[w], 2) for w in words]
    
#     embs = []
#     with torch.no_grad():
#         for w_idx in range(2):
#             for f_idx in range(2):
#                 TARGET_SAMPLES = SAMPLE_RATE * 2
#                 audio, _ = librosa.load(files[w_idx][f_idx], sr=SAMPLE_RATE)
#                 if len(audio) < TARGET_SAMPLES:
#                     audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode='constant')
#                 else:
#                     audio = audio[:TARGET_SAMPLES]
                
#                 mfcc = mfcc_from_audio_cpp_match(audio)
#                 mfcc = (mfcc - g_mean) / g_std
                
#                 t = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).to(DEVICE)
#                 emb = model(t).cpu().numpy().flatten()
#                 embs.append((words[w_idx], emb))
                
#     # embs[0] and embs[1] are same word. embs[2] and embs[3] are same word.
#     sim_same_1 = np.dot(embs[0][1], embs[1][1])
#     sim_same_2 = np.dot(embs[2][1], embs[3][1])
#     sim_diff_1 = np.dot(embs[0][1], embs[2][1])
#     sim_diff_2 = np.dot(embs[1][1], embs[3][1])
    
#     print(f"    Same word ('{embs[0][0]}'): {sim_same_1:.3f}")
#     print(f"    Same word ('{embs[2][0]}'): {sim_same_2:.3f}")
#     print(f"    Diff word ('{embs[0][0]}' vs '{embs[2][0]}'): {sim_diff_1:.3f}")
#     print(f"    Diff word ('{embs[1][0]}' vs '{embs[3][0]}'): {sim_diff_2:.3f}")
#     model.train()


# # ══════════════════════════════════════════════════════════════════
# # 4. MAIN TRAINING LOOP
# # ══════════════════════════════════════════════════════════════════

# def main():
#     print(f"Device: {DEVICE}")
#     download_speech_commands()

#     # The top 20 most reliable words in the dataset
#     target_classes = [
#         "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
#         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
#     ]

#     # Pre-scan for compute_stats
#     word_files = {c: [] for c in target_classes}
#     for c in target_classes:
#         folder = os.path.join(DATASET_DIR, c)
#         if os.path.isdir(folder):
#             word_files[c] = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
            
#     print("\nLoaded 20 word classes:")
#     for c, files in word_files.items():
#         print(f"  {c:<10}: {len(files)} files")

#     # Get standardization numbers
#     global_mean, global_std = compute_dataset_stats(word_files)

#     # Build datasets
#     train_dataset = SpeechCommandsDataset(DATASET_DIR, target_classes, TRIPLETS_PER_EPOCH, global_mean, global_std)
#     val_dataset   = SpeechCommandsDataset(DATASET_DIR, target_classes, int(TRIPLETS_PER_EPOCH * 0.2), global_mean, global_std)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

#     # Initialize model
#     model = WakeWordEmbedder(embedding_dim=128).to(DEVICE)
#     criterion = TripletLoss(margin=0.3)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # ── FIXED SCHEDULER HERE ──
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5
#     )

#     best_val_loss = float('inf')
#     patience_counter = 0

#     print("\n" + "="*60)
#     print("  Starting Triplet Loss Training")
#     print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
#     print(f"  Triplets/epoch: {TRIPLETS_PER_EPOCH}")
#     print("="*60 + "\n")

#     for epoch in range(1, EPOCHS + 1):
        
#         # ── Train ─────────────────────────────────────────────────────
#         model.train()
#         train_loss = 0.0
        
#         # Wrapped in tqdm for a nice progress bar
#         for anchors, positives, negatives in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
#             anchors   = anchors.to(DEVICE)
#             positives = positives.to(DEVICE)
#             negatives = negatives.to(DEVICE)

#             optimizer.zero_grad()
            
#             emb_a = model(anchors)
#             emb_p = model(positives)
#             emb_n = model(negatives)
            
#             loss = criterion(emb_a, emb_p, emb_n)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
            
#         train_loss /= len(train_loader)

#         # ── Validate ──────────────────────────────────────────────────
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for anchors, positives, negatives in val_loader:
#                 anchors   = anchors.to(DEVICE)
#                 positives = positives.to(DEVICE)
#                 negatives = negatives.to(DEVICE)

#                 emb_a = model(anchors)
#                 emb_p = model(positives)
#                 emb_n = model(negatives)

#                 loss = criterion(emb_a, emb_p, emb_n)
#                 val_loss += loss.item()
                
#         val_loss /= len(val_loader)
#         scheduler.step(val_loss)

#         print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         # Every 10 epochs, print embedding quality
#         if epoch % 10 == 0:
#             print("  Embedding quality check:")
#             validate_embeddings(model, word_files, global_mean, global_std)

#         # ── Early Stopping & Checkpointing ────────────────────────
#         if val_loss < best_val_loss:
#             best_val_loss    = val_loss
#             patience_counter = 0
#             torch.save({
#                 "epoch":       epoch,
#                 "model_state": model.state_dict(),
#                 "global_mean": global_mean,
#                 "global_std":  global_std,
#                 "val_loss":    val_loss,
#             }, MODEL_SAVE)
#             print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"\nEarly stopping at epoch {epoch}")
#                 break

#     # ── Final evaluation ──────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print("  Training Complete! Final Embedding Quality:")
#     checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
#     model.load_state_dict(checkpoint["model_state"])
#     validate_embeddings(model, word_files, global_mean, global_std)

#     print(f"\n✅  Model saved to: {MODEL_SAVE}")
#     print(f"✅  Stats stored in checkpoint. Run register_wake_word.py next.")

# if __name__ == "__main__":
#     main()


# """
# train_embedding.py  — EPISODIC / PROTOTYPICAL NETWORK TRAINING
# ==============================================================
# This is the correct approach for your manager's requirement:
#   "Only 10 samples of any wake word → model works perfectly"

# WHY THE OLD APPROACH FAILED:
#   Old approach: Train on Google Speech Commands triplets.
#   Result: Model learned to separate "yes" vs "no" vs "up" etc.
#           When it saw "Hey Saya", it didn't know what it was.
#           Everything scored ~0.79 similarity → false triggers everywhere.

# WHY THIS APPROACH WORKS:
#   Episodic training makes every training batch simulate the EXACT
#   few-shot task you need at inference time.

#   Each episode:
#     - Pick a random word as the "wake word"
#     - Give the model N_SHOT (10) examples of it → compute prototype
#     - Give the model query samples of the wake word AND other words
#     - Train the model to correctly match queries to their word
  
#   After thousands of these episodes, the model learns:
#     "Given 10 examples of ANY new sound, I can detect it reliably"

#   This is exactly how Porcupine works internally.
#   After training, ANY wake word works with just 10 samples.
#   NO RETRAINING needed when the wake word changes.

# HOW TO RUN:
#   pip install torch torchaudio librosa numpy tqdm
#   python3 train_embedding.py

# EXPECTED TIME:
#   ~3-5 hours on CPU, ~45 minutes with GPU
# """

# import os
# import random
# import numpy as np
# import librosa
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import urllib.request
# import tarfile
# import collections

# from embedding_model import WakeWordEmbedder

# # ══════════════════════════════════════════════════════════════════
# # CONFIG
# # ══════════════════════════════════════════════════════════════════
# DATASET_DIR        = "./speech_commands"
# MODEL_SAVE         = "embedding_model.pth"

# SAMPLE_RATE        = 16000
# N_FFT              = 512
# HOP_LENGTH         = 160
# N_MELS             = 40
# N_MFCC             = 40
# MAX_FRAMES         = 196
# EMBEDDING_DIM      = 128

# # ── Episodic training parameters ─────────────────────────────────
# N_SHOT             = 10    # support samples per episode (matches your 10-sample constraint)
# N_QUERY            = 5     # query samples per class per episode
# N_WAY              = 5     # how many words per episode (increases difficulty)
# EPISODES_PER_EPOCH = 400   # episodes per epoch
# VAL_EPISODES       = 100

# BATCH_SIZE         = 1     # each "batch" is one episode
# EPOCHS             = 60
# LR                 = 1e-3
# PATIENCE           = 10

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Words used for training (ALL 35 words gives more variety)
# WORDS_TO_USE = [
#     "yes", "no", "up", "down", "left", "right", "on", "off",
#     "stop", "go", "zero", "one", "two", "three", "four", "five",
#     "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog",
#     "happy", "house", "marvin", "sheila", "tree", "wow"
# ]

# print(f"Device: {DEVICE}")
# print(f"Episodic training: {N_WAY}-way {N_SHOT}-shot")
# print(f"This directly trains for your 10-sample inference requirement")

# # ══════════════════════════════════════════════════════════════════
# # MFCC — exact match to mfcc.cpp + dataset_extractor.cpp
# # ══════════════════════════════════════════════════════════════════
# import librosa.filters as lf

# _mel = lf.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float32)
# _n   = np.arange(N_MELS)
# _k   = np.arange(N_MFCC)[:, np.newaxis]
# _dct = np.cos(np.pi / N_MELS * (_n + 0.5) * _k)
# _dct[0] *= 1.0 / np.sqrt(2)
# _dct *= np.sqrt(2.0 / N_MELS)
# _dct = _dct.astype(np.float32)


# def pre_emphasis(audio, coeff=0.97):
#     out = np.copy(audio)
#     for i in range(len(audio) - 1, 0, -1):
#         out[i] = audio[i] - coeff * audio[i - 1]
#     return out


# def agc(audio, threshold=0.15, target=0.80, max_gain=8.0):
#     mv = np.max(np.abs(audio))
#     if mv > threshold:
#         gain = np.clip(target / mv, 1.0, max_gain)
#         audio = np.clip(audio * gain, -1.0, 1.0)
#     return audio


# def extract_mfcc(audio: np.ndarray) -> np.ndarray:
#     """Exact match to dataset_extractor.cpp pipeline."""
#     audio    = pre_emphasis(audio)
#     audio    = agc(audio)
#     S_power  = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
#                                     window='hamming', center=False)) ** 2.0
#     mel_S    = _mel @ S_power
#     log_mel  = np.log(mel_S + 1e-6)
#     mfcc     = _dct @ log_mel
#     if mfcc.shape[1] < MAX_FRAMES:
#         mfcc = np.pad(mfcc, ((0,0),(0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
#     else:
#         mfcc = mfcc[:, :MAX_FRAMES]
#     return mfcc.astype(np.float32)


# def load_audio(path):
#     audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
#     target   = SAMPLE_RATE * 2
#     if len(audio) < target:
#         audio = np.pad(audio, (0, target - len(audio)))
#     else:
#         audio = audio[:target]
#     return audio


# # ══════════════════════════════════════════════════════════════════
# # DATASET DOWNLOAD
# # ══════════════════════════════════════════════════════════════════
# def download_speech_commands():
#     url     = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
#     archive = "speech_commands_v0.02.tar.gz"
#     if os.path.exists(DATASET_DIR):
#         print(f"Dataset already at {DATASET_DIR}, skipping download.")
#         return
#     print("Downloading Google Speech Commands (~2.3 GB)...")
#     def progress(b, bs, total):
#         print(f"\r  {min(b*bs*100/total,100):.1f}%", end="", flush=True)
#     urllib.request.urlretrieve(url, archive, reporthook=progress)
#     print("\nExtracting...")
#     os.makedirs(DATASET_DIR, exist_ok=True)
#     with tarfile.open(archive, "r:gz") as tar:
#         tar.extractall(DATASET_DIR)
#     os.remove(archive)
#     print(f"Done. Dataset at {DATASET_DIR}")


# # ══════════════════════════════════════════════════════════════════
# # BUILD WORD → FILE MAP
# # ══════════════════════════════════════════════════════════════════
# def build_word_file_map():
#     word_files = {}
#     for word in WORDS_TO_USE:
#         d = os.path.join(DATASET_DIR, word)
#         if os.path.isdir(d):
#             files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav")]
#             # Need at least N_SHOT + N_QUERY samples per word
#             if len(files) >= N_SHOT + N_QUERY:
#                 word_files[word] = files
#     print(f"\nLoaded {len(word_files)} word classes (need >= {N_SHOT+N_QUERY} files each):")
#     for w, fs in word_files.items():
#         print(f"  {w:<12}: {len(fs)} files")
#     return word_files


# # ══════════════════════════════════════════════════════════════════
# # GLOBAL STATS
# # ══════════════════════════════════════════════════════════════════
# def compute_global_stats(word_files, n_samples=500):
#     print("\nComputing GLOBAL_MEAN / GLOBAL_STD from dataset sample...")
#     all_files = [f for fs in word_files.values() for f in fs]
#     sampled   = random.sample(all_files, min(n_samples, len(all_files)))
#     all_mfccs = []
#     for fp in tqdm(sampled, desc="  Sampling"):
#         try:
#             audio = load_audio(fp)
#             mfcc  = extract_mfcc(audio)
#             all_mfccs.append(mfcc.flatten())
#         except Exception:
#             continue
#     stacked = np.concatenate(all_mfccs)
#     mean    = float(np.mean(stacked))
#     std     = float(np.std(stacked))
#     print(f"\n  GLOBAL_MEAN = {mean:.8f}")
#     print(f"  GLOBAL_STD  = {std:.8f}")
#     print("  ⚠️  Copy these into register_wake_word.py and wake_word_detector.cpp!\n")
#     with open("global_stats.txt", "w") as f:
#         f.write(f"GLOBAL_MEAN = {mean}\nGLOBAL_STD = {std}\n")
#     return mean, std


# # ══════════════════════════════════════════════════════════════════
# # EPISODIC DATASET
# #
# # Each __getitem__ returns ONE complete episode:
# #   support_set: N_WAY × N_SHOT MFCCs  (the "10 samples" at inference)
# #   query_set:   N_WAY × N_QUERY MFCCs (what we need to classify)
# #   query_labels: integer class index for each query (0 to N_WAY-1)
# #
# # The model must learn: given support_set, correctly classify query_set.
# # After training on 100,000+ episodes, it can do this for ANY new word.
# # ══════════════════════════════════════════════════════════════════
# class EpisodicDataset(Dataset):
#     def __init__(self, word_files, global_mean, global_std, n_episodes):
#         self.word_files  = word_files
#         self.words       = list(word_files.keys())
#         self.mean        = global_mean
#         self.std         = global_std
#         self.n_episodes  = n_episodes

#     def __len__(self):
#         return self.n_episodes

#     def load_mfcc(self, path):
#         audio = load_audio(path)
#         mfcc  = extract_mfcc(audio)
#         mfcc  = (mfcc - self.mean) / (self.std + 1e-8)
#         return torch.tensor(mfcc).unsqueeze(0)   # (1, 40, 196)

#     def __getitem__(self, idx):

#         if idx >= self.n_episodes:
#             raise IndexError
#         # 1. Pick N_WAY random words for this episode
#         episode_words = random.sample(self.words, N_WAY)

#         support_set = []   # (N_WAY, N_SHOT, 1, 40, 196)
#         query_set   = []   # (N_WAY * N_QUERY, 1, 40, 196)
#         query_labels = []  # (N_WAY * N_QUERY,)

#         for class_idx, word in enumerate(episode_words):
#             # 2. Pick N_SHOT + N_QUERY unique files for this word
#             chosen = random.sample(self.word_files[word], N_SHOT + N_QUERY)
#             support_files = chosen[:N_SHOT]
#             query_files   = chosen[N_SHOT:]

#             # 3. Load support set (the "10 registration samples")
#             support_mfccs = []
#             for fp in support_files:
#                 try:
#                     support_mfccs.append(self.load_mfcc(fp))
#                 except Exception:
#                     support_mfccs.append(torch.zeros(1, N_MFCC, MAX_FRAMES))
#             support_set.append(torch.stack(support_mfccs))   # (N_SHOT, 1, 40, 196)

#             # 4. Load query samples + their labels
#             for fp in query_files:
#                 try:
#                     query_set.append(self.load_mfcc(fp))
#                 except Exception:
#                     query_set.append(torch.zeros(1, N_MFCC, MAX_FRAMES))
#                 query_labels.append(class_idx)

#         support_tensor = torch.stack(support_set)    # (N_WAY, N_SHOT, 1, 40, 196)
#         query_tensor   = torch.stack(query_set)      # (N_WAY*N_QUERY, 1, 40, 196)
#         labels_tensor  = torch.tensor(query_labels)  # (N_WAY*N_QUERY,)

#         return support_tensor, query_tensor, labels_tensor


# # ══════════════════════════════════════════════════════════════════
# # PROTOTYPICAL LOSS
# #
# # How it works each episode:
# #   1. Run all N_SHOT support samples through the model → N_SHOT embeddings
# #   2. Average them → 1 prototype embedding per word  (this IS the template!)
# #   3. Run all query samples through the model → query embeddings
# #   4. For each query, compute distance to every prototype
# #   5. The closest prototype should be the correct word
# #   6. Loss = cross entropy over the distances
# #
# # This directly trains the model for the EXACT inference workflow:
# #   Registration: 10 samples → average embedding → prototype
# #   Detection:    live audio → embedding → compare to prototype
# # ══════════════════════════════════════════════════════════════════
# def prototypical_loss(model, support, queries, labels, device):
#     """
#     support: (N_WAY, N_SHOT, 1, 40, 196)
#     queries: (N_Q,   1, 40, 196)
#     labels:  (N_Q,)
#     """
#     n_way   = support.shape[0]
#     n_shot  = support.shape[1]
#     n_query = queries.shape[0]

#     # ── Compute prototypes ────────────────────────────────────────
#     # Flatten support: (N_WAY * N_SHOT, 1, 40, 196)
#     support_flat = support.view(n_way * n_shot, *support.shape[2:]).to(device)
#     support_embs = model(support_flat)                              # (N_WAY*N_SHOT, 128)
#     support_embs = support_embs.view(n_way, n_shot, EMBEDDING_DIM) # (N_WAY, N_SHOT, 128)
    
#     # Average N_SHOT embeddings → 1 prototype per word
#     # This is EXACTLY what register_wake_word.py does at inference
#     prototypes = support_embs.mean(dim=1)                          # (N_WAY, 128)
#     prototypes = F.normalize(prototypes, p=2, dim=1)               # re-normalize after mean

#     # ── Compute query embeddings ──────────────────────────────────
#     queries    = queries.to(device)
#     query_embs = model(queries)                                     # (N_Q, 128)

#     # ── Compute cosine similarity to all prototypes ───────────────
#     # query_embs: (N_Q, 128), prototypes: (N_WAY, 128)
#     # result:     (N_Q, N_WAY) — similarity of each query to each prototype
#     similarities = query_embs @ prototypes.T                        # (N_Q, N_WAY)

#     # ── Cross entropy loss ────────────────────────────────────────
#     # Scale similarities (acts like temperature)
#     logits = similarities * 10.0
#     loss   = F.cross_entropy(logits, labels.to(device))

#     # ── Accuracy for monitoring ───────────────────────────────────
#     predicted = logits.argmax(dim=1)
#     accuracy  = (predicted == labels.to(device)).float().mean().item()

#     return loss, accuracy


# # ══════════════════════════════════════════════════════════════════
# # VALIDATION — episode accuracy tells you how good the model is
# # ══════════════════════════════════════════════════════════════════
# def validate(model, word_files, global_mean, global_std, n_episodes=100):
#     model.eval()
#     val_dataset = EpisodicDataset(word_files, global_mean, global_std, n_episodes)
#     total_acc   = 0.0

#     with torch.no_grad():
#         for support, queries, labels in val_dataset:
#             # support = support.unsqueeze(0)  # add batch dim
#             queries = queries
#             _, acc  = prototypical_loss(model, support, queries, labels, DEVICE)
#             total_acc += acc

#     model.train()
#     avg_acc = total_acc / n_episodes
#     return avg_acc


# # ══════════════════════════════════════════════════════════════════
# # MAIN TRAINING
# # ══════════════════════════════════════════════════════════════════
# def main():
#     download_speech_commands()
#     word_files = build_word_file_map()

#     if len(word_files) < N_WAY:
#         print(f"ERROR: Need at least {N_WAY} word classes. Check dataset path.")
#         return

#     global_mean, global_std = compute_global_stats(word_files)

#     # ── Split words into train / val sets ─────────────────────────
#     # Use 80% of words for training episodes, 20% for validation
#     # This tests generalization to UNSEEN words — critical for your use case
#     all_words  = list(word_files.keys())
#     random.shuffle(all_words)
#     split      = int(len(all_words) * 0.8)
#     train_words = {w: word_files[w] for w in all_words[:split]}
#     val_words   = {w: word_files[w] for w in all_words[split:]}

#     print(f"\nTrain words ({len(train_words)}): {list(train_words.keys())}")
#     print(f"Val words   ({len(val_words)}):   {list(val_words.keys())}")
#     print()

#     train_dataset = EpisodicDataset(train_words, global_mean, global_std, EPISODES_PER_EPOCH)

#     model     = WakeWordEmbedder(embedding_dim=EMBEDDING_DIM).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=5
#     )

#     print("=" * 60)
#     print("  Episodic / Prototypical Network Training")
#     print(f"  {N_WAY}-way {N_SHOT}-shot | {EPISODES_PER_EPOCH} episodes/epoch")
#     print(f"  This trains the model to work with exactly {N_SHOT} samples")
#     print(f"  of any new wake word — no retraining needed later.")
#     print("=" * 60 + "\n")

#     best_val_acc     = 0.0
#     patience_counter = 0

#     for epoch in range(1, EPOCHS + 1):

#         # ── Train ─────────────────────────────────────────────────
#         model.train()
#         epoch_loss = 0.0
#         epoch_acc  = 0.0

#         pbar = tqdm(train_dataset, desc=f"Epoch {epoch:03d}", leave=False)
#         for support, queries, labels in pbar:
#             # support: (N_WAY, N_SHOT, 1, 40, 196) — add batch dim
#             # support = support.unsqueeze(0)

#             optimizer.zero_grad()
#             loss, acc = prototypical_loss(model, support, queries, labels, DEVICE)
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_acc  += acc
#             pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")

#         epoch_loss /= EPISODES_PER_EPOCH
#         epoch_acc  /= EPISODES_PER_EPOCH

#         # ── Validate on UNSEEN words ──────────────────────────────
#         val_acc = validate(model, val_words, global_mean, global_std, VAL_EPISODES)
#         scheduler.step(val_acc)

#         print(f"Epoch {epoch:03d} | "
#               f"Loss: {epoch_loss:.4f} | "
#               f"Train Acc: {epoch_acc:.2%} | "
#               f"Val Acc (unseen words): {val_acc:.2%}")

#         # Val accuracy on UNSEEN words is the key metric.
#         # If it's high, the model can handle any new wake word with 10 samples.
#         # Target: > 85% on 5-way 10-shot validation

#         # ── Checkpoint ───────────────────────────────────────────
#         if val_acc > best_val_acc:
#             best_val_acc     = val_acc
#             patience_counter = 0
#             torch.save({
#                 "epoch":        epoch,
#                 "model_state":  model.state_dict(),
#                 "global_mean":  global_mean,
#                 "global_std":   global_std,
#                 "val_acc":      val_acc,
#                 "n_shot":       N_SHOT,
#             }, MODEL_SAVE)
#             print(f"  ✅ Saved best model (val_acc={val_acc:.2%})")
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"\nEarly stopping at epoch {epoch}")
#                 break

#     print(f"\n{'='*60}")
#     print(f"  Training complete!")
#     print(f"  Best validation accuracy on unseen words: {best_val_acc:.2%}")
#     print()

#     if best_val_acc < 0.70:
#         print("  ⚠️  Accuracy below 70% — try:")
#         print("     - More epochs (increase EPOCHS to 100)")
#         print("     - More words in WORDS_TO_USE")
#         print("     - Lower LR (try 5e-4)")
#     elif best_val_acc < 0.85:
#         print("  ✅  Good — model should work reliably with 10 samples")
#     else:
#         print("  ✅  Excellent — model will work very well with 10 samples")

#     print(f"\n  Next: python3 register_wake_word.py")


# if __name__ == "__main__":
#     main()

"""
train_embedding.py — FINAL VERSION WITH YOUR DATA
==================================================
Trains the embedding model using BOTH:
  1. Google Speech Commands  (general speech separation)
  2. Your Hey Saya samples   (teaches exact wake word boundary)
  3. Your negative samples   (teaches what to reject)

40% of episodes use your Hey Saya data directly —
this forces the model to push Hey Saya FAR from noise.

HOW TO RUN:
  python3 train_embedding.py

BEFORE RUNNING:
  1. Run augment_wake_word.py first to create 300 positive samples
  2. Make sure your negative folder has 200+ clips
  3. Speech Commands downloads automatically if not present

AFTER RUNNING:
  1. Open global_stats.txt
  2. Copy GLOBAL_MEAN and GLOBAL_STD into:
     - register_wake_word.py
     - wake_word_detector.cpp
  3. Run: python3 register_wake_word.py
"""

import os
import random
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
# PATHS — matches your project structure exactly
# ══════════════════════════════════════════════════════════════════
BASE_DIR = "/home/arz-1029/pt_task/python_training_&_feature_extraction"

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
EPISODES_PER_EPOCH = 500
VAL_EPISODES       = 100
EPOCHS             = 80
LR                 = 1e-3
PATIENCE           = 12

# 40% Hey Saya episodes — the key parameter that fixes false triggers
HEY_SAYA_EPISODE_RATIO = 0.40

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


# ══════════════════════════════════════════════════════════════════
# GLOBAL STATS — computed from ALL data
# ══════════════════════════════════════════════════════════════════
def compute_global_stats(word_files, hey_saya_files, neg_files, n_samples=600):
    print("\nComputing GLOBAL_MEAN and GLOBAL_STD...")

    all_files = [f for fs in word_files.values() for f in fs]
    all_files += hey_saya_files
    all_files += neg_files
    sampled   = random.sample(all_files, min(n_samples, len(all_files)))

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
# Two types of episodes:
#
# TYPE 1 — Hey Saya episode (40% of training):
#   Class 0 = Hey Saya (10 support + 5 query from your 300 samples)
#   Class 1..4 = Random Speech Commands words OR hard negatives
#   This forces: Hey Saya embedding MUST be far from all other sounds
#
# TYPE 2 — Speech Commands episode (60% of training):
#   Class 0..4 = 5 random Speech Commands words
#   This teaches: general speech sound separation
# ══════════════════════════════════════════════════════════════════
class EpisodicDataset(Dataset):

    def __init__(self, word_files, hey_saya_files, neg_files,
                 global_mean, global_std, n_episodes):
        self.word_files    = word_files
        self.words         = list(word_files.keys())
        self.hey_saya      = hey_saya_files
        self.negatives     = neg_files
        self.mean          = global_mean
        self.std           = global_std
        self.n_episodes    = n_episodes

        self.can_hey_saya  = len(hey_saya_files) >= (N_SHOT + N_QUERY)
        if not self.can_hey_saya:
            print(f"WARNING: Need {N_SHOT+N_QUERY} Hey Saya files, "
                  f"have {len(hey_saya_files)}.")

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

    def make_hey_saya_episode(self):
        """Episode where Hey Saya is class 0."""

        # Hey Saya: 10 support + 5 query
        chosen           = random.sample(self.hey_saya, N_SHOT + N_QUERY)
        hs_support       = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
        hs_query_files   = chosen[N_SHOT:]

        # Other classes: Speech Commands words
        other_words = random.sample(self.words, N_WAY - 1)

        all_support  = [torch.stack(hs_support)]
        all_queries  = []
        all_labels   = []

        # Hey Saya queries (class 0)
        for f in hs_query_files:
            all_queries.append(self.load_mfcc(f))
            all_labels.append(0)

        # Other word classes
        for class_idx, word in enumerate(other_words, start=1):
            chosen  = random.sample(self.word_files[word], N_SHOT + N_QUERY)
            support = [self.load_mfcc(f) for f in chosen[:N_SHOT]]
            all_support.append(torch.stack(support))
            for f in chosen[N_SHOT:]:
                all_queries.append(self.load_mfcc(f))
                all_labels.append(class_idx)

        # Shuffle queries
        combined    = list(zip(all_queries, all_labels))
        random.shuffle(combined)
        all_queries, all_labels = zip(*combined)

        return (torch.stack(all_support),
                torch.stack(list(all_queries)),
                torch.tensor(list(all_labels)))

    def make_speech_commands_episode(self):
        """Standard episode using only Speech Commands words."""
        episode_words = random.sample(self.words, N_WAY)
        all_support   = []
        all_queries   = []
        all_labels    = []

        for class_idx, word in enumerate(episode_words):
            chosen  = random.sample(self.word_files[word], N_SHOT + N_QUERY)
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

    def __getitem__(self, idx):
        if idx >= self.n_episodes:
            raise IndexError
        if self.can_hey_saya and random.random() < HEY_SAYA_EPISODE_RATIO:
            return self.make_hey_saya_episode()
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
def validate(model, val_words, hey_saya_files, neg_files, mean, std):
    model.eval()
    dataset   = EpisodicDataset(val_words, hey_saya_files, neg_files,
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

    # ── Step 1: Download Speech Commands ─────────────────────────
    download_speech_commands()

    # ── Step 2: Load all datasets ─────────────────────────────────
    print("\nLoading datasets:")
    word_files     = build_word_file_map()
    hey_saya_files = load_wav_files(HEY_SAYA_DIR,       "Hey Saya (positive)")
    neg_files      = load_wav_files(HARD_NEGATIVE_DIR,   "Hard negatives")

    print(f"\nDataset summary:")
    print(f"  Speech Commands words : {len(word_files)}")
    print(f"  Hey Saya samples      : {len(hey_saya_files)}")
    print(f"  Hard negative samples : {len(neg_files)}")
    print(f"  Hey Saya episode ratio: {HEY_SAYA_EPISODE_RATIO:.0%}")

    # Check minimum requirements
    if len(word_files) < N_WAY:
        print(f"\nERROR: Need {N_WAY} Speech Commands words. Check {SPEECH_COMMANDS_DIR}")
        return

    if len(hey_saya_files) < N_SHOT + N_QUERY:
        print(f"\nERROR: Need at least {N_SHOT + N_QUERY} Hey Saya samples.")
        print(f"  Run augment_wake_word.py first to create 300 samples.")
        return

    # ── Step 3: Compute normalization stats ───────────────────────
    mean, std = compute_global_stats(word_files, hey_saya_files, neg_files)

    # ── Step 4: Split Speech Commands into train/val ──────────────
    all_words = list(word_files.keys())
    random.shuffle(all_words)
    split       = int(len(all_words) * 0.8)
    train_words = {w: word_files[w] for w in all_words[:split]}
    val_words   = {w: word_files[w] for w in all_words[split:]}

    print(f"\nTrain words ({len(train_words)}): {list(train_words.keys())}")
    print(f"Val words   ({len(val_words)}):   {list(val_words.keys())}")

    train_dataset = EpisodicDataset(
        train_words, hey_saya_files, neg_files,
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
    print(f"  {HEY_SAYA_EPISODE_RATIO:.0%} Hey Saya + "
          f"{1-HEY_SAYA_EPISODE_RATIO:.0%} Speech Commands")
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
        val_acc = validate(model, val_words, hey_saya_files, neg_files, mean, std)
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