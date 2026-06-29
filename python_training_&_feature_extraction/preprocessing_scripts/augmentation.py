# # ==============================================================================
# # Generates a diverse 2000-sample positive dataset from just 10 real recordings
# # of "Hey Saya" using layered audio augmentation.
# #
# # Install dependencies:
# #   pip install audiomentations soundfile librosa numpy
# #
# # Folder structure expected:
# #   positive/original/    ← put your 10 real WAV recordings here
# #
# # Output:
# #   positive/augmented/   ← 2000 augmented WAV files
# #   dataset/train/positive/
# #   dataset/train/negative/
# #   dataset/test/positive/
# #   dataset/test/negative/
# #
# # Usage:
# #   python augmentation.py
# # ==============================================================================

# import os
# import random
# import numpy as np
# import soundfile as sf
# import librosa
# from audiomentations import (
#     Compose,
#     AddGaussianNoise,
#     TimeStretch,
#     PitchShift,
#     RoomSimulator,
#     LoudnessNormalization,
#     BandPassFilter,
#     HighPassFilter,
#     LowPassFilter,
#     Shift,
#     Gain,
# )
# import shutil

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# SAMPLE_RATE        = 16000     # Must match your MFCC pipeline
# TARGET_DURATION    = 2.0       # seconds — pad/trim all audio to this length
# SAMPLES_PER_FILE   = 200       # 10 originals × 200 = 2000 total positive samples
# RANDOM_SEED        = 42

# # Paths
# ORIGINAL_DIR  = "/home/arz-1029/wake_word_detection/dataset_preprocessing/raw_clean_few_shot"    # Your 10 real recordings go here
# AUGMENTED_DIR = "positive/augmented"   # All 2000 augmented samples go here

# # Final dataset split paths
# TRAIN_POS_DIR = "dataset/train/positive"
# TRAIN_NEG_DIR = "dataset/train/negative"
# TEST_POS_DIR  = "dataset/test/positive"
# TEST_NEG_DIR  = "dataset/test/negative"

# # Negative samples source directory
# # Put your 2000 negative WAV files here (Google Speech Commands, Common Voice etc.)
# NEGATIVE_SOURCE_DIR = "/home/arz-1029/wake_word_detection/dataset_preprocessing/processed_audio/dataset/negative"

# # Train/test split ratio for positives (split by ORIGINAL recording, not augmented)
# # 8 originals → train augmented pool
# # 2 originals → test augmented pool
# TRAIN_ORIGINALS = 8
# TEST_ORIGINALS  = 2

# # Train/test counts
# TRAIN_POS_COUNT = 1600   # from 8 originals × 200 augmentations
# TEST_POS_COUNT  = 400    # from 2 originals × 200 augmentations
# TRAIN_NEG_COUNT = 1600
# TEST_NEG_COUNT  = 400

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)


# # ==============================================================================
# # AUDIO UTILITIES
# # ==============================================================================

# def load_audio(path: str) -> np.ndarray:
#     """Load WAV file, convert to mono float32 at 16kHz."""
#     audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
#     return audio.astype(np.float32)


# def pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
#     """
#     Pad with silence or trim to exactly target_samples.
#     Padding is split: some silence before, some after — simulates
#     the wake word appearing at different positions in the window.
#     """
#     if len(audio) >= target_samples:
#         # Trim: randomly pick start point so the word isn't always at frame 0
#         max_start = len(audio) - target_samples
#         start = random.randint(0, max_start)
#         return audio[start : start + target_samples]
#     else:
#         # Pad: distribute silence randomly before and after
#         pad_total = target_samples - len(audio)
#         pad_before = random.randint(0, pad_total)
#         pad_after  = pad_total - pad_before
#         return np.pad(audio, (pad_before, pad_after), mode='constant')


# def save_audio(audio: np.ndarray, path: str):
#     """Save float32 audio as 16-bit PCM WAV at 16kHz."""
#     # Clip to prevent clipping distortion before int16 conversion
#     audio = np.clip(audio, -1.0, 1.0)
#     sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


# # ==============================================================================
# # AUGMENTATION PIPELINES
# # Three levels: mild, moderate, heavy
# # Each original recording gets a mix of all three levels so the dataset
# # covers both easy (clean) and hard (very noisy) conditions.
# # ==============================================================================

# def make_mild_augmenter():
#     """
#     Mild augmentation — simulates slightly different recording conditions.
#     Good for: different mic placement, slightly different speaker energy.
#     """
#     return Compose([
#         # Small volume variation ±4dB
#         Gain(min_gain_db=-4, max_gain_db=4, p=0.8),

#         # Very subtle pitch shift ±1 semitone (same speaker, different energy)
#         PitchShift(min_semitones=-1, max_semitones=1, p=0.7),

#         # Slight speed change ±5%
#         TimeStretch(min_rate=0.95, max_rate=1.05, p=0.7),

#         # Very light background noise (quiet room)
#         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.5),

#         # Small time shift — word appears slightly earlier/later in window
#         Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
#     ])


# def make_moderate_augmenter():
#     """
#     Moderate augmentation — simulates different speakers and environments.
#     Good for: different family members, different rooms in the house.
#     """
#     return Compose([
#         # Larger volume variation ±8dB
#         Gain(min_gain_db=-8, max_gain_db=8, p=0.9),

#         # Moderate pitch shift ±3 semitones (male↔female voice range)
#         PitchShift(min_semitones=-3, max_semitones=3, p=0.8),

#         # Speed variation ±15% — fast vs slow speakers
#         TimeStretch(min_rate=0.85, max_rate=1.15, p=0.8),

#         # Moderate noise (office, home background)
#         AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=0.7),

#         # FIXED: Explicit X, Y, Z dimensions for a medium-sized room
#         RoomSimulator(
#             min_size_x=3.0, max_size_x=6.0,
#             min_size_y=3.0, max_size_y=6.0,
#             min_size_z=2.4, max_size_z=3.0,
#             min_absorption_value=0.15, max_absorption_value=0.3,
#             p=0.5
#         ),

#         # Frequency coloring — simulates different microphones
#         HighPassFilter(min_cutoff_freq=80,  max_cutoff_freq=300,  p=0.3),
#         LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=7500, p=0.3),

#         # Time shift
#         Shift(min_shift=-0.2, max_shift=0.2, p=0.6),
#     ])


# def make_heavy_augmenter():
#     """
#     Heavy augmentation — simulates difficult real-world conditions.
#     Good for: noisy street, TV in background, far-field microphone.
#     Indian home environments are often quite noisy — this is critical.
#     """
#     return Compose([
#         # Wide volume range — very quiet whisper to loud speech
#         Gain(min_gain_db=-12, max_gain_db=6, p=0.9),

#         # Full pitch range — covers diverse Indian accents and ages
#         PitchShift(min_semitones=-4, max_semitones=4, p=0.9),

#         # Wide speed range — rushing vs slow speaker
#         TimeStretch(min_rate=0.75, max_rate=1.25, p=0.9),

#         # Heavy noise — street, kitchen, crowd
#         AddGaussianNoise(min_amplitude=0.02, max_amplitude=0.06, p=0.8),

#         # FIXED: Explicit X, Y, Z dimensions for large echoing rooms
#         RoomSimulator(
#             min_size_x=4.0, max_size_x=10.0,
#             min_size_y=4.0, max_size_y=10.0,
#             min_size_z=2.5, max_size_z=4.0,
#             min_absorption_value=0.05, max_absorption_value=0.15,
#             p=0.7
#         ),

#         # Telephone/low quality mic simulation
#         BandPassFilter(
#             min_center_freq=500,
#             max_center_freq=3000,
#             min_bandwidth_fraction=0.8,
#             max_bandwidth_fraction=1.5,
#             p=0.3
#         ),

#         # Large time shift
#         Shift(min_shift=-0.3, max_shift=0.3, p=0.7),

#         # Loudness normalization — simulates AGC in different devices
#         LoudnessNormalization(min_lufs=-31, max_lufs=-13, p=0.4),
#     ])

# # ==============================================================================
# # AUGMENTATION DISTRIBUTION
# # For each original recording, distribute 200 augmentations across three levels:
# #   60 mild      (30%) — easy cases, model learns the clean pattern
# #   80 moderate  (40%) — medium cases, core of the training
# #   60 heavy     (30%) — hard cases, robustness training
# # ==============================================================================

# AUGMENTATION_PLAN = [
#     ("mild",     60,  make_mild_augmenter),
#     ("moderate", 80,  make_moderate_augmenter),
#     ("heavy",    60,  make_heavy_augmenter),
# ]


# # ==============================================================================
# # MAIN AUGMENTATION FUNCTION
# # ==============================================================================

# def augment_file(original_path: str, output_dir: str, file_prefix: str, count: int):
#     """
#     Generate `count` augmented versions of a single original recording.
#     Distributes across mild/moderate/heavy levels per AUGMENTATION_PLAN.
#     """
#     audio = load_audio(original_path)
#     target_samples = int(TARGET_DURATION * SAMPLE_RATE)  # 32000 samples = 2 seconds

#     generated = 0
#     for level_name, level_count, augmenter_factory in AUGMENTATION_PLAN:
#         actual_count = min(level_count, count - generated)
#         if actual_count <= 0:
#             break

#         augmenter = augmenter_factory()

#         for i in range(actual_count):
#             try:
#                 # Apply augmentation
#                 aug_audio = augmenter(samples=audio.copy(), sample_rate=SAMPLE_RATE)

#                 # Pad or trim to fixed length
#                 aug_audio = pad_or_trim(aug_audio, target_samples)

#                 # Save
#                 out_filename = f"{file_prefix}_{level_name}_{i:04d}.wav"
#                 out_path = os.path.join(output_dir, out_filename)
#                 save_audio(aug_audio, out_path)

#                 generated += 1

#             except Exception as e:
#                 print(f"  [WARNING] Augmentation failed for {level_name} sample {i}: {e}")
#                 continue

#     return generated


# # ==============================================================================
# # NEGATIVE SAMPLE PREPARATION
# # ==============================================================================

# def prepare_negatives(source_dir: str, train_dir: str, test_dir: str,
#                        train_count: int, test_count: int):
#     """
#     Copy and prepare negative samples into train/test splits.
#     Pads/trims all negatives to the same TARGET_DURATION as positives.
#     """
#     if not os.path.exists(source_dir):
#         print(f"\n[WARNING] Negative source directory not found: {source_dir}")
#         print("  Please download Google Speech Commands dataset and put WAV files there.")
#         print("  Skipping negative preparation.")
#         return

#     all_negatives = [
#         os.path.join(source_dir, f)
#         for f in os.listdir(source_dir)
#         if f.lower().endswith('.wav')
#     ]

#     if len(all_negatives) < train_count + test_count:
#         print(f"[WARNING] Only {len(all_negatives)} negative files found.")
#         print(f"  Need {train_count + test_count}. Using all available.")

#     random.shuffle(all_negatives)
#     target_samples = int(TARGET_DURATION * SAMPLE_RATE)

#     total_needed = min(len(all_negatives), train_count + test_count)
#     train_files = all_negatives[:train_count]
#     test_files  = all_negatives[train_count : train_count + test_count]

#     print(f"\n[Negatives] Preparing {len(train_files)} train, {len(test_files)} test...")

#     for i, (files, out_dir) in enumerate([(train_files, train_dir), (test_files, test_dir)]):
#         split_name = "train" if i == 0 else "test"
#         for j, src_path in enumerate(files):
#             try:
#                 audio = load_audio(src_path)
#                 audio = pad_or_trim(audio, target_samples)
#                 out_name = f"neg_{split_name}_{j:05d}.wav"
#                 save_audio(audio, os.path.join(out_dir, out_name))
#             except Exception as e:
#                 print(f"  [WARNING] Failed to process negative {src_path}: {e}")

#             if (j + 1) % 200 == 0:
#                 print(f"  {split_name}: {j+1}/{len(files)} done...")

#     print("[Negatives] Done.")


# # ==============================================================================
# # DATASET SPLIT
# # ==============================================================================

# def split_augmented_positives(augmented_dir: str, original_files: list,
#                                train_dir: str, test_dir: str,
#                                train_originals: int):
#     """
#     Split augmented positives into train/test BY ORIGINAL RECORDING.
#     This prevents data leakage where augmented copies of the same
#     original appear in both train and test sets.
#     """
#     train_orig_names = [os.path.splitext(os.path.basename(f))[0]
#                         for f in original_files[:train_originals]]
#     test_orig_names  = [os.path.splitext(os.path.basename(f))[0]
#                         for f in original_files[train_originals:]]

#     all_augmented = [
#         f for f in os.listdir(augmented_dir)
#         if f.lower().endswith('.wav')
#     ]

#     train_count = 0
#     test_count  = 0

#     for filename in all_augmented:
#         src = os.path.join(augmented_dir, filename)

#         # Check which original this augmented file came from
#         is_train = any(filename.startswith(name) for name in train_orig_names)
#         is_test  = any(filename.startswith(name) for name in test_orig_names)

#         if is_train:
#             shutil.copy2(src, os.path.join(train_dir, filename))
#             train_count += 1
#         elif is_test:
#             shutil.copy2(src, os.path.join(test_dir, filename))
#             test_count += 1

#     return train_count, test_count


# # ==============================================================================
# # ENTRY POINT
# # ==============================================================================

# def main():
#     print("=" * 60)
#     print("  Hey Saya — Dataset Augmentation Pipeline")
#     print("  10 recordings → 2000 augmented positive samples")
#     print("=" * 60)

#     # --- Create output directories ---
#     for d in [AUGMENTED_DIR, TRAIN_POS_DIR, TRAIN_NEG_DIR,
#               TEST_POS_DIR, TEST_NEG_DIR]:
#         os.makedirs(d, exist_ok=True)

#     # --- Load original recordings ---
#     original_files = sorted([
#         os.path.join(ORIGINAL_DIR, f)
#         for f in os.listdir(ORIGINAL_DIR)
#         if f.lower().endswith('.wav')
#     ])

#     if len(original_files) == 0:
#         print(f"\n[ERROR] No WAV files found in '{ORIGINAL_DIR}'")
#         print("  Please put your 10 recorded 'Hey Saya' WAV files there.")
#         return

#     print(f"\nFound {len(original_files)} original recordings:")
#     for f in original_files:
#         audio = load_audio(f)
#         duration = len(audio) / SAMPLE_RATE
#         print(f"  {os.path.basename(f):40s}  {duration:.2f}s")

#     if len(original_files) < 10:
#         print(f"\n[WARNING] Only {len(original_files)} files found. Recommended: 10.")
#         print("  Continuing anyway — more originals = better diversity.")

#     # --- Augment each original ---
#     total_generated = 0
#     print(f"\n[Augmenting] Generating {SAMPLES_PER_FILE} samples per recording...")
#     print(f"  Plan: 60 mild + 80 moderate + 60 heavy = {SAMPLES_PER_FILE} per file\n")

#     for idx, orig_path in enumerate(original_files):
#         basename = os.path.splitext(os.path.basename(orig_path))[0]
#         print(f"  [{idx+1}/{len(original_files)}] {basename}...", end=" ", flush=True)

#         count = augment_file(
#             original_path=orig_path,
#             output_dir=AUGMENTED_DIR,
#             file_prefix=basename,
#             count=SAMPLES_PER_FILE
#         )
#         total_generated += count
#         print(f"{count} samples generated.")

#     print(f"\n✅ Total augmented positive samples: {total_generated}")

#     # --- Split into train/test by original recording ---
#     print(f"\n[Splitting] Train/test split (by original recording)...")
#     print(f"  Train originals: {TRAIN_ORIGINALS} files → ~{TRAIN_ORIGINALS * SAMPLES_PER_FILE} samples")
#     print(f"  Test originals:  {len(original_files) - TRAIN_ORIGINALS} files → ~{(len(original_files) - TRAIN_ORIGINALS) * SAMPLES_PER_FILE} samples")

#     train_pos, test_pos = split_augmented_positives(
#         augmented_dir=AUGMENTED_DIR,
#         original_files=original_files,
#         train_dir=TRAIN_POS_DIR,
#         test_dir=TEST_POS_DIR,
#         train_originals=TRAIN_ORIGINALS
#     )
#     print(f"  Positive train: {train_pos} | Positive test: {test_pos}")

#     # --- Prepare negative samples ---
#     prepare_negatives(
#         source_dir=NEGATIVE_SOURCE_DIR,
#         train_dir=TRAIN_NEG_DIR,
#         test_dir=TEST_NEG_DIR,
#         train_count=TRAIN_NEG_COUNT,
#         test_count=TEST_NEG_COUNT
#     )

#     # --- Final summary ---
#     print("\n" + "=" * 60)
#     print("  DATASET READY")
#     print("=" * 60)
#     print(f"  dataset/train/positive/  →  {train_pos} files")
#     print(f"  dataset/train/negative/  →  {TRAIN_NEG_COUNT} files")
#     print(f"  dataset/test/positive/   →  {test_pos} files")
#     print(f"  dataset/test/negative/   →  {TEST_NEG_COUNT} files")
#     print()
#     print("  Next steps:")
#     print("  1. Compile dataset_extractor and convert all WAVs to CSVs")
#     print("  2. Run build_cpp_data.py to get GLOBAL_MEAN and GLOBAL_STD")
#     print("  3. Run train_pytorch_cnn.py to train the model")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()


# ==============================================================================
# augment_dataset.py  — FIXED VERSION
# Generates a diverse 2000-sample positive dataset from just 10 real recordings.
#
# Install dependencies:
#   pip install audiomentations soundfile librosa numpy scipy
#
# Folder structure expected:
#   positive/original/    ← put your 10 real WAV recordings here
#   negative/raw/         ← put your 2000 negative WAVs here
#
# Output:
#   positive/augmented/
#   dataset/train/positive/
#   dataset/train/negative/
#   dataset/test/positive/
#   dataset/test/negative/
#
# Usage:
#   python3 augment_dataset.py
# ==============================================================================

# import os
# import random
# import shutil
# import numpy as np
# import soundfile as sf
# import librosa
# from audiomentations import (
#     Compose,
#     AddGaussianNoise,
#     TimeStretch,
#     PitchShift,
#     LoudnessNormalization,
#     BandPassFilter,
#     HighPassFilter,
#     LowPassFilter,
#     Shift,
#     Gain,
# )

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================
# SAMPLE_RATE      = 16000
# TARGET_DURATION  = 2.0       # seconds — all audio padded/trimmed to this
# SAMPLES_PER_FILE = 200       # 10 originals × 200 = 2000 total positives
# RANDOM_SEED      = 42

# ORIGINAL_DIR  = "/home/arz-1029/wake_word_detection/dataset_preprocessing/raw_clean_few_shot"
# AUGMENTED_DIR = "positive/augmented"

# TRAIN_POS_DIR = "dataset/train/positive"
# TRAIN_NEG_DIR = "dataset/train/negative"
# TEST_POS_DIR  = "dataset/test/positive"
# TEST_NEG_DIR  = "dataset/test/negative"

# NEGATIVE_SOURCE_DIR = "/home/arz-1029/wake_word_detection/dataset_preprocessing/processed_audio/dataset/negative"

# TRAIN_ORIGINALS = 8    # 8 of 10 originals → train
# TEST_ORIGINALS  = 2    # 2 of 10 originals → test

# TRAIN_NEG_COUNT = 1600
# TEST_NEG_COUNT  = 400

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)


# # ==============================================================================
# # AUDIO UTILITIES
# # ==============================================================================

# def load_audio(path: str) -> np.ndarray:
#     audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
#     return audio.astype(np.float32)


# def pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
#     """
#     Pad or trim to exactly target_samples.
#     Padding position is random — wake word appears at different positions
#     in the window, making the model position-invariant.
#     """
#     if len(audio) >= target_samples:
#         max_start = len(audio) - target_samples
#         start = random.randint(0, max_start)
#         return audio[start: start + target_samples]
#     else:
#         pad_total  = target_samples - len(audio)
#         pad_before = random.randint(0, pad_total)
#         pad_after  = pad_total - pad_before
#         return np.pad(audio, (pad_before, pad_after), mode='constant')


# def save_audio(audio: np.ndarray, path: str):
#     audio = np.clip(audio, -1.0, 1.0)
#     sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


# # ==============================================================================
# # MANUAL ROOM REVERB
# # Replaces RoomSimulator which requires pyroomacoustics (optional install).
# # Simulates room acoustics by adding delayed, decayed echo copies of the signal.
# # ==============================================================================

# def apply_room_reverb(audio: np.ndarray, sample_rate: int,
#                       room_size: str = 'medium') -> np.ndarray:
#     """
#     Simulate room acoustics without any extra dependency.

#     room_size:
#       'small'  — bedroom, small office     (short reverb)
#       'medium' — living room, classroom    (medium reverb)
#       'large'  — hall, outdoor, factory    (long reverb)
#     """
#     configs = {
#         'small':  {'base_delay': 0.015, 'decay': 0.25, 'n_echoes': 3},
#         'medium': {'base_delay': 0.035, 'decay': 0.45, 'n_echoes': 5},
#         'large':  {'base_delay': 0.070, 'decay': 0.65, 'n_echoes': 8},
#     }
#     cfg = configs.get(room_size, configs['medium'])
#     result = audio.copy()

#     for i in range(1, cfg['n_echoes'] + 1):
#         delay_samples = int(cfg['base_delay'] * i * sample_rate)
#         amplitude     = cfg['decay'] ** i
#         if delay_samples < len(audio):
#             echo = np.zeros_like(audio)
#             echo[delay_samples:] = audio[:-delay_samples] * amplitude
#             result = result + echo

#     # Normalize back to original peak level to avoid clipping
#     original_peak = np.max(np.abs(audio))
#     result_peak   = np.max(np.abs(result))
#     if result_peak > 0 and original_peak > 0:
#         result = result * (original_peak / result_peak)

#     return result.astype(np.float32)


# def random_reverb(audio: np.ndarray, sample_rate: int, p: float = 0.5) -> np.ndarray:
#     """Apply a randomly chosen room size reverb with probability p."""
#     if random.random() < p:
#         room = random.choice(['small', 'medium', 'large'])
#         return apply_room_reverb(audio, sample_rate, room)
#     return audio


# # ==============================================================================
# # THREE-LEVEL AUGMENTATION PIPELINES
# #
# # Distribution per original (total = 200 samples):
# #   60  mild      (30%) — easy clean conditions
# #   80  moderate  (40%) — medium, core of training
# #   60  heavy     (30%) — hard, noisy real-world conditions
# # ==============================================================================

# def make_mild_augmenter():
#     """Simulates same speaker on different day, mic slightly moved."""
#     return Compose([
#         Gain(min_gain_db=-4, max_gain_db=4, p=0.8),
#         PitchShift(min_semitones=-1, max_semitones=1, p=0.7),
#         TimeStretch(min_rate=0.95, max_rate=1.05, p=0.7),
#         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.5),
#         Shift(min_shift=-0.1, max_shift=0.1, shift_unit='seconds', p=0.5),
#     ])


# def make_moderate_augmenter():
#     """Simulates different family members, different rooms in the house."""
#     return Compose([
#         Gain(min_gain_db=-8, max_gain_db=8, p=0.9),
#         PitchShift(min_semitones=-3, max_semitones=3, p=0.8),   # male↔female range
#         TimeStretch(min_rate=0.85, max_rate=1.15, p=0.8),
#         AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=0.7),
#         HighPassFilter(min_cutoff_freq=80,  max_cutoff_freq=300,  p=0.3),
#         LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=7500, p=0.3),
#         Shift(min_shift=-0.2, max_shift=0.2, shift_unit='seconds', p=0.6),
#     ])


# def make_heavy_augmenter():
#     """Simulates street noise, TV background, far-field mic, phone call quality."""
#     return Compose([
#         Gain(min_gain_db=-12, max_gain_db=6, p=0.9),
#         PitchShift(min_semitones=-4, max_semitones=4, p=0.9),   # diverse Indian accents
#         TimeStretch(min_rate=0.75, max_rate=1.25, p=0.9),        # rushed vs slow speaker
#         AddGaussianNoise(min_amplitude=0.02, max_amplitude=0.06, p=0.8),
#         BandPassFilter(                                            # phone call quality
#             min_center_freq=500,
#             max_center_freq=3000,
#             min_bandwidth_fraction=0.8,
#             max_bandwidth_fraction=1.5,
#             p=0.3
#         ),
#         Shift(min_shift=-0.3, max_shift=0.3, shift_unit='seconds', p=0.7),
#         LoudnessNormalization(min_lufs=-31, max_lufs=-13, p=0.4),
#     ])


# AUGMENTATION_PLAN = [
#     ("mild",     60, make_mild_augmenter),
#     ("moderate", 80, make_moderate_augmenter),
#     ("heavy",    60, make_heavy_augmenter),
# ]


# # ==============================================================================
# # CORE AUGMENTATION FUNCTION
# # ==============================================================================

# def augment_file(original_path: str, output_dir: str,
#                  file_prefix: str, count: int) -> int:
#     audio = load_audio(original_path)
#     target_samples = int(TARGET_DURATION * SAMPLE_RATE)
#     generated = 0

#     for level_name, level_count, augmenter_factory in AUGMENTATION_PLAN:
#         actual_count = min(level_count, count - generated)
#         if actual_count <= 0:
#             break

#         augmenter    = augmenter_factory()
#         reverb_probs = {'mild': 0.3, 'moderate': 0.5, 'heavy': 0.7}
#         reverb_p     = reverb_probs[level_name]

#         for i in range(actual_count):
#             try:
#                 aug_audio = augmenter(samples=audio.copy(), sample_rate=SAMPLE_RATE)
#                 aug_audio = random_reverb(aug_audio, SAMPLE_RATE, p=reverb_p)
#                 aug_audio = pad_or_trim(aug_audio, target_samples)
#                 out_name  = f"{file_prefix}_{level_name}_{i:04d}.wav"
#                 save_audio(aug_audio, os.path.join(output_dir, out_name))
#                 generated += 1
#             except Exception as e:
#                 print(f"\n  [WARNING] Skipped {level_name} sample {i}: {e}")

#     return generated


# # ==============================================================================
# # TRAIN / TEST SPLIT  — split by ORIGINAL recording to prevent data leakage
# # ==============================================================================

# def split_augmented_positives(augmented_dir, original_files,
#                                train_dir, test_dir, train_originals):
#     train_prefixes = [
#         os.path.splitext(os.path.basename(f))[0]
#         for f in original_files[:train_originals]
#     ]
#     test_prefixes = [
#         os.path.splitext(os.path.basename(f))[0]
#         for f in original_files[train_originals:]
#     ]

#     train_count = test_count = 0
#     for filename in os.listdir(augmented_dir):
#         if not filename.lower().endswith('.wav'):
#             continue
#         src = os.path.join(augmented_dir, filename)
#         if any(filename.startswith(p) for p in train_prefixes):
#             shutil.copy2(src, os.path.join(train_dir, filename))
#             train_count += 1
#         elif any(filename.startswith(p) for p in test_prefixes):
#             shutil.copy2(src, os.path.join(test_dir, filename))
#             test_count += 1

#     return train_count, test_count


# # ==============================================================================
# # NEGATIVE SAMPLE PREPARATION
# # ==============================================================================

# def prepare_negatives(source_dir, train_dir, test_dir, train_count, test_count):
#     if not os.path.exists(source_dir):
#         print(f"\n  [WARNING] '{source_dir}' not found. Skipping negatives.")
#         print("  Download Google Speech Commands and put WAVs there.")
#         return

#     all_neg = [
#         os.path.join(source_dir, f)
#         for f in os.listdir(source_dir)
#         if f.lower().endswith('.wav')
#     ]
#     random.shuffle(all_neg)

#     total_needed = train_count + test_count
#     if len(all_neg) < total_needed:
#         print(f"  [WARNING] Only {len(all_neg)} negatives, need {total_needed}.")

#     target_samples = int(TARGET_DURATION * SAMPLE_RATE)
#     splits = [
#         (all_neg[:train_count],             train_dir, "train"),
#         (all_neg[train_count:total_needed], test_dir,  "test"),
#     ]
#     for files, out_dir, name in splits:
#         print(f"  Copying {len(files)} {name} negatives...", end=" ", flush=True)
#         for j, src in enumerate(files):
#             try:
#                 audio = load_audio(src)
#                 audio = pad_or_trim(audio, target_samples)
#                 save_audio(audio, os.path.join(out_dir, f"neg_{name}_{j:05d}.wav"))
#             except Exception as e:
#                 print(f"\n  [WARNING] Skipping {src}: {e}")
#         print("done.")


# # ==============================================================================
# # MAIN
# # ==============================================================================

# def main():
#     print("=" * 60)
#     print("  Hey Saya — Dataset Augmentation Pipeline")
#     print("  10 recordings → 2000 augmented positive samples")
#     print("=" * 60)

#     for d in [AUGMENTED_DIR, TRAIN_POS_DIR, TRAIN_NEG_DIR, TEST_POS_DIR, TEST_NEG_DIR]:
#         os.makedirs(d, exist_ok=True)

#     original_files = sorted([
#         os.path.join(ORIGINAL_DIR, f)
#         for f in os.listdir(ORIGINAL_DIR)
#         if f.lower().endswith('.wav')
#     ])

#     if not original_files:
#         print(f"\n[ERROR] No WAV files in '{ORIGINAL_DIR}'. Add recordings and rerun.")
#         return

#     print(f"\nFound {len(original_files)} original recordings:")
#     for f in original_files:
#         dur = len(load_audio(f)) / SAMPLE_RATE
#         print(f"  {os.path.basename(f):40s}  {dur:.2f}s")

#     if len(original_files) < 10:
#         print(f"\n  [WARNING] Only {len(original_files)} recordings. More = better diversity.")

#     total_generated = 0
#     print(f"\n[Augmenting] {SAMPLES_PER_FILE} samples per recording...")
#     print(f"  60 mild + 80 moderate + 60 heavy = {SAMPLES_PER_FILE} per file")
#     print(f"  Manual room reverb included (no pyroomacoustics needed)\n")

#     for idx, orig_path in enumerate(original_files):
#         basename = os.path.splitext(os.path.basename(orig_path))[0]
#         print(f"  [{idx+1:2d}/{len(original_files)}] {basename}...", end=" ", flush=True)
#         n = augment_file(orig_path, AUGMENTED_DIR, basename, SAMPLES_PER_FILE)
#         total_generated += n
#         print(f"{n} samples generated.")

#     print(f"\n  Total augmented positives: {total_generated}")

#     print(f"\n[Splitting] Train/test by original recording (no data leakage)...")
#     train_pos, test_pos = split_augmented_positives(
#         AUGMENTED_DIR, original_files,
#         TRAIN_POS_DIR, TEST_POS_DIR, TRAIN_ORIGINALS
#     )
#     print(f"  Train positives: {train_pos}")
#     print(f"  Test positives:  {test_pos}")

#     print(f"\n[Negatives] Preparing train/test splits...")
#     prepare_negatives(NEGATIVE_SOURCE_DIR, TRAIN_NEG_DIR, TEST_NEG_DIR,
#                       TRAIN_NEG_COUNT, TEST_NEG_COUNT)

#     print("\n" + "=" * 60)
#     print("  DATASET READY")
#     print("=" * 60)
#     print(f"  dataset/train/positive/  →  {train_pos} files")
#     print(f"  dataset/train/negative/  →  up to {TRAIN_NEG_COUNT} files")
#     print(f"  dataset/test/positive/   →  {test_pos} files")
#     print(f"  dataset/test/negative/   →  up to {TEST_NEG_COUNT} files")
#     print()
#     print("  Next steps:")
#     print("  1. Compile and run dataset_extractor on all WAVs → CSVs")
#     print("  2. Run build_cpp_data.py → get GLOBAL_MEAN and GLOBAL_STD")
#     print("  3. Paste those values into train_pytorch_cnn.py")
#     print("  4. Train the model")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()

"""
augment_wake_word.py
====================
Takes your 10 real "Hey Saya" recordings and augments them
into 300 samples ready for training.

HOW TO RUN:
  python3 augment_wake_word.py

STEP 1: Put your 10 real recordings in:
  /home/arz-1029/wake_word_detection/python_training_&_feature_extraction/hey_saya_samples/

STEP 2: Run this script:
  python3 augment_wake_word.py

STEP 3: Check the output folder:
  /home/arz-1029/wake_word_detection/python_training_&_feature_extraction/dataset/train/positive/
  Should contain 290+ WAV files
"""

import os
import numpy as np
import librosa
import soundfile as sf
import random

# ══════════════════════════════════════════════════════════════════
# PATHS — matching your project structure exactly
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = "/home/arz-1029/pt_task/python_training_&_feature_extraction"
INPUT_DIR  = os.path.join(BASE_DIR, "hey_saya_samples")        # your 10 real recordings
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "train", "positive")  # augmented output

TARGET_COUNT = 300
SAMPLE_RATE  = 16000
DURATION     = 2.0   # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# AUDIO HELPERS
# ══════════════════════════════════════════════════════════════════
def load_wav(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target   = int(SAMPLE_RATE * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


def save_wav(audio, path):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, SAMPLE_RATE, subtype='PCM_16')


# ══════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def pitch_shift(audio, semitones):
    return librosa.effects.pitch_shift(
        audio, sr=SAMPLE_RATE, n_steps=semitones
    ).astype(np.float32)


def time_stretch(audio, rate):
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target    = int(SAMPLE_RATE * DURATION)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    else:
        stretched = stretched[:target]
    return stretched.astype(np.float32)


def volume_change(audio, factor):
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def add_gaussian_noise(audio, level):
    noise = np.random.randn(len(audio)).astype(np.float32) * level
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def time_shift(audio, shift_ms):
    shift_samples = int(SAMPLE_RATE * shift_ms / 1000)
    if shift_samples > 0:
        shifted = np.pad(audio, (shift_samples, 0))[:len(audio)]
    else:
        shifted = np.pad(audio, (0, abs(shift_samples)))
        shifted = shifted[:len(audio)]
    return shifted.astype(np.float32)


def simulate_distance(audio, factor):
    quieter  = audio * factor
    smoothed = np.convolve(quieter, np.ones(3) / 3, mode='same').astype(np.float32)
    return np.clip(smoothed, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# ALL AUGMENTATION COMBINATIONS
# 10 originals × 29 augmentations = 290 + 10 originals = 300 total
# ══════════════════════════════════════════════════════════════════
def get_augmentations():
    aug = []

    # Pitch shifts (6)
    for s in [-3, -2, -1, +1, +2, +3]:
        aug.append((f"pitch_{s:+d}", lambda a, s=s: pitch_shift(a, s)))

    # Time stretch (4)
    for r in [0.85, 0.90, 1.10, 1.15]:
        aug.append((f"speed_{r:.2f}", lambda a, r=r: time_stretch(a, r)))

    # Volume (4)
    for v in [0.5, 0.7, 1.3, 1.5]:
        aug.append((f"vol_{v:.1f}", lambda a, v=v: volume_change(a, v)))

    # Gaussian noise (3)
    for l in [0.003, 0.008, 0.015]:
        aug.append((f"noise_{l:.3f}", lambda a, l=l: add_gaussian_noise(a, l)))

    # Time shift (4)
    for ms in [-150, -75, +75, +150]:
        aug.append((f"shift_{ms}ms", lambda a, ms=ms: time_shift(a, ms)))

    # Distance simulation (2)
    for d in [0.4, 0.65]:
        aug.append((f"dist_{d:.2f}", lambda a, d=d: simulate_distance(a, d)))

    # Combinations (6)
    aug.append(("combo_p1_n1",   lambda a: add_gaussian_noise(pitch_shift(a, +1), 0.005)))
    aug.append(("combo_pm1_n2",  lambda a: add_gaussian_noise(pitch_shift(a, -1), 0.008)))
    aug.append(("combo_slow_q",  lambda a: volume_change(time_stretch(a, 0.88), 0.6)))
    aug.append(("combo_fast_l",  lambda a: volume_change(time_stretch(a, 1.12), 1.4)))
    aug.append(("combo_p2_dist", lambda a: simulate_distance(pitch_shift(a, -2), 0.5)))
    aug.append(("combo_p2_n1",   lambda a: add_gaussian_noise(pitch_shift(a, +2), 0.008)))

    return aug  # 29 total


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Wake Word Augmentation")
    print("=" * 55)
    print(f"  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Check input folder exists
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder not found: {INPUT_DIR}")
        print(f"Create the folder and put your 10 WAV files inside it.")
        return

    # Load all real recordings
    wav_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"ERROR: No WAV files found in {INPUT_DIR}")
        return

    print(f"Found {len(wav_files)} real recordings:")
    originals = []
    for path in wav_files:
        try:
            audio = load_wav(path)
            originals.append((os.path.basename(path), audio))
            print(f"  ✅ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ❌ {os.path.basename(path)} — {e}")

    if len(originals) == 0:
        print("ERROR: Could not load any recordings.")
        return

    augmentations = get_augmentations()
    expected      = len(originals) * (1 + len(augmentations))
    print(f"\n{len(originals)} originals × {1 + len(augmentations)} versions")
    print(f"Expected output: ~{expected} samples")
    print()

    saved = 0

    # Save originals first
    for name, audio in originals:
        save_wav(audio, os.path.join(OUTPUT_DIR, f"orig_{name}"))
        saved += 1

    # Apply all augmentations
    print("Generating augmented samples...")
    for name, audio in originals:
        base = name.replace(".wav", "")
        for aug_name, aug_fn in augmentations:
            try:
                augmented = aug_fn(audio)
                save_wav(augmented, os.path.join(OUTPUT_DIR, f"{base}_{aug_name}.wav"))
                saved += 1
            except Exception:
                pass

    print()
    print("=" * 55)
    print(f"  Done! Saved {saved} samples to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 55)
    print()
    print("Next step: python3 train_embedding.py")


if __name__ == "__main__":
    main()