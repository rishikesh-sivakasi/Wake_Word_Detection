# Few-Shot Real-Time Wake Word Detection for Edge Systems

[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Raspberry%20Pi-orange.svg)](https://www.raspberrypi.com/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Quantization](https://img.shields.io/badge/Precision-INT8%20%2B%20FP32-green.svg)]()
[![Inference](https://img.shields.io/badge/Audio%20Backend-ALSA%20%2B%20KissFFT-lightgrey.svg)]()

A high-performance, lightweight, and privacy-preserving wake word detection engine designed specifically for embedded Linux and resource-constrained edge hardware (e.g., Raspberry Pi, Jetson Nano, robotics controllers). 

Powered by **Few-Shot Prototypical Metric Learning**, this system embeds 2-second audio segments onto a 128-dimensional unit hypersphere. Users can configure or swap custom wake words (defaulting to *"Hey Saya"*) using only **~10 spoken audio samples** without retraining the neural network.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [System Architecture](#system-architecture)
- [Signal Processing & Audio Specs](#signal-processing--audio-specs)
- [Neural Network Architecture](#neural-network-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start Guide](#quick-start-guide)
- [End-to-End Pipeline Walkthrough](#end-to-end-pipeline-walkthrough)
  - [1. Data Preparation & Augmentation](#1-data-preparation--augmentation)
  - [2. Model Training (Prototypical Networks)](#2-model-training-prototypical-networks)
  - [3. INT8 Quantization & C++ Header Export](#3-int8-quantization--c-header-export)
  - [4. Few-Shot Wake Word Registration](#4-few-shot-wake-word-registration)
  - [5. Real-Time Edge C++ Deployment](#5-real-time-edge-c-deployment)
- [Gating & Decision Logic](#gating--decision-logic)
- [Configuration & Tuning](#configuration--tuning)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Key Highlights

- **Few-Shot Registration (No Retraining Needed)**:
  Once the metric embedder is trained, registering a new wake word requires recording only ~10 audio samples. The registration tool calculates an average prototype vector and saves it into `templates.npy`.
- **Zero Train/Inference Audio Skew**:
  Feature extraction matches identically between Python (Librosa-based) and C++ (KissFFT-based): pre-emphasis filter ($\alpha=0.97$), Automatic Gain Control (AGC), Hamming windowing, 40-band Mel filterbank, and 40-point DCT.
- **Quantized Edge Runtime (INT8 + FP32)**:
  Convolutional and Linear projection weights are quantized to INT8 with per-layer scaling factors. Batch Normalization statistics are preserved in FP32 to ensure representation accuracy while minimizing edge CPU memory and cache footprints.
- **Robust Multi-Source Episodic Training**:
  The embedder is trained across a diverse 5-source mixture:
  1. Google Speech Commands v2 (30 distinct English words)
  2. AI4Bharat Svarah Dataset (Indian-accented English speech)
  3. ESC-50 Environmental Noise (50 categories of background sounds)
  4. Synthetic Hard Negatives (phonetic distractors like *"Hey Maya"*, *"Say Saya"*, *"Taste Papaya"* generated via neural Edge-TTS)
  5. 30x Augmented target wake word clips (*"Hey Saya"*)
- **Dual-Gate Margin Verification & Debouncing**:
  Inference executes against a template bank holding both the **Target Wake Word** prototype and a **Background Noise** prototype. Triggers must satisfy both an absolute threshold and a relative margin ($S_{wake} - S_{noise} \ge \text{margin}$) sustained over consecutive sliding frames.

---

## System Architecture

```
                                  LIVE MICROPHONE (ALSA)
                                            │
                                  16 kHz, 16-bit PCM Mono
                                            │
                                            ▼
                             ┌──────────────────────────────┐
                             │  Ring Buffer (32,000 samples)│
                             │   Sliding Window: 100ms hop  │
                             └──────────────┬───────────────┘
                                            │
                             ┌──────────────▼───────────────┐
                             │  Signal Conditioning (DSP)   │
                             │  • Pre-emphasis (α = 0.97)   │
                             │  • Automatic Gain Control    │
                             │  • Silence Gate (RMS > 0.03) │
                             └──────────────┬───────────────┘
                                            │
                             ┌──────────────▼───────────────┐
                             │    Feature Extraction        │
                             │  • KissFFT (512-point)       │
                             │  • 40 Mel Filterbanks        │
                             │  • 40 DCT Coeffs (196 frames)│
                             │  • Global Z-Score Norm       │
                             └──────────────┬───────────────┘
                                            │
                             ┌──────────────▼───────────────┐
                             │    Quantized CNN Embedder    │
                             │  • 3x Conv2D (INT8) + BN +   │
                             │    ReLU + MaxPool2D          │
                             │  • Global AvgPool2D          │
                             │  • Linear Projection (INT8)  │
                             │  • L2 Normalization (FP32)   │
                             └──────────────┬───────────────┘
                                            │ Live Embedding [128-dim]
                                            ▼
                             ┌──────────────────────────────┐
                             │    Dual-Gate Decision Unit   │
                             │  • Cosine Sim (Wake Word)    │
                             │  • Cosine Sim (Noise Bank)   │
                             │  • Margin: Swake - Snoise    │
                             │  • 2-Frame Debounce Window   │
                             └──────────────┬───────────────┘
                                            │
                                            ▼
                                🟢 WAKE WORD DETECTED!
```

---

## Signal Processing & Audio Specs

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Audio Sample Rate** | `16,000 Hz` | 16 kHz Mono PCM (16-bit signed integer) |
| **Window Duration** | `2.0 seconds` | Exactly 32,000 audio samples |
| **Sliding Stride (Hop)** | `100 ms` | 1,600 samples per evaluation step |
| **FFT Size (`N_FFT`)** | `512` | 32 ms window duration |
| **Hop Size (`HOP_LENGTH`)** | `160` | 10 ms step between successive spectral frames |
| **Spectral Window** | `Hamming` | Minimizes spectral leakage on speech frequencies |
| **Mel Filterbanks (`N_MELS`)** | `40` | Triangular Mel filters covering human speech bands |
| **MFCC Coefficients (`N_MFCC`)**| `40` | 40 cepstral coefficients per frame |
| **Max Frames (`MAX_FRAMES`)** | `196` | Matrix dimension: `(40, 196)` per 2-second clip |
| **Pre-emphasis Factor** | `0.97` | High-frequency boost: $y[t] = x[t] - 0.97 \cdot x[t-1]$ |
| **Automatic Gain Control** | `Target 0.80` | Clamped gain $\in [1.0, 8.0]$ when peak exceeds $0.15$ |

---

## Neural Network Architecture

The embedder (`WakeWordEmbedder`) maps a normalized MFCC matrix of shape `(1, 40, 196)` to a 128-dimensional L2-normalized vector on a unit hypersphere.

```
Input: (Batch, 1, 40, 196)
  │
  ├── Block 1: Conv2d(1 -> 16, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
  │            Output: (Batch, 16, 20, 98)
  │
  ├── Block 2: Conv2d(16 -> 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
  │            Output: (Batch, 32, 10, 49)
  │
  ├── Block 3: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
  │            Output: (Batch, 64, 5, 24)
  │
  ├── Global Pooling: AdaptiveAvgPool2d((1, 1))
  │            Output: (Batch, 64, 1, 1)
  │
  ├── Projection: Linear(64 -> 128)
  │            Output: (Batch, 128)
  │
  └── Normalization: F.normalize(p=2, dim=1)
               Output: (Batch, 128) [Unit Hypersphere Vector]
```

### Quantization Strategy

| Layer Component | Stored Precision | Runtime Execution | Notes |
| :--- | :--- | :--- | :--- |
| **Conv & Linear Weights** | `int8_t` | Fixed-point multiplication with float scale | Minimizes parameter storage to ~140 KB |
| **Linear Biases** | `int8_t` | Quantized with scale factor | Low memory footprint |
| **BatchNorm ($\gamma, \beta, \mu, \sigma^2$)** | `float32` | Floating-point normalization | Preserves dynamic range and stability |
| **Embedding Projection** | `float32` | L2-normalized vector comparison | Exact cosine distance computation |

---

## Repository Structure

```
.
├── cpp_inference/                               # Real-time C++ inference engine
│   ├── wake_word_detector.cpp                   # Main edge executable (ALSA audio stream + sliding window)
│   ├── cnn_inference.cpp / cnn_inference.h      # INT8 CNN forward pass & dual-gate matcher
│   ├── mfcc.cpp / mfcc.h                        # Edge MFCC feature extraction engine
│   ├── mfcc_data.h                              # Precomputed lookup tables (Hamming, Mel, DCT)
│   ├── dataset_extractor.cpp                    # Standalone WAV-to-CSV verification utility
│   ├── weights_embedding.h                      # Symlink/copy of quantized weights
│   ├── templates.npy                            # Symlink/copy of prototype bank
│   └── kissfft/                                 # KissFFT submodule for fast embedded FFT
└── python_training_&_feature_extraction/        # Training, evaluation & registration pipeline
    ├── embedding_model.py                       # PyTorch WakeWordEmbedder model definition
    ├── train_embedding.py                       # Episodic Prototypical network trainer
    ├── register_wake_word.py                    # Few-shot template generator (templates.npy)
    ├── quantize_exporter.py                     # Weight quantization & C++ header exporter
    ├── generate_tables.py                       # Generator for mfcc_data.h lookup arrays
    ├── global_stats.txt                         # Stored dataset normalization statistics
    ├── embedding_model.pth                      # Trained PyTorch checkpoint
    └── preprocessing_scripts/                   # Audio data preparation tools
        ├── augment_300_samples.py               # 30x pitch/speed/noise augmentation
        ├── synthetic_hard_negatives.py          # Edge-TTS neural hard negative synthesizer
        ├── prepare_svarah.py                    # AI4Bharat Svarah Indian accent extraction
        ├── prepare_esc50_noise.py               # ESC-50 environmental noise processor
        ├── format_audio.py                      # Audio padding, trimming & standardization
        └── record_samples.py                    # Terminal-based audio recording tool
```

---

## Quick Start Guide

### 1. Prerequisites (Ubuntu / Debian / Raspberry Pi OS)

Install the required C++ build toolchain and ALSA development headers:

```bash
sudo apt update
sudo apt install -y build-essential libasound2-dev git ffmpeg
```

### 2. Python Environment Setup

```bash
# Clone the repository
git clone https://github.com/intern-asimov/wakeword--task8.git
cd wakeword--task8

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchaudio librosa soundfile numpy scipy tqdm edge-tts audiomentations
```

### 3. Build & Run C++ Real-Time Inference

Compile the C++ inference engine:

```bash
cd cpp_inference

g++ -O2 wake_word_detector.cpp cnn_inference.cpp mfcc.cpp \
    kissfft/kiss_fft.c -Ikissfft -lasound -o wake_word_detector
```

Launch the detector (uses `templates.npy` with a detection threshold of `0.75`):

```bash
./wake_word_detector templates.npy 0.75
```

Speak your wake word (*"Hey Saya"*). When detected, the engine prints:
```
[Saya: 84% | Noise: 22% | Margin: 62%]
🟢 WAKE WORD DETECTED! (#1)
```

---

## End-to-End Pipeline Walkthrough

### 1. Data Preparation & Augmentation

A robust wake word model needs diverse positive examples, phonetic distractors, accent variations, and environmental noise.

#### A. Record Clean Positive Clips
Record 10 natural utterances of the wake word into `python_training_&_feature_extraction/dataset/hey_saya_samples/`.

#### B. Augment to 300 Samples
Run the audio augmentation script to expand each sample 30-fold using pitch shifts, speed variations (0.75x–1.20x), time shifts, and room background noise:

```bash
cd python_training_&_feature_extraction/preprocessing_scripts
python3 augment_300_samples.py
```
*Outputs 300 WAV files to `dataset/train/positive/`.*

#### C. Generate Neural Hard Negatives
Synthesize phonetically similar phrases using diverse neural voices (via Microsoft Edge-TTS):

```bash
python3 synthetic_hard_negatives.py
```
*Creates audio clips for phrases like "Hey Maya", "Say Saya", "Hey Siya", "Play Saya" in `dataset/train/negative/`.*

#### D. Prepare Accents & Environmental Noise
- **AI4Bharat Svarah Dataset**: Downloads Indian English speech and segments it by word boundaries using forced alignment:
  ```bash
  python3 prepare_svarah.py --out_dir ../indian_accent_data
  ```
- **ESC-50 Dataset**: Slices 2,000 environmental noise clips into 2-second chunks across 50 categories:
  ```bash
  python3 prepare_esc50_noise.py --esc50 /path/to/ESC-50 --output ../noise_esc50_prepared
  ```

---

### 2. Model Training (Prototypical Networks)

The model is trained using **Few-Shot Episodic Training**. In each episode:
- $N$ classes are sampled ($N\text{-way} = 5$).
- For each class, $K$ support samples ($K\text{-shot} = 10$) and $Q$ query samples ($Q\text{-query} = 5$) are drawn.
- The model computes class prototypes $\mathbf{c}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} f_\theta(\mathbf{x})$ and optimizes cross-entropy loss on query embeddings.

#### Training Mixture per Epoch:
- **35%**: Wake Word & Hard Negative episodes
- **25%**: Indian-accented speech episodes
- **15%**: ESC-50 Environmental Noise episodes
- **25%**: Google Speech Commands episodes

To run training:

```bash
cd python_training_&_feature_extraction

python3 train_embedding.py \
    --indian_data indian_accent_data \
    --esc50_noise noise_esc50_prepared
```

Outputs:
- `embedding_model.pth`: PyTorch weights checkpoint.
- `global_stats.txt`: Global mean and standard deviation for MFCC normalization.

---

### 3. INT8 Quantization & C++ Header Export

Convert the trained PyTorch model into an INT8 quantized C++ header:

```bash
cd python_training_&_feature_extraction
python3 quantize_exporter.py
```

This script:
1. Loads `embedding_model.pth`.
2. Quantizes Conv2D and Linear weights to `int8_t` and determines per-tensor scale factors: $\text{scale} = \frac{\max(|W|)}{127.0}$.
3. Exports Batch Normalization parameters in `float32`.
4. Writes `weights_embedding.h`.

Copy the generated header to the C++ inference directory:
```bash
cp weights_embedding.h ../cpp_inference/
```

---

### 4. Few-Shot Wake Word Registration

To register any wake word (either default *"Hey Saya"* or a brand-new custom phrase):

1. Place 10 spoken `.wav` files into `hey_saya_samples/`.
2. Place ambient background noise `.wav` files into `noise_samples/`.
3. Run the registration tool:

```bash
cd python_training_&_feature_extraction
python3 register_wake_word.py
```

#### What `register_wake_word.py` Does:
- Computes 128-dimensional embeddings for all samples using `embedding_model.pth`.
- Performs pairwise cosine similarity quality checks (ensuring consistency $> 0.60$).
- Averages positive embeddings into a **Wake Word Prototype** $\mathbf{p}_{wake}$.
- Averages noise embeddings into a **Noise Prototype** $\mathbf{p}_{noise}$.
- Saves a $(2, 128)$ matrix into `templates.npy` (Row 0 = Wake Word, Row 1 = Noise).

Copy the template file to the C++ directory:
```bash
cp templates.npy ../cpp_inference/
```

> [!TIP]
> **Changing the Wake Word**: To switch to another phrase (e.g., *"Hey Robot"*), you **never** need to retrain the neural network! Simply record 10 samples of the new phrase, re-run `register_wake_word.py`, and replace `templates.npy`.

---

### 5. Real-Time Edge C++ Deployment

Compile with optimization flags on the target edge device:

```bash
cd cpp_inference

g++ -O2 -Wall -Wextra wake_word_detector.cpp cnn_inference.cpp mfcc.cpp \
    kissfft/kiss_fft.c -Ikissfft -lasound -o wake_word_detector
```

Run the detector:

```bash
./wake_word_detector templates.npy 0.75
```

Arguments:
- `argv[1]` *(optional)*: Path to prototype templates file (default: `templates.npy`).
- `argv[2]` *(optional)*: Detection threshold (default: `0.75`).

---

## Gating & Decision Logic

During each 100ms stride, the engine computes a candidate score. To avoid false triggers, two verification gates must pass simultaneously:

$$\text{Gate 1 (Absolute Gate):} \quad \mathbf{e}_{live} \cdot \mathbf{p}_{wake} > \text{threshold}$$

$$\text{Gate 2 (Relative Margin Gate):} \quad (\mathbf{e}_{live} \cdot \mathbf{p}_{wake}) - (\mathbf{e}_{live} \cdot \mathbf{p}_{noise}) \ge \text{NOISE\_MARGIN}$$

Where:
- $\mathbf{e}_{live}$ is the live 128-dimensional L2-normalized embedding.
- $\mathbf{p}_{wake}$ is Row 0 of `templates.npy`.
- $\mathbf{p}_{noise}$ is Row 1 of `templates.npy`.
- $\text{NOISE\_MARGIN} = 0.10$ (requires the wake word similarity to lead noise similarity by at least 10 percentage points).

### Multi-Frame Debouncing

To eliminate transient noise spikes:
1. **Consecutive Frame Requirement**: The candidate must satisfy both gates for **2 consecutive frames** (~200ms sustained match).
2. **Post-Detection Cooldown**: Once triggered, a cooldown period of **20 chunks (2.0 seconds)** is enforced to avoid multiple detections from a single utterance.

---

## Configuration & Tuning

Key constants can be configured in `cpp_inference/wake_word_detector.cpp` and `cpp_inference/cnn_inference.cpp`:

| Parameter | Default Value | Location | Effect |
| :--- | :--- | :--- | :--- |
| `threshold` | `0.75` | CLI arg / `wake_word_detector.cpp` | Primary sensitivity threshold (range: 0.65 – 0.85) |
| `NOISE_MARGIN` | `0.10f` | `cnn_inference.cpp` | Minimum margin above noise prototype (prevents quiet-room false triggers) |
| `REQUIRED_CONSECUTIVE` | `2` | `wake_word_detector.cpp` | Consecutive matching windows before firing |
| `COOLDOWN_CHUNKS` | `20` | `wake_word_detector.cpp` | Frame silence lockout after trigger (20 chunks = 2.0s) |
| `CAPTURE_CHUNK` | `1600` | `wake_word_detector.cpp` | Stride size (1600 samples = 100ms step) |
| `silence_threshold` | `0.03f` | `wake_word_detector.cpp` | Skip CNN inference if window peak amplitude is $< 0.03$ |

---

## Troubleshooting & FAQs

### 1. ALSA Audio Device Error (`snd_pcm_open failed`)
- Ensure your microphone is recognized:
  ```bash
  arecord -l
  ```
- If the default device is not index 0, test recording:
  ```bash
  arecord -D hw:1,0 -d 3 -f S16_LE -r 16000 -c 1 test.wav
  ```
- To specify a custom ALSA device, change `"default"` in `wake_word_detector.cpp` (line 154) to `"hw:1,0"` or your specific device ID and recompile.

### 2. High False Positive Rate
- Increase the threshold (e.g., `./wake_word_detector templates.npy 0.80`).
- Increase `NOISE_MARGIN` in `cnn_inference.cpp` from `0.10f` to `0.15f`.
- Re-run `register_wake_word.py` with 5–10 room noise recordings captured directly in the deployment environment.

### 3. Missed Wake Words (High False Negative Rate)
- Lower the threshold slightly (e.g., `0.70`).
- Check that the microphone input volume is properly set using `alsamixer`.
- Verify sample consistency: inspect the pairwise similarity matrix printed by `register_wake_word.py`. Samples scoring below `0.60` indicate inconsistent pronunciation or background clipping.

---

## License

This project is licensed under the Apache License 2.0 / MIT License. Refer to the repository root for full licensing terms.

