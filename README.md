# Few-Shot Multilingual Keyword Spotting (KWS) & Wake Word Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Legacy%20(tf__keras)-red.svg)](https://github.com/tensorflow/tf-keras)
[![Interspeech 2021](https://img.shields.io/badge/Paper-Interspeech%202021-brightgreen.svg)](https://www.isca-archive.org/interspeech_2021/mazumder21_interspeech.pdf)
[![NeurIPS 2021](https://img.shields.io/badge/Dataset-NeurIPS%202021%20MSWC-yellow.svg)](https://openreview.net/pdf?id=c20jiJ5K2H)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end framework for **Few-Shot Keyword Spotting (KWS)** and custom wake word detection across multiple languages. Using a pre-trained multilingual acoustic embedding representation, you can train a highly accurate keyword classifier with as few as **5 to 20 recorded samples** without training an acoustic model from scratch.

This repository includes:
- **Pretrained Multilingual Embedding Model**: Transfer learn onto new keywords in minutes.
- **Automated Content Downloader**: One-command scripts to fetch and verify all necessary dataset assets.
- **Audio Preprocessing Pipeline**: Automatic padding, centering, and background noise blending for 16kHz mono audio.
- **Few-Shot Training CLI**: Train custom models with negative word and background noise sampling.
- **Streaming Inference & Visualizer**: Run detection over long audio files and inspect predictions in a web browser.
- **Real-Time Live Microphone Inference**: Low-latency sliding-window wake word detection directly from your microphone.

---

## Academic References

This repository is built upon research from Harvard University:
* **[Few-Shot Keyword Spotting in Any Language](https://www.isca-archive.org/interspeech_2021/mazumder21_interspeech.pdf)**, *Interspeech 2021*
* **[Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H)**, *NeurIPS 2021 Datasets and Benchmarks Track*

### Colab Tutorial & Videos
* **[Google Colab Introductory Tutorial](https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb)**
* **Full Dataset**: [Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/words)
* **[Few-Shot Keyword Spotting Overview (YouTube)](https://www.youtube.com/watch?v=1JRlJoEhYxE)**
* **[Multilingual Spoken Words Corpus Overview (YouTube)](https://www.youtube.com/watch?v=eGPCwnRtRng)**

---

## Repository Structure

```text
multilingual_kws/
├── multilingual_kws/                     # Core library package
│   ├── run.py                           # CLI entrypoint for train and inference
│   └── embedding/                       # Feature extraction & transfer learning logic
├── download_content.py                   # Automated Python asset downloader
├── download_content.sh                   # Automated Bash asset downloader
├── record_test.py                        # Helper to record custom mic audio
├── format_wavs.py                        # Preprocessing: normalize, pad with room noise to 1s
├── live_test.py                          # Basic real-time microphone keyword listener
├── live_test_2.py                        # Advanced real-time listener (hop, cooldown, debug)
├── room_noise/                           # Ambient room noise clips for audio padding
├── visualizer/                           # Web UI for inspecting detection timelines
└── requirements.txt                      # Python dependencies
```

---

## Prerequisites & Installation

### 1. System Dependencies (Linux / Ubuntu)

Audio processing and microphone streaming require system-level tools:

```bash
sudo apt-get update
sudo apt-get install -y sox libsox-fmt-all portaudio19-dev ffmpeg
```

> **Important**: `sox` and `soxi` are strictly required by the training script to validate audio parameters (16 kHz, 1.0 second duration, mono).

### 2. Python Environment Setup

We recommend using Python 3.10 or 3.11 with a virtual environment:

```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd multilingual_kws

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Critical TensorFlow Configuration

TensorFlow 2.16+ uses Keras 3 by default. Because the pretrained embedding model was saved using Keras 2 formats, you **must** export the legacy Keras flag before running training or inference:

```bash
export TF_USE_LEGACY_KERAS=1
export TF_ENABLE_ONEDNN_OPTS=0
```

Add these lines to your `~/.bashrc` or virtual environment's `bin/activate` script for convenience.

---

## Content & Pretrained Model Download Guide

Before training models or running inference, you need the **`content/`** directory containing the three foundational assets:

| Asset | Description | Size | Required By |
|---|---|---|---|
| **Pretrained Embedding** (`multilingual_context_73_0.8011`) | Core acoustic feature backbone trained on 73+ languages | ~83 MB | `train`, `inference` |
| **Unknown Words Dataset** (`unknown_files`) | Negative speech samples across 9 languages for `_unknown_` class | ~52 MB | `train` |
| **Background Noise** (`speech_commands`) | Ambient noise files from Google Speech Commands v0.02 | ~12 MB (`_background_noise_`) | `train` |
| *(Optional)* **MSWC Microset** (`mswc_microset`) | Subset of MSWC clips for multilingual experimentation | ~612 MB | Colab exploration |

### Method 1: Automated Python Downloader (Recommended)

Run the included standalone Python script. It uses standard Python libraries (no extra tools required) and extracts only what is needed:

```bash
# Download and verify the 3 core assets into ./content/
python download_content.py

# Optionally include the MSWC microset:
python download_content.py --include-mswc
```

### Method 2: Automated Bash Script

Alternatively, use the shell script (requires `wget` and `tar`):

```bash
chmod +x download_content.sh
./download_content.sh
```

### Method 3: Manual CLI Download (`wget` & `tar`)

If you prefer running the commands manually:

```bash
mkdir -p content/speech_commands content/unknown_files

# 1. Pretrained Multilingual Embedding Model (~83 MB)
wget https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz
tar -xzf multilingual_context_73_0.8011.tar.gz -C content/
rm multilingual_context_73_0.8011.tar.gz

# 2. Unknown Words Dataset (~52 MB)
wget https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz
tar -xzf unknown_files.tar.gz -C content/unknown_files/
rm unknown_files.tar.gz

# 3. Google Speech Commands Background Noise (~2.4 GB archive, extracts only _background_noise_)
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C content/speech_commands/ --wildcards '*_background_noise_*'
rm speech_commands_v0.02.tar.gz
```

### Method 4: Python / Google Colab (`tf.keras.utils.get_file`)

If working inside a Jupyter Notebook or Google Colab:

```python
import tensorflow as tf

assets = [
    ("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", "content/speech_commands"),
    ("https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz", "content/embedding_model"),
    ("https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz", "content/unknown_files"),
]

for url, cache_subdir in assets:
    print(f"Downloading {url}...")
    tf.keras.utils.get_file(origin=url, untar=True, cache_subdir=cache_subdir)
```

### Expected `content/` Directory Layout

Once downloaded, your `content/` folder should look like this:

```text
content/
├── multilingual_context_73_0.8011/
│   ├── assets/
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── speech_commands/
│   └── _background_noise_/
│       ├── doing_the_dishes.wav
│       ├── dude_miaowing.wav
│       ├── exercise_bike.wav
│       ├── pink_noise.wav
│       ├── running_tap.wav
│       └── white_noise.wav
└── unknown_files/
    ├── unknown_files.txt
    └── unknown_words/
        ├── ca/
        ├── de/
        ├── en/
        ├── es/
        ├── fa/
        ├── fr/
        ├── it/
        ├── nl/
        └── rw/
```

---

## End-to-End Walkthrough: Train a Custom Wake Word

### Step 1: Record Target Audio Samples

Record **10 to 30 spoken samples** of your target keyword (e.g., *"hey saya"* or *"alexa"*). You can record using your smartphone, an audio editor (like Audacity), or using `record_test.py`:

```bash
# Record 15 seconds from your microphone to practice
python record_test.py 15
```

Save all raw `.wav` recordings into a dedicated directory:
```text
hey_saya_raw/
├── sample_01.wav
├── sample_02.wav
└── ...
```

### Step 2: Audio Preprocessing (`format_wavs.py`)

The training system expects audio clips that strictly adhere to:
* **Sample Rate**: 16,000 Hz
* **Channels**: 1 (Mono)
* **Duration**: Exactly 1.0 second (16,000 samples)
* **Bit Depth**: 16-bit PCM

Use `format_wavs.py` to automatically center the speech, trim extra silence, and pad shorter clips using realistic background room noise:

```bash
python format_wavs.py \
  --input hey_saya_raw \
  --output hey_saya_processed \
  --noise room_noise
```

### Step 3: Fine-Tune the Few-Shot Model

Run few-shot transfer learning using `multilingual_kws.run train`:

```bash
TF_USE_LEGACY_KERAS=1 PYTHONPATH=. python -m multilingual_kws.run train \
  --keyword "hey_saya" \
  --samples_dir "hey_saya_processed" \
  --embedding "content/multilingual_context_73_0.8011" \
  --unknown_words "content/unknown_files/unknown_words" \
  --background_noise "content/speech_commands/_background_noise_" \
  --output "hey_saya_model" \
  --num_epochs 4 \
  --num_batches 1 \
  --primary_learning_rate 0.001 \
  --batch_size 64 \
  --unknown_percentage 50.0
```

#### Training Parameter Breakdown:
* `--keyword`: The string name of your custom wake word.
* `--samples_dir`: Path to processed 1-second 16kHz WAV files.
* `--embedding`: Path to the pretrained embedding model.
* `--unknown_words`: Directory of multilingual negative words.
* `--background_noise`: Directory containing `_background_noise_`.
* `--output`: Directory where the trained TensorFlow model will be saved.
* `--num_epochs`: Training epochs (default: `4`).
* `--num_batches`: Number of data generation batches (default: `1`).
* `--unknown_percentage`: Percentage of negative samples drawn from unknown words (default: `50.0`).

### Step 4: Stream Inference & Visualizer

Test the fine-tuned model against a long continuous audio stream (e.g., recorded podcast, conversation, or background noise) and view detections in your browser:

```bash
TF_USE_LEGACY_KERAS=1 PYTHONPATH=. python -m multilingual_kws.run inference \
  --keywords '["hey_saya"]' \
  --modelpaths "hey_saya_model" \
  --wav "test_stream.wav" \
  --detection_threshold 0.85 \
  --visualizer True \
  --serve_port 8080
```

Open `http://localhost:8080` in your browser to view interactive detection spikes overlaid on the audio timeline.

### Step 5: Real-Time Live Microphone Testing

Run live microphone streaming inference using `live_test_2.py`:

```bash
python live_test_2.py \
  --model "hey_saya_model" \
  --keyword "hey_saya" \
  --threshold 0.80 \
  --hop 0.15 \
  --cooldown 2.0
```

* **`--threshold`**: Confidence threshold (0.0 – 1.0) required to trigger detection.
* **`--hop`**: Sliding window shift in seconds (default: `0.15s` = 150ms latency).
* **`--cooldown`**: Minimum time in seconds before triggering another detection.

---

## How to Push to Your Own GitHub Repository

If you cloned this repository from Harvard Edge and want to push your custom branch and changes to your personal GitHub account:

### 1. Check Current Remotes

```bash
git remote -v
```

### 2. Set Remote to Your GitHub Repository

If you created a new empty repository on your GitHub (e.g. `https://github.com/<your-username>/multilingual_kws.git`):

```bash
# Option A: Update existing origin
git remote set-url origin https://github.com/<your-username>/multilingual_kws.git

# Option B: Or add a new remote (e.g., my-github)
git remote add my-github https://github.com/<your-username>/multilingual_kws.git
```

### 3. Create and Switch to a New Branch

```bash
git checkout -b feature/custom-wake-word-pipeline
```

### 4. Stage and Commit Your Changes

```bash
git add .
git commit -m "Add automated content downloader, custom audio pipeline, live testing, and enhanced README"
```

### 5. Push to Your GitHub Repository

```bash
# If using origin:
git push -u origin feature/custom-wake-word-pipeline

# If using my-github:
git push -u my-github feature/custom-wake-word-pipeline
```

---

## Troubleshooting & FAQs

### Q1: `soxi: command not found` or `ValueError: ... appears to not be a 16KHz 1-second wav file`
* **Fix**: Install `sox` and `libsox-fmt-all` via `sudo apt-get install sox libsox-fmt-all`.
* Make sure you ran `format_wavs.py` on all sample files. Every file must be exactly 16,000 samples (1.00s) and 16-bit mono.

### Q2: `ValueError: File format not recognized` or Keras 3 compatibility errors
* **Fix**: TensorFlow 2.16+ defaults to Keras 3. Run:
  ```bash
  export TF_USE_LEGACY_KERAS=1
  ```
  and ensure `tf-keras` is installed (`pip install tf-keras`).

### Q3: `ALSA lib pcm.c ... cannot open device` or Microphone Errors
* Ensure your user has permissions to access audio devices:
  ```bash
  sudo usermod -aG audio $USER
  ```
* Test listing audio input devices using Python:
  ```python
  import sounddevice as sd
  print(sd.query_devices())
  ```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
