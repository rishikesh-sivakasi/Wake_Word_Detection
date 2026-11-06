#!/usr/bin/env bash
# ==============================================================================
# download_content.sh - Shell script to download Multilingual KWS content assets
# ==============================================================================
set -e

TARGET_DIR="${1:-content}"

echo "================================================================="
echo " Multilingual Keyword Spotting (KWS) Content Asset Downloader"
echo " Destination Directory: ${TARGET_DIR}"
echo "================================================================="

mkdir -p "${TARGET_DIR}"

# 1. Pretrained Multilingual Embedding Model
EMBED_DIR="${TARGET_DIR}/multilingual_context_73_0.8011"
EMBED_URL="https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz"

if [ -f "${EMBED_DIR}/saved_model.pb" ]; then
    echo "✓ Pretrained Embedding Model already exists at: ${EMBED_DIR}/saved_model.pb"
else
    echo "[-] Downloading pretrained embedding model (~83 MB)..."
    wget -c "${EMBED_URL}" -O "${TARGET_DIR}/multilingual_context_73_0.8011.tar.gz"
    echo "[-] Extracting pretrained embedding model..."
    tar -xzf "${TARGET_DIR}/multilingual_context_73_0.8011.tar.gz" -C "${TARGET_DIR}"
    rm -f "${TARGET_DIR}/multilingual_context_73_0.8011.tar.gz"
    echo "✓ Pretrained embedding model ready."
fi

# 2. Unknown Words Dataset
UNKNOWN_DIR="${TARGET_DIR}/unknown_files"
UNKNOWN_URL="https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz"

if [ -d "${UNKNOWN_DIR}/unknown_words" ]; then
    echo "✓ Unknown Words Dataset already exists at: ${UNKNOWN_DIR}/unknown_words"
else
    echo "[-] Downloading unknown words dataset (~52 MB)..."
    mkdir -p "${UNKNOWN_DIR}"
    wget -c "${UNKNOWN_URL}" -O "${TARGET_DIR}/unknown_files.tar.gz"
    echo "[-] Extracting unknown words dataset..."
    tar -xzf "${TARGET_DIR}/unknown_files.tar.gz" -C "${UNKNOWN_DIR}"
    rm -f "${TARGET_DIR}/unknown_files.tar.gz"
    echo "✓ Unknown words dataset ready."
fi

# 3. Background Noise (from Google Speech Commands v0.02)
SC_DIR="${TARGET_DIR}/speech_commands"
SC_URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

if [ -d "${SC_DIR}/_background_noise_" ]; then
    echo "✓ Background noise already exists at: ${SC_DIR}/_background_noise_"
else
    echo "[-] Downloading Google Speech Commands dataset (extracting _background_noise_)..."
    mkdir -p "${SC_DIR}"
    wget -c "${SC_URL}" -O "${TARGET_DIR}/speech_commands_v0.02.tar.gz"
    echo "[-] Extracting _background_noise_..."
    tar -xzf "${TARGET_DIR}/speech_commands_v0.02.tar.gz" -C "${SC_DIR}" --wildcards '*_background_noise_*' || \
    tar -xzf "${TARGET_DIR}/speech_commands_v0.02.tar.gz" -C "${SC_DIR}" _background_noise_
    rm -f "${TARGET_DIR}/speech_commands_v0.02.tar.gz"
    echo "✓ Background noise ready."
fi

echo ""
echo "================================================================="
echo "✅ All required assets are verified and ready in '${TARGET_DIR}'!"
echo "================================================================="

