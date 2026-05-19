// ==============================================================================
// dataset_extractor.cpp
// Converts a WAV file into a CSV of MFCC features using the SAME C++ math
// engine that runs during live inference — guaranteeing zero train/test mismatch.
//
// Compile:
//   g++ -O3 dataset_extractor.cpp mfcc.cpp kissfft/kiss_fft.c -Ikissfft -o dataset_extractor
//
// Usage:
//   ./dataset_extractor <input.wav> <output.csv>
// ==============================================================================

#include "mfcc.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>   // strncmp
#include <cstdint>   // uint32_t, int16_t
#include <cmath>     // std::sqrt

// ==============================================================================
// CONFIGURATION
// ==============================================================================
static const int N_FFT      = 512;
static const int HOP_LENGTH = 160;
static const int N_MFCC     = 40;
static const int MAX_FRAMES = 196;

// ENABLED: Set to 0.97f to perfectly match your live edge device inference!
static const float PRE_EMPHASIS = 0.97f; 

// ==============================================================================
// WAV HEADER PARSER
// ==============================================================================
bool parse_wav_header(std::ifstream& f, int& sample_rate, int& num_channels, int& bits_per_sample) {
    char riff_id[4];
    f.read(riff_id, 4);
    if (strncmp(riff_id, "RIFF", 4) != 0) {
        std::cerr << "[ERROR] Not a valid RIFF WAV file.\n";
        return false;
    }

    uint32_t file_size;
    f.read(reinterpret_cast<char*>(&file_size), 4); 

    char wave_id[4];
    f.read(wave_id, 4);
    if (strncmp(wave_id, "WAVE", 4) != 0) {
        std::cerr << "[ERROR] RIFF type is not WAVE.\n";
        return false;
    }

    bool found_fmt  = false;
    bool found_data = false;

    while (!f.eof() && !found_data) {
        char chunk_id[4];
        uint32_t chunk_size;

        f.read(chunk_id, 4);
        f.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (f.fail()) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format;
            uint16_t channels;
            uint32_t sample_rate_u;
            uint32_t byte_rate;
            uint16_t block_align;
            uint16_t bits;

            f.read(reinterpret_cast<char*>(&audio_format),   2);
            f.read(reinterpret_cast<char*>(&channels),       2);
            f.read(reinterpret_cast<char*>(&sample_rate_u),  4);
            f.read(reinterpret_cast<char*>(&byte_rate),      4);
            f.read(reinterpret_cast<char*>(&block_align),    2);
            f.read(reinterpret_cast<char*>(&bits),           2);

            if (chunk_size > 16) {
                f.seekg(chunk_size - 16, std::ios::cur);
            }

            if (audio_format != 1) {
                std::cerr << "[ERROR] Only PCM (format=1) WAV files are supported.\n";
                return false;
            }

            sample_rate    = static_cast<int>(sample_rate_u);
            num_channels   = static_cast<int>(channels);
            bits_per_sample = static_cast<int>(bits);
            found_fmt = true;

        } else if (strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) {
        std::cerr << "[ERROR] WAV file is missing required chunks.\n";
        return false;
    }

    return true;
}

// ==============================================================================
// MAIN
// ==============================================================================
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./dataset_extractor <input.wav> <output.csv>\n";
        return 1;
    }

    std::ifstream wav_file(argv[1], std::ios::binary);
    if (!wav_file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << argv[1] << "\n";
        return 1;
    }

    int sample_rate = 0, num_channels = 0, bits_per_sample = 0;
    if (!parse_wav_header(wav_file, sample_rate, num_channels, bits_per_sample)) {
        return 1;
    }

    if (bits_per_sample != 16) {
        std::cerr << "[ERROR] Only 16-bit PCM is supported.\n";
        return 1;
    }

    std::vector<int16_t> pcm_data;
    int16_t sample;
    while (wav_file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        pcm_data.push_back(sample);
    }
    wav_file.close();

    std::vector<float> audio_data;
    if (num_channels == 2) {
        audio_data.resize(pcm_data.size() / 2);
        for (size_t i = 0; i < audio_data.size(); i++) {
            float left  = pcm_data[i * 2]     / 32768.0f;
            float right = pcm_data[i * 2 + 1] / 32768.0f;
            audio_data[i] = (left + right) * 0.5f;
        }
    } else {
        audio_data.resize(pcm_data.size());
        for (size_t i = 0; i < pcm_data.size(); i++) {
            audio_data[i] = pcm_data[i] / 32768.0f;
        }
    }

    // ==========================================
    // SAFETY GUARD: Prevent Segfaults on tiny files
    // ==========================================
    if (audio_data.size() < static_cast<size_t>(N_FFT)) {
        std::cerr << "[ERROR] Audio file too short (" << audio_data.size() 
                  << " samples). Need at least " << N_FFT << " samples.\n";
        return 1;
    }

    // ==========================================
    // 1. PRE-EMPHASIS FILTER (Hardware Match)
    // ==========================================
    if (PRE_EMPHASIS > 0.0f) {
        for (size_t i = audio_data.size() - 1; i > 0; i--) {
            audio_data[i] = audio_data[i] - PRE_EMPHASIS * audio_data[i - 1];
        }
    }

    // ==========================================
    // 2. AUTOMATIC GAIN CONTROL (Hardware Match)
    // ==========================================
    float max_val = 0.0f;
    for (float s : audio_data) {
        max_val = std::max(max_val, std::abs(s));
    }

    if (max_val > 0.15f) {
        const float TARGET_LEVEL = 0.80f;
        float dynamic_gain = TARGET_LEVEL / max_val; 
        
        if (dynamic_gain > 8.0f) dynamic_gain = 8.0f; 
        if (dynamic_gain < 1.0f) dynamic_gain = 1.0f; 

        for (float& s : audio_data) {
            s = s * dynamic_gain;
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
        }
    }

    mfcc_init();

    // Raw un-normalized buffer layout: [N_MFCC rows][MAX_FRAMES cols] 
    std::vector<float> mfcc_buffer(N_MFCC * MAX_FRAMES, 0.0f);

    int frame_idx = 0;
    for (size_t i = 0; i + N_FFT <= audio_data.size(); i += HOP_LENGTH) {
        if (frame_idx >= MAX_FRAMES) break;

        float audio_frame[N_FFT];
        for (int j = 0; j < N_FFT; j++) {
            audio_frame[j] = audio_data[i + j];
        }

        float mfcc_frame[N_MFCC];
        mfcc_compute(audio_frame, mfcc_frame);

        for (int m = 0; m < N_MFCC; m++) {
            mfcc_buffer[m * MAX_FRAMES + frame_idx] = mfcc_frame[m];
        }
        frame_idx++;
    }

    mfcc_free();

    // NOTE: Z-Score Normalization was explicitly removed here.
    // The CSVs will contain RAW data. Python will calculate the Mean/Std.

    std::ofstream csv_file(argv[2]);
    if (!csv_file.is_open()) {
        std::cerr << "[ERROR] Cannot create output file: " << argv[2] << "\n";
        return 1;
    }

    for (int m = 0; m < N_MFCC; m++) {
        for (int f = 0; f < MAX_FRAMES; f++) {
            csv_file << mfcc_buffer[m * MAX_FRAMES + f];
            if (f < MAX_FRAMES - 1) csv_file << ",";
        }
        csv_file << "\n";
    }
    csv_file.close();

    return 0;
}