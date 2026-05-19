//==============================================================================
// wake_word_detector.cpp
// Real-time Embedding-based Wake Word Detection
//
// WHAT CHANGED FROM PREVIOUS VERSION:
//
//   1. LOADS ALL TEMPLATES (templates.npy) not just one template
//      embedder.load_templates() replaces embedder.load_template()
//
//   2. 100ms SLIDING WINDOW STRIDE instead of fixed 400ms chunks
//      CAPTURE_CHUNK = 1600 samples = 100ms
//      Every chunk triggers a new inference window
//      → Much less likely to miss a wake word between windows
//
//   3. TOP-K AVERAGE SIMILARITY instead of single dot product
//      compare_topk() scores all N templates, returns avg of top-3
//      → More robust than pure MAX against false positives
//
// UNCHANGED: 2-second audio window, MFCC pipeline, pre-emphasis, AGC
//
// COMPILE:
//   g++ -O2 wake_word_detector.cpp cnn_inference.cpp mfcc.cpp \
//       kissfft/kiss_fft.c -Ikissfft -lasound -o wake_word_detector
//
// USAGE:
//   ./wake_word_detector templates.npy [threshold] [top_k]
//   e.g.: ./wake_word_detector templates.npy 0.75 3
//==============================================================================

#include "mfcc.h"
#include "cnn_inference.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <alsa/asoundlib.h>

static const int   SAMPLE_RATE   = 16000;
static const int   N_FFT         = 512;
static const int   HOP_LENGTH    = 160;
static const int   N_MFCC        = 40;
static const int   MAX_FRAMES    = 196;

static const float GLOBAL_MEAN   = -1.9461231231689453f;
static const float GLOBAL_STD    = 12.231356620788574f;

static const int   CAPTURE_CHUNK  = 1600;  // 100ms Stride
static const int   TOP_K          = 3;
static const int   REQUIRED_CONSECUTIVE = 5; // 500ms sustained match
static const int   COOLDOWN_CHUNKS      = 20;

// Proper Ring Buffer implementation
class RingBuffer {
public:
    explicit RingBuffer(int capacity) : buf(capacity, 0.0f), cap(capacity), write_pos(0), is_full(false) {}

    void push(const float* samples, int count) {
        for (int i = 0; i < count; i++) {
            buf[write_pos] = samples[i];
            if (++write_pos >= cap) {
                write_pos = 0;
                is_full   = true;
            }
        }
    }

    void read_linear(float* out) const {
        if (!is_full) {
            memcpy(out, buf.data(), write_pos * sizeof(float));
            memset(out + write_pos, 0, (cap - write_pos) * sizeof(float));
        } else {
            int first_chunk = cap - write_pos;
            memcpy(out, buf.data() + write_pos, first_chunk * sizeof(float));
            memcpy(out + first_chunk, buf.data(), write_pos * sizeof(float));
        }
    }

    bool full() const { return is_full; }

private:
    std::vector<float> buf;
    int  cap;
    int  write_pos;
    bool is_full;
};

static void apply_pre_emphasis(float* audio, int len, float coeff = 0.97f) {
    for (int i = len - 1; i > 0; i--) {
        audio[i] = audio[i] - coeff * audio[i - 1];
    }
}

static void apply_agc(float* audio, int len) {
    float max_val = 0.0f;
    for (int i = 0; i < len; i++) {
        if (std::abs(audio[i]) > max_val) max_val = std::abs(audio[i]);
    }
    if (max_val > 0.15f) {
        float gain = std::min(0.80f / max_val, 8.0f);
        for (int i = 0; i < len; i++) {
            audio[i] = std::max(std::min(audio[i] * gain, 1.0f), -1.0f);
        }
    }
}

int main(int argc, char** argv) {
    std::string templates_path = "templates.npy";
    float       threshold      = 0.75f;

    if (argc >= 2) templates_path = argv[1];
    if (argc >= 3) threshold      = std::stof(argv[2]);

    mfcc_init();
    WakeWordEmbedder embedder;
    
    if (!embedder.load_templates(templates_path)) {
        std::cerr << "FATAL: Could not load templates.npy\n";
        return 1;
    }

    const int WINDOW_SAMPLES = SAMPLE_RATE * 2;
    RingBuffer ring(WINDOW_SAMPLES);

    std::vector<int16_t> pcm_buf(CAPTURE_CHUNK);
    std::vector<float>   float_buf(CAPTURE_CHUNK);
    std::vector<float>   audio_window(WINDOW_SAMPLES);
    std::vector<float>   mfcc_buf(N_MFCC * MAX_FRAMES);

    snd_pcm_t* handle;
    snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(handle, hw_params);
    snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(handle, hw_params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(handle, hw_params, 1);
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate, 0);
    snd_pcm_hw_params(handle, hw_params);
    snd_pcm_prepare(handle);

    int consecutive = 0, cooldown = 0, total_detections = 0;

    std::cout << "🎤 Listening... Say your wake word!\n";

    while (true) {
        int frames = snd_pcm_readi(handle, pcm_buf.data(), CAPTURE_CHUNK);
        if (frames < 0) { 
            snd_pcm_recover(handle, frames, 0); 
            continue; 
        }

        for (int i = 0; i < frames; i++) {
            float_buf[i] = pcm_buf[i] / 32768.0f;
        }
        ring.push(float_buf.data(), frames);

        if (!ring.full()) continue;
        if (cooldown > 0) { 
            cooldown--; 
            consecutive = 0; 
            continue; 
        }

        ring.read_linear(audio_window.data());
        apply_pre_emphasis(audio_window.data(), WINDOW_SAMPLES);
        apply_agc(audio_window.data(), WINDOW_SAMPLES);

        std::fill(mfcc_buf.begin(), mfcc_buf.end(), 0.0f);
        int frame_idx = 0;
        for (int i = 0; i + N_FFT <= WINDOW_SAMPLES && frame_idx < MAX_FRAMES; i += HOP_LENGTH) {
            float audio_frame[N_FFT];
            std::copy(&audio_window[i], &audio_window[i + N_FFT], audio_frame);
            float mfcc_frame[N_MFCC];
            mfcc_compute(audio_frame, mfcc_frame);
            for (int m = 0; m < N_MFCC; m++) {
                mfcc_buf[m * MAX_FRAMES + frame_idx] = mfcc_frame[m];
            }
            frame_idx++;
        }

        for (int i = 0; i < N_MFCC * MAX_FRAMES; i++) {
            mfcc_buf[i] = (mfcc_buf[i] - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8f);
        }

        float similarity = embedder.predict(mfcc_buf.data(), TOP_K);

        if (similarity > threshold) {
            consecutive++;
        } else {
            consecutive = 0;
        }

        // Clean interface: \r overwrites the current line instead of spamming down
        std::cout << "\rSimilarity: " << (int)(similarity * 100) << "% [" 
                  << consecutive << "/" << REQUIRED_CONSECUTIVE << "]      " << std::flush;

        if (consecutive >= REQUIRED_CONSECUTIVE) {
            total_detections++;
            cooldown = COOLDOWN_CHUNKS;
            consecutive = 0;
            std::cout << "\n🟢 WAKE WORD DETECTED! (#" << total_detections << ")\n";
        }
    }
    snd_pcm_close(handle);
    mfcc_free();
    return 0;
}