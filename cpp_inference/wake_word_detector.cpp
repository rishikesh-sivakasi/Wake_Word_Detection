//==============================================================================
// wake_word_detector.cpp
// Real-time Embedding-based Wake Word Detection
//
// CURRENT MATCHING STRATEGY (templates.npy holds exactly 2 rows):
//   Row 0 = "Hey Saya" prototype embedding (mean of registered samples)
//   Row 1 = "Noise"    prototype embedding (mean of registered samples)
//
//   Every inference window computes a live 128-dim embedding and takes
//   its cosine similarity (dot product, both sides are L2-normalized)
//   against BOTH prototypes:
//     saya_dot  = live . saya_prototype
//     noise_dot = live . noise_prototype
//
//   A detection candidate must pass BOTH gates (see compare_topk() in
//   cnn_inference.cpp):
//     Gate 1: saya_dot  > threshold        (caller-supplied, argv[2])
//     Gate 2: saya_dot - noise_dot > NOISE_MARGIN
//
//   NOTE: this is NOT a top-K search over many templates. The "top_k"
//   parameter threaded through predict()/compare_topk() is currently
//   unused (kept only so the signature can support a multi-template
//   bank later, e.g. several speakers' prototypes instead of one).
//   If you see code or docs elsewhere claiming "top-K average over N
//   templates", that describes an earlier design this project moved
//   away from — see register_wake_word.py's docstring for why.
//
// SLIDING WINDOW: 100ms stride (CAPTURE_CHUNK = 1600 samples) over a
//   2-second window, so a new inference happens on every chunk rather
//   than waiting for a full new window to fill.
//
// COMPILE:
//   g++ -O2 wake_word_detector.cpp cnn_inference.cpp mfcc.cpp \
//       kissfft/kiss_fft.c -Ikissfft -lasound -o wake_word_detector
//
// USAGE:
//   ./wake_word_detector templates.npy [threshold]
//   e.g.: ./wake_word_detector templates.npy 0.75
//   (there is currently no third CLI argument — top_k is not wired up;
//    see NOTE above)
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

static const float GLOBAL_MEAN   = -1.9617609977722168f;
static const float GLOBAL_STD    = 12.306929588317871f;

static const int   CAPTURE_CHUNK  = 1600;  // 100ms Stride
static const int   TOP_K          = 3;  // NOTE: currently unused by compare_topk() — see file header
static const int   REQUIRED_CONSECUTIVE = 2; // 200ms sustained match
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
        // FIX: clamp gain to [1.0, 8.0], matching the Python AGC used during
        // training/registration: np.clip(target / mv, 1.0, max_gain).
        // Without the 1.0 floor, loud audio (max_val between 0.8 and 1.0)
        // got silently *attenuated* here at inference time, while the
        // training-time Python version left it untouched in that range.
        // That mismatch meant some live audio was being fed through a
        // slightly different transform than anything the model ever saw
        // in training -> a real (if narrow) train/inference skew.
        float gain = std::min(std::max(0.80f / max_val, 1.0f), 8.0f);
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
        float max_amp = 0.0f;
        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            if (std::abs(audio_window[i]) > max_amp) max_amp = std::abs(audio_window[i]);
        }
        
        // Skip CNN if someone is just murmuring or room is quiet
        if (max_amp < 0.03f) {
            consecutive = 0;
            continue; 
        }
        // ----------------------------
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
        // std::cout << "\rSimilarity: " << (int)(similarity * 100) << "% [" 
        //           << consecutive << "/" << REQUIRED_CONSECUTIVE << "]      " << std::flush;

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