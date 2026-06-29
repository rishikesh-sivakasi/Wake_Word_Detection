//==============================================================================
// cnn_inference.cpp
//
// BUFFER SIZES (correct — do not change):
//   conv1: (1,40,196)  → (16,40,196) → pool → (16,20,98)
//   conv2: (16,20,98)  → (32,20,98)  → pool → (32,10,49)
//   conv3: (32,10,49)  → (64,10,49)  → pool → (64,5,24)
//
// COMPARE_TOPK STRATEGY (updated):
//   Uses a margin-based filter instead of a hard noise_dot > saya_dot cut.
//
//   WHY THE CHANGE:
//     The old check (noise_dot > saya_dot → return 0) is too fragile.
//     If your noise prototype doesn't cover a particular background sound
//     well, noise_dot stays low AND saya_dot stays low → false triggers
//     still leak through because both scores are low but saya_dot wins.
//     Worse, in a quiet room noise_dot is near 0, so almost anything
//     can satisfy saya_dot > noise_dot and pass at a low similarity.
//
//   NEW LOGIC:
//     Two independent gates must BOTH pass:
//       Gate 1: saya_dot must exceed the threshold (set by caller)
//       Gate 2: saya_dot must exceed noise_dot by at least NOISE_MARGIN
//
//     NOISE_MARGIN (default 0.10) means: "Hey Saya" must score at least
//     10 percentage points MORE than noise before the score is returned.
//     This keeps the filter meaningful even when both scores are low.
//
//   TUNING:
//     Raise NOISE_MARGIN → fewer false positives, slightly higher miss rate
//     Lower NOISE_MARGIN → more sensitive, risk of more false positives
//     Start at 0.10 and adjust based on your test results.
//
// COMPILE:
//   g++ -O2 wake_word_detector.cpp cnn_inference.cpp mfcc.cpp \
//       kissfft/kiss_fft.c -Ikissfft -lasound -o wake_word_detector
//
// USAGE:
//   ./wake_word_detector templates.npy [threshold]
//   e.g.: ./wake_word_detector templates.npy 0.75
//==============================================================================

#include "cnn_inference.h"
#include "weights_embedding.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

static const float BN_EPS      = 1e-5f;
// FIX: this was 0.05f, contradicting the "default 0.10" documented above.
// Restored to 0.10 to match the documented design intent.
// IMPORTANT: with the current templates.npy, the Saya/Noise prototypes are
// only ~0.20 apart in cosine similarity (saya·noise ≈ 0.80), so this margin
// alone can't fully compensate for a poorly-separated embedding space.
// Re-measure saya·noise separation after retraining with hard negatives
// wired into the episodes (see train_embedding.py), and re-tune this value
// against real false-accept/false-reject data rather than trusting 0.10 blindly.
static const float NOISE_MARGIN = 0.10f;

// ════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR
// ════════════════════════════════════════════════════════════════════════════

WakeWordEmbedder::WakeWordEmbedder() : templates_loaded(false), threshold(0.75f) {
    conv1_out.resize(16 * 40 * 196, 0.0f);
    pool1_out.resize(16 * 20 * 98,  0.0f);
    conv2_out.resize(32 * 20 * 98,  0.0f);
    pool2_out.resize(32 * 10 * 49,  0.0f);
    conv3_out.resize(64 * 10 * 49,  0.0f);
    pool3_out.resize(64 *  5 * 24,  0.0f);
    global_pool_out.resize(64,      0.0f);
    proj_out.resize(EMBEDDING_DIM,  0.0f);
}

WakeWordEmbedder::~WakeWordEmbedder() {}

// ════════════════════════════════════════════════════════════════════════════
// LAYER PRIMITIVES
// ════════════════════════════════════════════════════════════════════════════

void WakeWordEmbedder::conv2d(
    const float*  input,
    const int8_t* weights,  float weight_scale,
    const float*  bn_gamma,
    const float*  bn_beta,
    const float*  bn_mean,
    const float*  bn_var,
    float*        output,
    int in_h,  int in_w,  int in_ch,
    int out_ch, int ks,
    int stride, int padding)
{
    int out_h = (in_h + 2 * padding - ks) / stride + 1;
    int out_w = (in_w + 2 * padding - ks) / stride + 1;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < ks; kh++) {
                        for (int kw = 0; kw < ks; kw++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                float x     = input[ic * in_h * in_w + ih * in_w + iw];
                                int   w_idx = oc * (in_ch * ks * ks)
                                            + ic * (ks * ks)
                                            + kh * ks + kw;
                                acc += x * (float)weights[w_idx] * weight_scale;
                            }
                        }
                    }
                }
                float norm = (acc - bn_mean[oc]) / sqrtf(bn_var[oc] + BN_EPS);
                output[oc * out_h * out_w + oh * out_w + ow] =
                    bn_gamma[oc] * norm + bn_beta[oc];
            }
        }
    }
}

void WakeWordEmbedder::relu(float* data, int size) {
    for (int i = 0; i < size; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

void WakeWordEmbedder::maxpool2d(
    const float* input, float* output,
    int in_h, int in_w, int channels,
    int pool_size, int stride)
{
    int out_h = (in_h - pool_size) / stride + 1;
    int out_w = (in_w - pool_size) / stride + 1;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float mv = -1e9f;
                for (int ph = 0; ph < pool_size; ph++)
                    for (int pw = 0; pw < pool_size; pw++) {
                        float v = input[c * in_h * in_w
                                      + (oh * stride + ph) * in_w
                                      + (ow * stride + pw)];
                        if (v > mv) mv = v;
                    }
                output[c * out_h * out_w + oh * out_w + ow] = mv;
            }
        }
    }
}

void WakeWordEmbedder::global_avgpool2d(
    const float* input, float* output,
    int h, int w, int channels)
{
    int sp = h * w;
    for (int c = 0; c < channels; c++) {
        float s = 0.0f;
        for (int i = 0; i < sp; i++) s += input[c * sp + i];
        output[c] = s / (float)sp;
    }
}

void WakeWordEmbedder::linear(
    const float*  input,
    const int8_t* weight,  float weight_scale,
    const int8_t* bias,    float bias_scale,
    float*        output,
    int in_features, int out_features)
{
    for (int o = 0; o < out_features; o++) {
        float s = 0.0f;
        for (int i = 0; i < in_features; i++)
            s += input[i] * (float)weight[o * in_features + i] * weight_scale;
        s += (float)bias[o] * bias_scale;
        output[o] = s;
    }
}

void WakeWordEmbedder::l2_normalize(float* vec, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) norm += vec[i] * vec[i];
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < size; i++) vec[i] /= norm;
}

// ════════════════════════════════════════════════════════════════════════════
// TEMPLATE LOADING — reads (2, EMBEDDING_DIM) matrix from .npy file
//   Row 0 = Hey Saya prototype
//   Row 1 = Noise prototype
// ════════════════════════════════════════════════════════════════════════════

int WakeWordEmbedder::num_templates() const {
    return (int)template_bank.size();
}

bool WakeWordEmbedder::load_templates(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Embedder] ERROR: Cannot open: " << path << "\n";
        return false;
    }

    // Validate .npy magic
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        std::cerr << "[Embedder] ERROR: Not a valid .npy file.\n";
        return false;
    }

    uint8_t  major, minor;
    uint16_t header_len;
    f.read((char*)&major,      1);
    f.read((char*)&minor,      1);
    f.read((char*)&header_len, 2);
    f.seekg(header_len, std::ios::cur);

    std::vector<float> all_floats;
    float val;
    while (f.read(reinterpret_cast<char*>(&val), sizeof(float)))
        all_floats.push_back(val);

    if (all_floats.size() != 2 * EMBEDDING_DIM) {
        std::cerr << "[Embedder] ERROR: Expected exactly 2 x " << EMBEDDING_DIM
                  << " = " << 2 * EMBEDDING_DIM << " floats, got "
                  << all_floats.size() << "\n";
        return false;
    }

    template_bank.resize(2, std::vector<float>(EMBEDDING_DIM));
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < EMBEDDING_DIM; j++)
            template_bank[i][j] = all_floats[i * EMBEDDING_DIM + j];

    templates_loaded = true;
    std::cout << "[Embedder] Templates loaded: Row0=Hey Saya, Row1=Noise\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// COMPARE_TOPK — margin-based dual gate
//
// Gate 1: saya_dot must exceed the threshold passed by the caller
//         (checked in wake_word_detector.cpp after predict() returns)
// Gate 2: saya_dot must exceed noise_dot by at least NOISE_MARGIN
//         This prevents false triggers when both scores are low
//         (quiet room, non-speech audio, distant speech).
//
// Returns saya_dot if both gates pass, otherwise 0.0
// ════════════════════════════════════════════════════════════════════════════

float WakeWordEmbedder::compare_topk(const float* live_embedding, int /*top_k*/) const {
    if (template_bank.size() < 2) return 0.0f;

    // Dot product = cosine similarity (embeddings are L2-normalized)
    float saya_dot  = 0.0f;
    float noise_dot = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        saya_dot  += live_embedding[i] * template_bank[0][i];
        noise_dot += live_embedding[i] * template_bank[1][i];
    }

    // Diagnostic output — shows live scores on the same terminal line.
    // The #ifdef wrapper has been removed to permanently expose the scores.
    std::cout << "\r[Saya: " << (int)(saya_dot * 100)
              << "% | Noise: " << (int)(noise_dot * 100)
              << "% | Margin: " << (int)((saya_dot - noise_dot) * 100)
              << "%]      " << std::flush;

    // Gate 2: margin filter
    // saya must beat noise by NOISE_MARGIN, not just by any amount
    if ((saya_dot - noise_dot) < NOISE_MARGIN) {
        return 0.0f;
    }

    return saya_dot;
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN FORWARD PASS
// ════════════════════════════════════════════════════════════════════════════

void WakeWordEmbedder::get_embedding(const float* mfcc_features, float* embedding_out)
{
    // Block 1: (1,40,196) → conv → (16,40,196) → pool → (16,20,98)
    conv2d(mfcc_features,
           features_0_weight, features_0_weight_scale,
           features_1_weight, features_1_bias,
           features_1_running_mean, features_1_running_var,
           conv1_out.data(),
           40, 196, 1, 16, 3, 1, 1);
    relu(conv1_out.data(), (int)conv1_out.size());
    maxpool2d(conv1_out.data(), pool1_out.data(), 40, 196, 16, 2, 2);

    // Block 2: (16,20,98) → conv → (32,20,98) → pool → (32,10,49)
    conv2d(pool1_out.data(),
           features_4_weight, features_4_weight_scale,
           features_5_weight, features_5_bias,
           features_5_running_mean, features_5_running_var,
           conv2_out.data(),
           20, 98, 16, 32, 3, 1, 1);
    relu(conv2_out.data(), (int)conv2_out.size());
    maxpool2d(conv2_out.data(), pool2_out.data(), 20, 98, 32, 2, 2);

    // Block 3: (32,10,49) → conv → (64,10,49) → pool → (64,5,24)
    conv2d(pool2_out.data(),
           features_8_weight, features_8_weight_scale,
           features_9_weight, features_9_bias,
           features_9_running_mean, features_9_running_var,
           conv3_out.data(),
           10, 49, 32, 64, 3, 1, 1);
    relu(conv3_out.data(), (int)conv3_out.size());
    maxpool2d(conv3_out.data(), pool3_out.data(), 10, 49, 64, 2, 2);

    // GlobalAvgPool: (64,5,24) → (64,)
    global_avgpool2d(pool3_out.data(), global_pool_out.data(), 5, 24, 64);

    // Linear: 64 → 128
    linear(global_pool_out.data(),
           projector_1_weight, projector_1_weight_scale,
           projector_1_bias,   projector_1_bias_scale,
           proj_out.data(),
           64, EMBEDDING_DIM);

    // L2 Normalize
    memcpy(embedding_out, proj_out.data(), EMBEDDING_DIM * sizeof(float));
    l2_normalize(embedding_out, EMBEDDING_DIM);
}

float WakeWordEmbedder::predict(const float* mfcc_features, int top_k) {
    float embedding[EMBEDDING_DIM];
    get_embedding(mfcc_features, embedding);
    return compare_topk(embedding, top_k);
}

void  WakeWordEmbedder::set_threshold(float t) { threshold = t; }
float WakeWordEmbedder::get_threshold()  const  { return threshold; }