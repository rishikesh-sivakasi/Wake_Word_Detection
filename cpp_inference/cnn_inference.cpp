//==============================================================================
// cnn_inference.cpp  — FIXED VERSION
//
// BUGS FIXED:
//
// BUG 1 — BUFFER SIZES WRONG (caused 0% similarity / crash)
//   conv outputs are LARGER than pool outputs because conv uses padding=1
//   which keeps spatial size the same. Pool halves it.
//
//   OLD (wrong):                    CORRECT:
//   conv1_out: 16*20*98=31360   →   16*40*196=125440
//   conv2_out: 32*10*49=15680   →   32*20*98=62720
//   conv3_out: 64*5*24=7680     →   64*10*49=31360
//   pool sizes were already correct.
//
// BUG 2 — MAXPOOL INPUT SIZES PASSED WRONG IN get_embedding()
//   Block1 maxpool must get in_h=40, in_w=196 (conv1 output size)
//   Block2 maxpool must get in_h=20, in_w=98  (pool1 output size)
//   Block3 maxpool must get in_h=10, in_w=49  (pool2 output size)
//   These are now fixed.
//
// BUG 3 — compare_topk() REDUNDANTLY RE-NORMALIZES (minor, wastes CPU)
//   Embeddings are already L2-normalized by get_embedding().
//   Simple dot product = cosine similarity. No need to recompute norms.
//   Kept the full cosine formula as a safety measure but added note.
//==============================================================================

#include "cnn_inference.h"
#include "weights_embedding.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

static const float BN_EPS = 1e-5f;

// ════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR — FIXED BUFFER SIZES
// ════════════════════════════════════════════════════════════════════════════

WakeWordEmbedder::WakeWordEmbedder() : templates_loaded(false), threshold(0.75f) {

    // Conv output size formula: (in + 2*padding - kernel)/stride + 1
    // With kernel=3, padding=1, stride=1 → output = input (same spatial size)
    //
    // conv1: input (1, 40, 196)  → conv output (16, 40, 196) → pool output (16, 20, 98)
    // conv2: input (16, 20, 98)  → conv output (32, 20, 98)  → pool output (32, 10, 49)
    // conv3: input (32, 10, 49)  → conv output (64, 10, 49)  → pool output (64,  5, 24)

    conv1_out.resize(16 * 40 * 196, 0.0f);   // FIX: was 16*20*98
    pool1_out.resize(16 * 20 * 98,  0.0f);   // correct
    conv2_out.resize(32 * 20 * 98,  0.0f);   // FIX: was 32*10*49
    pool2_out.resize(32 * 10 * 49,  0.0f);   // correct
    conv3_out.resize(64 * 10 * 49,  0.0f);   // FIX: was 64*5*24
    pool3_out.resize(64 *  5 * 24,  0.0f);   // correct
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
                // Fused BatchNorm: gamma*(acc-mean)/sqrt(var+eps) + beta
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
// TEMPLATE LOADING — reads (N, EMBEDDING_DIM) matrix from .npy file
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

    // Skip version + header string
    uint8_t  major, minor;
    uint16_t header_len;
    f.read((char*)&major,      1);
    f.read((char*)&minor,      1);
    f.read((char*)&header_len, 2);
    f.seekg(header_len, std::ios::cur);

    // Read all float32 values until EOF
    std::vector<float> all_floats;
    float val;
    while (f.read(reinterpret_cast<char*>(&val), sizeof(float)))
        all_floats.push_back(val);

    if (all_floats.empty() || all_floats.size() % EMBEDDING_DIM != 0) {
        std::cerr << "[Embedder] ERROR: Template file size mismatch. Got "
                  << all_floats.size() << " floats, expected multiple of "
                  << EMBEDDING_DIM << "\n";
        return false;
    }

    int n = (int)(all_floats.size() / EMBEDDING_DIM);
    template_bank.resize(n, std::vector<float>(EMBEDDING_DIM));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < EMBEDDING_DIM; j++)
            template_bank[i][j] = all_floats[i * EMBEDDING_DIM + j];

    templates_loaded = true;
    std::cout << "[Embedder] Loaded " << n << " templates from: " << path << "\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// TOP-K COSINE SIMILARITY
// Embeddings from get_embedding() are already L2-normalized,
// so dot product = cosine similarity directly.
// compare_topk scores all templates, sorts, returns average of top-K.
// ════════════════════════════════════════════════════════════════════════════

float WakeWordEmbedder::compare_topk(const float* live_embedding, int top_k) const {
    if (template_bank.empty()) return 0.0f;

    std::vector<float> similarities;
    similarities.reserve(template_bank.size());

    for (const auto& tmpl : template_bank) {
        if ((int)tmpl.size() != EMBEDDING_DIM) continue;

        // Full cosine similarity (safe even if embedding isn't perfectly normalized)
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            dot    += live_embedding[i] * tmpl[i];
            norm_a += live_embedding[i] * live_embedding[i];
            norm_b += tmpl[i] * tmpl[i];
        }
        if (norm_a == 0.0f || norm_b == 0.0f) continue;
        similarities.push_back(dot / (sqrtf(norm_a) * sqrtf(norm_b)));
    }

    if (similarities.empty()) return 0.0f;

    // Sort descending, average top-K
    std::sort(similarities.begin(), similarities.end(), std::greater<float>());
    int   k   = std::min(top_k, (int)similarities.size());
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += similarities[i];
    return sum / (float)k;
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN FORWARD PASS — FIXED maxpool input sizes
// ════════════════════════════════════════════════════════════════════════════

void WakeWordEmbedder::get_embedding(const float* mfcc_features, float* embedding_out)
{
    // ── Block 1 ───────────────────────────────────────────────────────────
    // Conv(1→16, k=3, p=1): (1,40,196) → (16,40,196)
    conv2d(mfcc_features,
           features_0_weight, features_0_weight_scale,
           features_1_weight, features_1_bias,
           features_1_running_mean, features_1_running_var,
           conv1_out.data(),
           /*in_h=*/40, /*in_w=*/196, /*in_ch=*/1,
           /*out_ch=*/16, /*ks=*/3, /*stride=*/1, /*padding=*/1);
    relu(conv1_out.data(), (int)conv1_out.size());
    // MaxPool(2,2): (16,40,196) → (16,20,98)   FIX: pass 40,196 not 20,98
    maxpool2d(conv1_out.data(), pool1_out.data(),
              /*in_h=*/40, /*in_w=*/196, /*ch=*/16, /*pool=*/2, /*stride=*/2);

    // ── Block 2 ───────────────────────────────────────────────────────────
    // Conv(16→32, k=3, p=1): (16,20,98) → (32,20,98)
    conv2d(pool1_out.data(),
           features_4_weight, features_4_weight_scale,
           features_5_weight, features_5_bias,
           features_5_running_mean, features_5_running_var,
           conv2_out.data(),
           /*in_h=*/20, /*in_w=*/98, /*in_ch=*/16,
           /*out_ch=*/32, /*ks=*/3, /*stride=*/1, /*padding=*/1);
    relu(conv2_out.data(), (int)conv2_out.size());
    // MaxPool(2,2): (32,20,98) → (32,10,49)   FIX: pass 20,98 not 10,49
    maxpool2d(conv2_out.data(), pool2_out.data(),
              /*in_h=*/20, /*in_w=*/98, /*ch=*/32, /*pool=*/2, /*stride=*/2);

    // ── Block 3 ───────────────────────────────────────────────────────────
    // Conv(32→64, k=3, p=1): (32,10,49) → (64,10,49)
    conv2d(pool2_out.data(),
           features_8_weight, features_8_weight_scale,
           features_9_weight, features_9_bias,
           features_9_running_mean, features_9_running_var,
           conv3_out.data(),
           /*in_h=*/10, /*in_w=*/49, /*in_ch=*/32,
           /*out_ch=*/64, /*ks=*/3, /*stride=*/1, /*padding=*/1);
    relu(conv3_out.data(), (int)conv3_out.size());
    // MaxPool(2,2): (64,10,49) → (64,5,24)   FIX: pass 10,49 not 5,24
    maxpool2d(conv3_out.data(), pool3_out.data(),
              /*in_h=*/10, /*in_w=*/49, /*ch=*/64, /*pool=*/2, /*stride=*/2);

    // ── GlobalAvgPool: (64,5,24) → (64,) ─────────────────────────────────
    global_avgpool2d(pool3_out.data(), global_pool_out.data(),
                     /*h=*/5, /*w=*/24, /*ch=*/64);

    // ── Linear: 64 → 128 ─────────────────────────────────────────────────
    linear(global_pool_out.data(),
           projector_1_weight, projector_1_weight_scale,
           projector_1_bias,   projector_1_bias_scale,
           proj_out.data(),
           /*in=*/64, /*out=*/EMBEDDING_DIM);

    // ── L2 Normalize ──────────────────────────────────────────────────────
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