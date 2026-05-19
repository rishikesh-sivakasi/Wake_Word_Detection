#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

#include <vector>
#include <string>
#include <stdint.h>

#define EMBEDDING_DIM 128

class WakeWordEmbedder {
public:
    WakeWordEmbedder();
    ~WakeWordEmbedder();

    void get_embedding(const float* mfcc_features, float* embedding_out);

    // NEW: Load a 2D matrix of templates
    bool load_templates(const std::string& template_path);

    // NEW: Compare against all templates and average the top_k
    float compare_topk(const float* embedding, int top_k) const;

    float predict(const float* mfcc_features, int top_k = 3);

    void  set_threshold(float t);
    float get_threshold() const;
    int   num_templates() const;

private:
    std::vector<float> conv1_out;
    std::vector<float> pool1_out;
    std::vector<float> conv2_out;
    std::vector<float> pool2_out;
    std::vector<float> conv3_out;
    std::vector<float> pool3_out;
    std::vector<float> global_pool_out;
    std::vector<float> proj_out;         

    // NEW: 2D vector to hold ALL registered samples
    std::vector<std::vector<float>> template_bank;
    bool  templates_loaded;

    float threshold;

    void conv2d(
        const float*   input,
        const int8_t*  weights,   float weight_scale,
        const float*   bn_gamma,
        const float*   bn_beta,
        const float*   bn_mean,
        const float*   bn_var,
        float*         output,
        int in_h, int in_w, int in_channels,
        int out_channels, int kernel_size,
        int stride, int padding
    );

    void relu(float* data, int size);
    void maxpool2d(const float* input, float* output, int in_h, int in_w, int channels, int pool_size, int stride);
    void global_avgpool2d(const float* input, float* output, int h, int w, int channels);
    void linear(const float* input, const int8_t* weight, float weight_scale, const int8_t* bias, float bias_scale, float* output, int in_features, int out_features);
    void l2_normalize(float* vec, int size);
};

#endif // CNN_INFERENCE_H