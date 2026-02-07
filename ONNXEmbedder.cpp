#include "ONNXEmbedder.h"
#include <cmath>
#include <numeric>
#include <vector>
#include <utility>
#include <string>
#include <cstdint>
#include <array>
#include <thread>
#include "packages/Microsoft.ML.OnnxRuntime.1.23.2/build/native/include/onnxruntime_c_api.h"
#include "packages/Microsoft.ML.OnnxRuntime.1.23.2/build/native/include/onnxruntime_cxx_api.h"

// Convert std::string to ORTCHAR_T*, needed for Windows compatibility
#ifdef _WIN32
inline static const ORTCHAR_T* ToOrtString(const std::string& s) {
    static std::wstring ws;
    ws.assign(s.begin(), s.end());
    return ws.c_str();
}
#else
inline static const ORTCHAR_T* ToOrtString(const std::string& s) {
    return s.c_str();
}
#endif

// Constructor
ONNXEmbedder::ONNXEmbedder(
    const std::string& modelPath,
    const std::string& vocabPath,
    size_t maxLen)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXEmbedder"),
    sessionOptions(),
    session(nullptr),
    tokenizer(vocabPath),
    maxLen(maxLen)
{
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(
        env,
        ToOrtString(modelPath),
        sessionOptions);
}

// Embed a batch of texts
std::vector<std::vector<float>> ONNXEmbedder::embedBatch(const std::vector<std::string>& texts) {
    size_t B = texts.size();
    if (B == 0) return {};

    std::vector<int64_t> flat_ids(B * maxLen);
    std::vector<int64_t> flat_mask(B * maxLen);
    std::vector<int64_t> flat_types(B * maxLen, 0);

	// Tokenize each text
    for (size_t i = 0; i < B; ++i) {
        auto encoded = tokenizer.encode(texts[i], maxLen);

        for (size_t j = 0; j < maxLen; ++j) {
            int64_t id = encoded[j];
            flat_ids[i * maxLen + j] = id;
            flat_mask[i * maxLen + j] = (id != tokenizer.pad_id);
        }
    }

	// Create input tensors
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape{
        static_cast<int64_t>(B),
        static_cast<int64_t>(maxLen)
    };

    Ort::Value idsTensor = Ort::Value::CreateTensor<int64_t>(
        mem, flat_ids.data(), flat_ids.size(), shape.data(), 2);

    Ort::Value maskTensor = Ort::Value::CreateTensor<int64_t>(
        mem, flat_mask.data(), flat_mask.size(), shape.data(), 2);

    Ort::Value typeTensor = Ort::Value::CreateTensor<int64_t>(
        mem, flat_types.data(), flat_types.size(), shape.data(), 2);

    // Run inference
    const char* inputNames[] = {
        "input_ids",
        "attention_mask",
        "token_type_ids"
    };

    Ort::Value inputTensors[] = {
        std::move(idsTensor),
        std::move(maskTensor),
        std::move(typeTensor)
    };

    const char* outputNames[] = {
        "token_embeddings"
    };

	// Execute the model
    auto outputs = session.Run(
        Ort::RunOptions{ nullptr },
        inputNames,
        inputTensors,
        3,
        outputNames,
        1
    );

	// Process output tensor
    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto outShape = info.GetShape();

    int64_t hidden = outShape.back();
    float* data = out.GetTensorMutableData<float>();

	// Prepare result vector
    std::vector<std::vector<float>> result(
        B, std::vector<float>(hidden)
    );

	// Compute mean pooling, ignoring padding tokens
    for (size_t i = 0; i < B; ++i) {
        float* start = data + i * maxLen * hidden;
        std::vector<float> sum(hidden, 0.0f);
        int count = 0;
        for (size_t j = 0; j < maxLen; ++j) {
            int64_t id = flat_ids[i * maxLen + j];
            if (id != tokenizer.pad_id) {
                for (size_t k = 0; k < hidden; ++k) {
                    sum[k] += start[j * hidden + k];
                }
                count++;
            }
        }
        for (auto& x : sum) x /= count;
        normalize(sum);
        result[i] = std::move(sum);
    }

    return result;
}

// Normalize a vector to unit length
void ONNXEmbedder::normalize(std::vector<float>& v) {
    float norm = std::sqrt(
        std::accumulate(
            v.begin(), v.end(), 0.0f,
            [](float sum, float val) {
                return sum + val * val;
            }
        )
    );
    if (norm > 0.0f) {
        for (auto& x : v) {
            x /= norm;
        }
    }
}