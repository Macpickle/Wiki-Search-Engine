#pragma once
#include <onnxruntime_cxx_api.h>
#include "WordPieceTokenizer.h"

#include <vector>
#include <string>

/*
This class is responsible for embedding text using an ONNX model
*/

class ONNXEmbedder {
private:
	Ort::Env env;                           // ONNX Runtime environment
	Ort::SessionOptions sessionOptions;     // Session options
	Ort::Session session;                   // ONNX Runtime session

	WordPieceTokenizer tokenizer;           // Tokenizer instance
	size_t maxLen;                          // Maximum sequence length

	void normalize(std::vector<float>& v);  // Normalize a vector to unit length

public:
    ONNXEmbedder(
        const std::string& modelPath,
        const std::string& vocabPath,
        size_t maxLen = 256
    );

	// Embed a batch of texts
    std::vector<std::vector<float>>
        embedBatch(const std::vector<std::string>& texts
    );
};