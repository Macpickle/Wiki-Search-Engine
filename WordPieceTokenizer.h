#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

/*
This class implements a WordPiece tokenizer, used on ONNX models
*/

class WordPieceTokenizer {
private:
	std::unordered_map<std::string, int64_t> vocab; // token to ID mapping

public:
    explicit WordPieceTokenizer(const std::string& vocabPath);

	// Encode text into token IDs with padding/truncation
    std::vector<int64_t> encode(
        const std::string& text,
        size_t maxLen
    ) const;

	int64_t pad_id, cls_id, sep_id, unk_id;         // special token IDs
};