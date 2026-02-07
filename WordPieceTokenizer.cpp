#include "WordPieceTokenizer.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>

// Constructor
WordPieceTokenizer::WordPieceTokenizer(const std::string& vocabPath) {
    std::ifstream f(vocabPath);
    std::string token;
    int64_t id = 0;

	// Load vocab file and build token to ID mapping
    while (std::getline(f, token)) {
        vocab[token] = id++;
    }

	// Set special token IDs
    pad_id = vocab["[PAD]"];    
    cls_id = vocab["[CLS]"];
    sep_id = vocab["[SEP]"];
    unk_id = vocab["[UNK]"];
}

// Encode text into token IDs with padding/truncation
std::vector<int64_t> WordPieceTokenizer::encode(const std::string& text, size_t maxLen) const {
    std::vector<int64_t> ids;
    ids.reserve(maxLen);

    ids.push_back(cls_id);

    std::istringstream iss(text);
    std::string word;
    while (iss >> word && ids.size() < maxLen - 1) {
        auto it = vocab.find(word);
        ids.push_back(it != vocab.end() ? it->second : unk_id);
    }

    ids.push_back(sep_id);
    ids.resize(maxLen, pad_id);

    return ids;
}
