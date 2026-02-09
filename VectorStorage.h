#pragma once
#include "ONNXEmbedder.h"
#include "PageItem.h"

#include <vector>
#include <mutex>
#include <httplib.h>
#include <unordered_set>
#include <string>
#include <memory>
#include <cstdint>
#include <pqxx/pqxx>
#include <pqxx/connection.hxx>

/*
This class manages vector storage, including embedding texts, searching, and indexing using HNSW.
*/

constexpr size_t DIM = 384;                 // Dimension of embeddings
constexpr size_t MAX_ELEMENTS = 2'000'000;  // Maximum number of elements in HNSW index

// Holds search result
struct SearchResult {
    int64_t id;
    float score = 0.0f;
    std::string title;
    std::string description;
    std::string link;
};

// Holds token hash and frequency for a document, used for token overlap scoring
struct TokenStat {
    int64_t hash;
    int16_t freq;
};

class VectorStorage {
public:
    VectorStorage(
        pqxx::connection& conn,
        size_t threadCount
    );

    void ingestBatch(const std::vector<PageItem>& pages);

    std::vector<SearchResult> search(
        const std::string& query,
        size_t topK
    );

private:
    pqxx::connection& conn;
    size_t threadCount;

    std::unique_ptr<ONNXEmbedder> embedder;

    std::vector<int64_t> insertBatch(
        const std::vector<PageItem>& pages,
        const std::vector<std::vector<float>>& embeddings
    );

    std::vector<std::vector<float>> embedBatch(
        const std::vector<std::string>& texts
    );

    std::vector<float> EmbedText(
        const std::string& text
    );

    std::string VectorToPGVector(
        const std::vector<float>& v
    );

    std::unordered_map<std::string, int> tokenizeWithFrequency(
        const std::string& text
    );

    std::string buildTokenStatArray(
        const std::unordered_map<std::string, int>& tokenFreq
    );

    std::unordered_set<std::string> tokenizeText(
        const std::string& text
    );

    std::vector<int64_t> hashTokens(
        const std::unordered_set<std::string>& tokens
    );

    std::string extractEntity(
        const std::string& query
    );

    std::unordered_map<int64_t, int> parseTokenStats(
        const pqxx::field& field
    );

    float keywordScore(
        const std::unordered_set<int64_t>& queryHashes,
        const std::unordered_map<int64_t, int>& docFreqs
    );

    float titleScore(
        const std::string& cleanTitle,
        const std::unordered_set<std::string>& queryTokens,
        const std::string& cleanQuery
    );

    std::string cleanString(
        const std::string& text
    );

private:
    const std::unordered_set<std::string> stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "of", "to", "in", "on", "for", "with",
        "what", "who", "when", "where", "why", "how",
        "define", "definition", "explain"
    };

    httplib::Client client;                 // HTTP client for embedding server
};