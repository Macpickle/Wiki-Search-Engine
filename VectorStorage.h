#pragma once
#include "ONNXEmbedder.h"
#include "hnswlib/hnswlib.h"
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
#include <hnswlib/space_l2.h>
#include <hnswlib/hnswalg.h>

/*
This class manages vector storage, including embedding texts, searching, and indexing using HNSW.
*/

constexpr size_t DIM = 384;                 // Dimension of embeddings
constexpr size_t MAX_ELEMENTS = 2'000'000;  // Maximum number of elements in HNSW index

// Holds search result
struct SearchResult {
    int64_t id;
	float score;
    std::string title;
    std::string description;
    std::string link;
};

// Candidate struct for ranking
struct Candidate {
    int64_t id;
    float score;
    std::string title;
    std::string desc;
    std::string link;
};

class VectorStorage {
private:
	pqxx::connection& conn;                 // Database connection
	std::string indexPath;                  // Path to HNSW index file
	size_t threadCount;                     // Number of threads for parallel processing

	httplib::Client client;                 // HTTP client for embedding server

	hnswlib::L2Space space;                 // L2 space for HNSW
	hnswlib::HierarchicalNSW<float> index;  // HNSW index
	std::mutex hnswMutex;                   // Mutex for thread-safe HNSW access

	std::unique_ptr<ONNXEmbedder> embedder; // ONNX embedder instance

	const std::unordered_set<std::string> stopwords = { "the", "of", "and", "to", "in", "a", "is", "for", "on", "with" }; // basic stopwords
	const int topHNSW = 200;     // number of HNSW neighbors to retrieve

	// search helper, is a basic COSINE similarity search fallback
    std::vector<SearchResult> defaultSearch(std::vector<float>& queryEmbedding, size_t topK);

	// Embedding a batch of articles
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

	// Insert a batch of vectors and metadata into the database
    std::vector<int64_t> insertBatch(
        const std::vector<PageItem>& pages,
        const std::vector<std::vector<float>>& embeddings
    );

    // HELPERS
    float scoreCandidate(
        const size_t queryTokenCount,
        const std::vector<int64_t> candidateHashes,
        const std::unordered_set<int64_t> queryHashSet,
        const float knnScore
    );

	// String cleaning
    std::string cleanString(const std::string& text);

	// Tokenization
    std::unordered_set<std::string> tokenizeText(const std::string& text);

	// Hashing tokens, for BIGINT[] storage
    std::vector<int64_t > hashTokens(const std::unordered_set<std::string>& tokens);

	// Convert vector to Postgres vector string
    std::string VectorToPGVector(const std::vector<float>& v);

    // Normalize a token (remove s, ing, y)
    std::string extractEntity(const std::string& query);

public:
    VectorStorage(pqxx::connection& conn,
        const std::string& indexPath,
        size_t threadCount);

	// Public API
    void ingestBatch(const std::vector<PageItem>& pages);

    std::vector<SearchResult> search(
        const std::string& query, 
        size_t topK = 5
    );

	// Embed a single text
    std::vector<float> EmbedText(const std::string& text);

	void save();    // Save HNSW index to disk
	void load();    // Load HNSW index from disk
};
