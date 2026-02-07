#include "VectorStorage.h"
#include "PageItem.h"
#include "ONNXEmbedder.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <vector>
#include <httplib.h>
#include <utility>
#include <unordered_set>
#include <string>
#include <mutex>
#include <memory>
#include <map>
#include <ios>
#include <exception>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <pqxx/transaction.hxx>
#include <pqxx/result.hxx>
#include <pqxx/prepared_statement.hxx>
#include <pqxx/params.hxx>
#include <pqxx/connection.hxx>
#include <nlohmann/json_fwd.hpp>

#include <chrono>

using json = nlohmann::json;    // for JSON handling

// Constructor
VectorStorage::VectorStorage(
    pqxx::connection& conn,
    const std::string& indexPath,
    size_t threadCount)
    : conn(conn),
    indexPath(indexPath),
    threadCount(threadCount),
    space(DIM),
    index(&space, MAX_ELEMENTS),
    client("localhost", 8000)
{
    client.set_connection_timeout(8);
    client.set_read_timeout(8);

    // DB initial check
    pqxx::work w(conn);
    w.exec("CREATE EXTENSION IF NOT EXISTS vector;");
    w.exec(R"(
        CREATE TABLE IF NOT EXISTS vectors (
            id SERIAL PRIMARY KEY,
            title TEXT,
            description TEXT,
            link TEXT,
            embedding vector(384),
            token_hashes BIGINT[]
        );
    )");

	// Prepare insert statement
    conn.prepare(
        "insert_vector",
        "INSERT INTO vectors (title, description, link, embedding, token_hashes) "
        "VALUES ($1, $2, $3, $4::vector, $5)"
        "RETURNING id"
    );

    w.commit();
    this->load(); 	                            // Attempt to load previously computed HNSW file

    try {
        embedder = std::make_unique<ONNXEmbedder>(
            "./models/model.onnx",              // ModelPath
            "./models/vocab.txt",               // VocabPath
            128                                 // Max length
        );
        std::cout << "ONNX model and tokenizer loaded successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model or tokenizer: " << e.what() << std::endl;
        exit(1); // Exit with a code indicating failure
    }
}

// Save HNSW index
void VectorStorage::save() {
    std::lock_guard<std::mutex> lock(hnswMutex);
    index.saveIndex(indexPath);
}

// Attempt to load HNSW index
void VectorStorage::load() {
    try {
        index.loadIndex(indexPath, &space);
    }
    catch (...) {
        std::cerr << "HNSW index not found, starting fresh.\n";
    }
}

// Public API
void VectorStorage::ingestBatch(const std::vector<PageItem>& pages) {
    if (pages.empty()) return;

    // Embed texts in parallel
    std::vector<std::string> texts;
    texts.reserve(pages.size());
    for (const auto& p : pages) texts.push_back(p.text);

    std::vector<std::vector<float>> embeddings = embedBatch(texts);

    if (embeddings.empty()) {
        std::cerr << "Embedding failed for entire batch, skipping.\n";
        return;
    }

    // Filter out failed embeddings
    std::vector<PageItem> validPages;
    std::vector<std::vector<float>> validEmbeddings;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (!embeddings[i].empty()) {
            validPages.push_back(pages[i]);
            validEmbeddings.push_back(std::move(embeddings[i]));
        }
        else {
            std::cerr << "Skipping article due to embedding failure: " << pages[i].title << "\n";
        }
    }
    if (validPages.empty()) return;

    // Insert into DB
    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> ids = insertBatch(validPages, validEmbeddings);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
	std::cout << "Inserted batch into DB in " << elapsed2.count() << " seconds.\n";

    {
        for (size_t i = 0; i < validEmbeddings.size(); ++i) {
            index.addPoint(validEmbeddings[i].data(), ids[i]);
        }
    }
}

// DB insert
std::vector<int64_t> VectorStorage::insertBatch(
    const std::vector<PageItem>& pages,
    const std::vector<std::vector<float>>& embeddings)
{
    if (pages.empty()) return {};

    pqxx::work w(conn);
    std::ostringstream sql;
    sql << "INSERT INTO vectors (title, description, link, embedding, token_hashes) VALUES ";

    for (size_t i = 0; i < pages.size(); ++i) {
        if (i > 0) sql << ", ";

        sql << "("
            << w.quote(pages[i].title) << ", "
            << w.quote(pages[i].text) << ", "
            << w.quote(pages[i].link) << ", "
            << w.quote(VectorToPGVector(embeddings[i])) << "::vector, "
            << w.quote(hashTokens(tokenizeText(pages[i].text)))
            << ")";
    }
    sql << " RETURNING id";
    pqxx::result r = w.exec(sql.str());
    w.commit();

    std::vector<int64_t> ids;
    for (auto const& row : r) ids.push_back(row["id"].as<int64_t>());
    return ids;
}

// Embedding
std::vector<std::vector<float>> VectorStorage::embedBatch(const std::vector<std::string>& texts) {
    return embedder->embedBatch(texts);
}

std::vector<float> VectorStorage::EmbedText(const std::string& text) {
    return embedder->embedBatch({ text })[0];
}

// Helper default search, regular COSINE similarity search
std::vector<SearchResult> VectorStorage::defaultSearch(std::vector<float>& queryEmbedding, size_t topK) {
    std::vector<SearchResult> results;

    pqxx::work worker(conn);
    std::ostringstream sql;

    std::cout << "USING DEFAULT\n";

    sql <<
        "SELECT id, title, description, link, "
		"embedding <-> '" << VectorToPGVector(queryEmbedding) << "' AS distance "
        "FROM vectors "
        "ORDER BY distance ASC "
        "LIMIT " << topK;

    pqxx::result r = worker.exec(sql.str());

    for (auto row : r) {
        results.push_back({
            row["id"].as<int64_t>(),
			row["score"].as<float>(),
            row["title"].c_str(),
            row["description"].c_str(),
            row["link"].c_str()
         });
    }

	return results;
}

std::vector<SearchResult> VectorStorage::search(const std::string& query, size_t topK) {
    // Query processing
    std::string normalizedQuery = cleanString(query);
    std::unordered_set<std::string> queryTokens = tokenizeText(normalizedQuery);
    std::string entityQuery = extractEntity(query);
    std::vector<float> queryEmbedding = EmbedText(entityQuery);

    std::vector<int64_t> queryHashes = hashTokens(queryTokens);
    std::unordered_set<int64_t> queryHashSet(queryHashes.begin(), queryHashes.end());

    if (queryEmbedding.empty()) return {};

    // Check for valid load of HNSW index
    std::lock_guard<std::mutex> lock(hnswMutex);
    if (index.cur_element_count == 0) {
        std::cout << "HNSW index is empty. Using basic search instead.\n";          // if HNSW index is empty, fallback to default search
        return defaultSearch(queryEmbedding, topK);
    }

    // HNSW + Keyword Ranking Search
    index.setEf(200);                                                                // higher ef for better accuracy
    auto knn = index.searchKnn(queryEmbedding.data(), topHNSW);

    // Map KNN results
    std::vector<int64_t> ids;
    std::map<int64_t, float> knnScores;
    while (!knn.empty()) {
        auto [dist, id] = knn.top();
        ids.push_back(id);
        knnScores[id] = 1.0f / (1.0f + dist); // Convert distance to similarity
        knn.pop();
    }

    pqxx::work worker(conn);

    // Fetch metadata from DB
    std::ostringstream sql;
    sql << "SELECT id, title, description, link, token_hashes FROM vectors WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) sql << ",";
        sql << ids[i];
    }
    sql << ")";

    pqxx::result r = worker.exec(sql.str());

    // Rank candidates algorithm
    std::vector<Candidate> candidates;
    for (auto const& row : r) {
        int64_t id = row["id"].as<int64_t>();

        std::vector<int64_t> candidateHashes;
        const char* p = row["token_hashes"].c_str();
        while (*p) {
            if (isdigit(*p) || *p == '-') {
                char* end;
                candidateHashes.push_back(std::strtoll(p, &end, 10));
                p = end;
            }
            else {
                p++;
            }
        }

        // Check matches in title
        std::string rowTitle = row["title"].as<std::string>();
        std::unordered_set<std::string> titleTokens = tokenizeText(cleanString(rowTitle));
        int titleMatches = 0;
        for (const auto& t : queryTokens) { // queryTokens from top of search()
            if (titleTokens.count(t)) titleMatches++;
        }

		// Calculate score
        float score = scoreCandidate(queryHashSet.size(), candidateHashes, queryHashSet, knnScores[id]);

		score += static_cast<float>(titleMatches) * 0.5f;                               // Title match weight
        if (rowTitle == normalizedQuery) score += 10.0f;                                // Exact match bonus
        else if (rowTitle.find(normalizedQuery) != std::string::npos) score += 5.0f;    // Substring bonus

        candidates.push_back({
            id,
            score,
            row["title"].c_str(),
            row["description"].c_str(),
            row["link"].c_str()
        });
    }

    std::vector<SearchResult> results;  // final result vector

    // Sort by score
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

    // Return topK
    for (size_t i = 0; i < candidates.size() && i < topK; ++i) {
        results.push_back({
            candidates[i].id,
            candidates[i].score,
            candidates[i].title,
            candidates[i].desc,
            candidates[i].link
        });
    }

    // If keyword ranking found nothing, fallback
    if (results.empty()) {
        return defaultSearch(queryEmbedding, topK);
    }

    return results;
}

// Converts a vector to a string, used for SQL queries
std::string VectorStorage::VectorToPGVector(const std::vector<float>& v) {
    std::ostringstream vec;
    vec << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) vec << ",";
        vec << std::fixed << std::setprecision(6) << v[i];
    }
    vec << "]";
    return vec.str();
}

// Scores candidate based on tokenization
float VectorStorage::scoreCandidate(
    const size_t queryTokenCount,
    const std::vector<int64_t> candidateHashes,
    const std::unordered_set<int64_t> queryHashSet,
    const float knnScore
) {
	// Keyword overlap score (Jaccard similarity)
    int intersectionCount = 0;
    for (int64_t h : candidateHashes) {
        if (queryHashSet.find(h) != queryHashSet.end()) {
            intersectionCount++;
        }
    }

    float keywordScore = static_cast<float>(intersectionCount) /
        static_cast<float>(queryTokenCount + candidateHashes.size() - intersectionCount);

    return (knnScore * 0.3f) + (keywordScore * 0.7f);
}

std::string VectorStorage::cleanString(const std::string& text) {
    std::string clean = text;
    // Lowercase
    std::transform(clean.begin(), clean.end(), clean.begin(),
        [](unsigned char c) { return std::tolower(c); });

    // Remove punctuation
    clean.erase(std::remove_if(clean.begin(), clean.end(),
        [](unsigned char c) { return std::ispunct(c); }), clean.end());

    return clean;
}

// Normalize text for tokenization
std::unordered_set<std::string> VectorStorage::tokenizeText(const std::string& text) {
    std::string cleaned = cleanString(text);

    std::stringstream ss(cleaned);
    std::unordered_set<std::string> tokens;
    std::string word;

    while (ss >> word) {
		// length filter + simple stemming
        if (word.ends_with("s") && word.length() > 3) {
            word.pop_back();
        }

        // stopword filter
        if (stopwords.find(word) != stopwords.end()) continue;
        tokens.insert(word);
    }

    return tokens;
}

// Creates hash of a set
std::vector<int64_t> VectorStorage::hashTokens(const std::unordered_set<std::string>& tokens) {
    std::vector<int64_t> hashes;
    hashes.reserve(tokens.size());

    std::hash<std::string> hasher;
    for (const auto& token : tokens) {
        uint64_t h = hasher(token);                 // 64-bit unsigned hash
        hashes.push_back(static_cast<int64_t>(h));  // cast to signed 64-bit
    }

    return hashes;
}

std::string VectorStorage::extractEntity(const std::string& query) {
    static const std::vector<std::string> prefixes = {
        "what is", "what are", "define", "definition of", "explain"
    };

    std::string q = cleanString(query);

    for (const auto& p : prefixes) {
        if (q.rfind(p, 0) == 0) {
            return q.substr(p.size());
        }
    }

    return q;
}