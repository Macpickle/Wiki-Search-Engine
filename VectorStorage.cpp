#include "VectorStorage.h"
#include "PageItem.h"
#include "ONNXEmbedder.h"

#include <pqxx/connection.hxx>
#include <pqxx/transaction.hxx>
#include <pqxx/result.hxx>
#include <pqxx/field.hxx>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <chrono>

// Constructor
VectorStorage::VectorStorage(pqxx::connection& conn, size_t threadCount)
    : conn(conn),
    threadCount(threadCount),
    client("localhost", 8000)
{
    client.set_connection_timeout(8);
    client.set_read_timeout(8);

    pqxx::work w(conn);

	// Create vector extension if it doesn't exist
    w.exec("CREATE EXTENSION IF NOT EXISTS vector;");

	// Create custom type for token stats if it doesn't exist
    w.exec(R"(
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_type
                WHERE typname = 'token_stat'
            ) THEN
                CREATE TYPE token_stat AS (
                    hash BIGINT,
                    freq SMALLINT
                );
            END IF;
        END$$;
    )");

	// Create table in DB if it doesn't exist already
    w.exec(R"(
        CREATE TABLE IF NOT EXISTS vectors (
            id SERIAL PRIMARY KEY,
            title TEXT,
            description TEXT,
            link TEXT,
            embedding vector(384),
            token_stats token_stat[]
        );
    )");

	// Create HNSW index on embedding column if it doesn't exist
    w.exec(R"(
        CREATE INDEX IF NOT EXISTS idx_vectors_embedding_hnsw
        ON vectors
        USING hnsw (embedding vector_cosine_ops);
    )");

    w.exec("SET hnsw.ef_search = 64");
    w.commit();

	// initialize ONNX embedder
    embedder = std::make_unique<ONNXEmbedder>(
        "./models/model.onnx",
        "./models/vocab.txt",
        128
    );
}

// Ingest batch of data into DB
void VectorStorage::ingestBatch(const std::vector<PageItem>& pages) 
{
    if (pages.empty()) return;

    std::vector<std::string> texts;
    texts.reserve(pages.size());
    for (const auto& p : pages) {
        texts.push_back(p.text);
    }

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

    std::vector<int64_t> ids = insertBatch(validPages, validEmbeddings);
}

// DB insert
std::vector<int64_t> VectorStorage::insertBatch(
    const std::vector<PageItem>& pages,
    const std::vector<std::vector<float>>& embeddings)
{
    pqxx::work w(conn);
    std::ostringstream sql;
    sql << "INSERT INTO vectors (title, description, link, embedding, token_stats) VALUES ";

    for (size_t i = 0; i < pages.size(); ++i) {
        if (i > 0) sql << ", ";

        auto tokenFreq = tokenizeWithFrequency(pages[i].text);
        std::string tokenStatArray = buildTokenStatArray(tokenFreq);

        sql << "("
            << w.quote(cleanString(pages[i].title)) << ", "
            << w.quote(pages[i].text) << ", "
            << w.quote(pages[i].link) << ", "
            << w.quote(VectorToPGVector(embeddings[i])) << "::vector, "
            << tokenStatArray
            << ")";
    }
    sql << " RETURNING id";
    pqxx::result r = w.exec(sql.str());
    w.commit();

    std::vector<int64_t> ids;
    for (auto const& row : r) ids.push_back(row["id"].as<int64_t>());
    return ids;
}

// Embedding batch of texts using ONNX embedder
std::vector<std::vector<float>> VectorStorage::embedBatch(const std::vector<std::string>& texts) {
    return embedder->embedBatch(texts);
}

// Embedding single text
std::vector<float> VectorStorage::EmbedText(const std::string& text) {
    return embedder->embedBatch({ text })[0];
}

// Public search API - performs vector search + token matching + title heuristics
std::vector<SearchResult> VectorStorage::search(
    const std::string& query,
    size_t topK)
{
    std::string cleanQuery = cleanString(query);
    std::unordered_set<std::string> queryTokens = tokenizeText(cleanQuery);

    std::vector<int64_t> queryHashVec = hashTokens(queryTokens);

    std::unordered_set<int64_t> queryHashes(
        queryHashVec.begin(),
        queryHashVec.end()
    );

    std::string entityQuery = extractEntity(query);

    auto queryEmbedding = EmbedText(entityQuery);
    if (queryEmbedding.empty()) return {};

    std::string queryVec = VectorToPGVector(queryEmbedding);
    size_t expandedK = std::max(topK, static_cast<size_t>(topK * 1.5));

    pqxx::work w(conn);
    std::ostringstream sql;

    sql <<
        "SELECT id "
        "FROM vectors "
        "ORDER BY embedding <=> $1::vector "
        "LIMIT $2";

    pqxx::params p1;
    p1.append(queryVec);
    p1.append(expandedK);

    pqxx::result r = w.exec(sql.str(), p1);
    if (r.empty()) return {};

	// get all fields for the top results to compute final scores
    std::vector<int64_t> topIds;
    topIds.reserve(r.size());
    for (auto const& row : r) {
        topIds.push_back(row["id"].as<int64_t>());
    }

    std::ostringstream detailSql;
    detailSql <<
        "SELECT id, title, description, link, "
        "token_stats::text AS token_stats, "
        "1.0 / (1.0 + (embedding <=> $1::vector)) AS knn_score "
        "FROM vectors "
        "WHERE id = ANY($2)";

    pqxx::params p2;
    p2.append(queryVec);
    p2.append(topIds);

    pqxx::result detailedResults = w.exec(detailSql.str(), p2);

	std::vector<SearchResult> results;
    results.reserve(r.size());

	// Combine KNN score with token overlap and title heuristics for final scoring
    for (auto const& row : detailedResults) {
        float knnScore = row["knn_score"].as<float>();

        auto freqs = parseTokenStats(row["token_stats"]);
        float keyword = keywordScore(queryHashes, freqs);

        std::string title = row["title"].as<std::string>();
        std::string cleanTitle = cleanString(title);

        std::unordered_set<std::string> titleTokens = tokenizeText(cleanString(title));

        float titleBoost = titleScore(
            cleanTitle,
            queryTokens,
            cleanQuery
        );

        float finalScore =
            knnScore * 0.55f +
            keyword * 0.30f +
            titleBoost * 0.15f;

        results.push_back({
            row["id"].as<int64_t>(),
            finalScore,
            title,
            row["description"].as<std::string>(),
            row["link"].as<std::string>()
        });
    }

	// Sort results by final score and return top K
    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

	// Resize to topK if we got more results from DB due to expandedK
    if (results.size() > topK)
        results.resize(topK);

    return results;
}

std::string VectorStorage::cleanString(const std::string& text) 
{
    std::string out;
    out.reserve(text.size());

    for (unsigned char c : text) {
        if (!std::ispunct(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }

    return out;
}

// Normalize text for tokenization, also count frequency
std::unordered_set<std::string> VectorStorage::tokenizeText(const std::string& text) 
{
    std::stringstream ss(text);
    std::unordered_set<std::string> tokens;
    std::string word;

    while (ss >> word) {
        if (word.size() > 3 && word.ends_with("s")) {
            word.pop_back();
        }
        if (!stopwords.contains(word)) {
            tokens.insert(word);
        }
    }

    return tokens;
}

// Tokenize text and count frequency of each token, used for token_stats column
std::unordered_map<std::string, int> VectorStorage::tokenizeWithFrequency(const std::string& text) 
{
    std::stringstream ss(cleanString(text));
    std::unordered_map<std::string, int> freq;
    std::string word;

    while (ss >> word) {
        if (word.size() > 3 && word.ends_with("s")) {
            word.pop_back();
        }
        if (!stopwords.contains(word)) {
            freq[word]++;
        }
    }

    return freq;
}

// Extracts main entity from query by removing common question words and stopwords, also normalizes text
std::string VectorStorage::extractEntity(const std::string& query)
{
    static const std::vector<std::string> prefixes = {
        "what is", "what are", "define", "definition of", "explain"
    };

    std::string q = cleanString(query);

    for (const auto& p : prefixes) {
        if (q.rfind(p, 0) == 0) {
            q.erase(0, p.size());
            break;
        }
    }

    std::stringstream ss(q);
    std::string word;
    std::vector<std::string> words;

    while (ss >> word) {
        if (word == "a" || word == "an" || word == "the") continue;
        if (word.size() > 3 && word.ends_with("s")) word.pop_back();
        words.push_back(word);
    }

    std::ostringstream out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out << " ";
        out << words[i];
    }

    return out.str();
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

// Parses the token_stats field from the database, converting the array of (hash, freq) tuples back into a map of hash to frequency
std::unordered_map<int64_t, int> VectorStorage::parseTokenStats(const pqxx::field& field)
{
    std::unordered_map<int64_t, int> freqMap;

    if (field.is_null()) return freqMap;

    std::string_view s = field.view();
    size_t i = 0;

    while ((i = s.find('(', i)) != std::string_view::npos) {
        ++i;

        // parse hash
        size_t comma = s.find(',', i);
        int64_t hash = std::stoll(std::string(s.substr(i, comma - i)));

        // parse freq
        size_t close = s.find(')', comma);
        int freq = std::stoi(std::string(s.substr(comma + 1, close - comma - 1)));

        freqMap[hash] = freq;
        i = close + 1;
    }

    return freqMap;
}

// Computes a keyword score based on the overlap of hashed query tokens and document token frequencies, using logarithmic scaling for frequency and normalizing by the number of query tokens
float VectorStorage::keywordScore(
    const std::unordered_set<int64_t>& queryHashes,
    const std::unordered_map<int64_t, int>& docFreqs
) {
    if (queryHashes.empty() || docFreqs.empty()) return 0.0f;

    float score = 0.0f;

    for (int64_t q : queryHashes) {
        auto it = docFreqs.find(q);
        if (it != docFreqs.end()) {
            score += std::log1p(static_cast<float>(it->second));
        }
    }

    return score / static_cast<float>(queryHashes.size());
}

// Computes a title score based on exact match, partial match, and token overlap between the query and the document title
float VectorStorage::titleScore(
    const std::string& cleanTitle,
    const std::unordered_set<std::string>& queryTokens,
    const std::string& cleanQuery
) {
    float score = 0.0f;

    if (cleanTitle == cleanQuery)
        return 2.5f;               // exact match wins immediately

    if (cleanTitle.find(cleanQuery) != std::string::npos)
        score += 1.5f;             // strong partial match

    if (!queryTokens.empty()) {
        int overlap = 0;
        for (const auto& t : queryTokens)
            overlap += (cleanTitle.find(t) != std::string::npos);

        score += static_cast<float>(overlap) /
            static_cast<float>(queryTokens.size());
    }

    return score;
}

// Used to build the token_stats array for SQL insertion, converts token frequencies to an array of (hash, freq) tuples
std::string VectorStorage::buildTokenStatArray(
    const std::unordered_map<std::string, int>& tokenFreq
) {
    std::ostringstream out;
    std::hash<std::string> hasher;

    out << "ARRAY[";

    bool first = true;
    for (const auto& [token, count] : tokenFreq) {
        if (!first) out << ",";
        first = false;

        int64_t hash = static_cast<int64_t>(hasher(token));
        int freq = std::min(count, 32767); // SMALLINT safety

        out << "ROW(" << hash << "," << freq << ")::token_stat";
    }

    out << "]";

    return out.str();
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