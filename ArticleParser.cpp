#include "ArticleParser.h"
#include "PageItem.h"
#include "VectorStorage.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

namespace fs = std::filesystem;     // for directory iteration
using json = nlohmann::json;        // for JSON parsing

// Constructor
ArticleParser::ArticleParser(
    const std::string& jsonPath,
    size_t batchSize,
    VectorStorage& storage,
    int maxPages)
	: jsonPath(jsonPath), batchSize(batchSize), storage(storage), maxPages(maxPages) {  
}

// Parse JSON files in the specified directory
void ArticleParser::parseJSONFiles() {
    std::vector<PageItem> batch;
	int pageCount = 0;

	// Iterate over all JSON files in the directory
    for (const auto& entry : fs::directory_iterator(jsonPath)) {
        if (entry.path().extension() != ".json") continue;

        std::ifstream file(entry.path());
        if (!file) continue;

		// Read each line (article) in the JSON file
        std::string line;
        while (std::getline(file, line)) {
			++pageCount;
            auto j = json::parse(line, nullptr, false);
            if (j.is_discarded()) continue;

			std::cout << "Processing article: " << j.value("title", "") << std::endl;

            std::string text = j.value("text", "");
            if (text.find("#REDIRECT") != std::string::npos) continue;

			// Add article to batch
            batch.push_back({
                j.value("title", ""),
                text,
                "https://en.wikipedia.org/wiki/" + j.value("title", "")
            });

            if (batch.size() >= batchSize) {
				std::cout << "Flushing batch of size: " << batch.size() << std::endl;
                flushBatch(batch);
				batch.clear();
            }

			// Check if max pages limit is reached
            if (maxPages != -1 && pageCount >= maxPages) {
                std::cout << "MAX PAGES REACHED: " << maxPages << std::endl;
                flushBatch(batch);
                return;
            }
        } 
		file.close();
    }
}

// Flush the current batch to storage
void ArticleParser::flushBatch(std::vector<PageItem>& batch) {
    if (batch.empty()) return;
    
	storage.ingestBatch(batch);
}
