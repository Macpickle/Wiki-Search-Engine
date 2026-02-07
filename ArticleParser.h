#pragma once
#include "PageItem.h"
#include "VectorStorage.h"

#include <string>
#include <vector>
#include <future>
#include <list>

/*
This class is responsible for parsing JSON files containing articles
Relies on WikipediaSearch.py to generate JSON files from Wikipedia dumps
*/

// Parses JSON files containing articles and stores in vector storage
class ArticleParser {
private:
	std::list<std::future<void>> activeTasks;
	void flushBatch(std::vector<PageItem>& batch);	// Flush a batch of PageItems to vector storage

	std::string jsonPath;       // relative path to JSON files
	size_t batchSize;           // batch size for processing, will input into DB after n articles
	VectorStorage& storage;     // reference to vector storage
	int maxPages;               // maximum number of pages to parse (-1 for no limit)

public:
    ArticleParser(
		const std::string& jsonPath,
        size_t batchSize,
        VectorStorage& storage,
		int maxPages
    );

	void parseJSONFiles();      // Parse JSON files
};
