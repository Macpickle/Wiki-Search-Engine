#include "ArticleParser.h"
#include "VectorStorage.h"

#include <iostream>
#include <string>
#include <exception>
#include <pqxx/connection.hxx>

int main() {
	pqxx::connection conn("host=localhost port=5432 dbname=VectorStore user=postgres password=??????");
	char userInput;										// user input for options

	// options for parsing
	std::string parsedJSONpath = "./Data/output";		// path to where JSON files are stored
	size_t batchSize = 250;								// batch value for parsing to embedding server
	size_t maxThreads = 8;	
	int maxPages = 500;								// maximum number of pages to parse (-1 for no limit)

	VectorStorage storage(conn, maxThreads);		// Initialize vector storage

	ArticleParser parser(parsedJSONpath, batchSize, storage, maxPages);	// Initialize article parser, used for option 1

	// Get user input for options (search, parse, exit)
	while (true) {
		std::cout << "Select an option:\n";
		std::cout << "1. Parse JSON files and store vectors\n";
		std::cout << "2. Search\n";
		std::cout << "3. Exit\n";
		std::cout << "Enter choice (1-3): ";
		std::cin >> userInput;

		// Parse JSON files and store vectors
		if (userInput == '1') {
			try {
				parser.parseJSONFiles();
			}
			catch (const std::exception& e) {
				std::cerr << "Error during parsing and storing vectors: " << e.what() << std::endl;
			}
		}

		// Search interface
		else if (userInput == '2') {
			std::cin.ignore(); // clear leftover newline
			std::string query;

			while (true) {
				std::cout << "\nSearch query (or 'exit'): ";
				std::getline(std::cin, query);

				if (query == "exit" || query.empty())
					break;

				auto results = storage.search(query, 10);

				if (results.empty()) {
					std::cout << "No results found.\n";
					continue;
				}

				for (auto& r : results) {
					std::cout << "Title: " << r.title << "\n";
					std::cout << "Link: " << r.link << "\n";
					std::cout << "Score: " << r.score << "\n";
				}
			}
		}

		// Exit program
		else if (userInput == '3') {
			break;
		}

		else {
			std::cout << "Invalid choice. Please try again.\n";
		}
	}
	return 0;
}
