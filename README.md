# EngineDB - Semantic Search Engine

EngineDB is a high-performance semantic search engine that processes Wikipedia articles and enables fast similarity-based searching using neural embeddings and approximate nearest neighbor indexing.
A working demo is now available, with limited usage. Please allow the backend server a few minutes to warm up and turn online. You can view the project [here](https://wiki-search-prod-1.onrender.com/).
## Overview

EngineDB combines modern NLP techniques with efficient database technologies to create a searchable knowledge base:

- **Text Embedding**: Uses the all-MiniLM-L6-v2 model (384-dimensional embeddings) via ONNX Runtime for fast CPU inference
- **Vector Storage**: PostgreSQL with HNSW (Hierarchical Navigable Small World) indexing for efficient approximate nearest-neighbor search
- **Data Pipeline**: Automated parsing of Wikipedia XML dumps to JSON, with batch processing and asynchronous embedding

## Project Structure

```
EngineDB/
├── main.cpp                    # Entry point with interactive CLI
├── ArticleParser.cpp/h         # Parses JSON files and coordinates batch processing
├── VectorStorage.cpp/h         # Manages PostgreSQL storage and HNSW indexing
├── ONNXEmbedder.cpp/h          # Text embedding using ONNX models
├── WordPieceTokenizer.cpp/h    # Tokenization for embedding models
├── PageItem.h                  # Data structure for articles
├── Embedding.py                # Script to export ONNX models
├── models/                     # Pre-trained model files
│   ├── model.onnx              # All-MiniLM-L6-v2 in ONNX format
│   └── vocab.txt               # Tokenizer vocabulary
├── Data/                       # Data pipeline and processing
│   ├── WikipediaParse.py       # Wikipedia XML dump parser
│   ├── output/                 # Parsed JSON output (created by WikipediaParse.py)
│   └── README.md               # Data preparation instructions
├── hnswlib/                    # HNSW library for nearest-neighbor search
├── packages/                   # NuGet packages (ONNX Runtime)
├── vcpkg_installed/            # C++ dependencies (vcpkg)
└── venv310/                    # Python virtual environment
```

## Components

### Core C++ Components

**ArticleParser**
- Reads JSON files containing parsed Wikipedia articles
- Processes articles in configurable batch sizes (default: 250)
- Uses multi-threaded processing for efficient embedding
- Coordinates with VectorStorage to store embeddings

**VectorStorage**
- Manages all interaction with PostgreSQL database
- Handles HNSW index creation and management
- Provides semantic search functionality
- Stores article metadata (title, description, link)
- Supports configurable embedding dimensions (384-dim by default)
- Can handle up to 2 million vectors in HNSW index

**ONNXEmbedder**
- Loads and runs the all-MiniLM-L6-v2 model via ONNX Runtime
- Processes variable-length text inputs
- Handles tokenization, padding, and truncation
- Returns normalized 384-dimensional vectors
- Supports batch embedding for efficiency

**WordPieceTokenizer**
- Implements WordPiece tokenization algorithm
- Loads vocabulary from `vocab.txt`
- Handles special tokens (CLS, SEP, PAD, UNK)
- Supports configurable max sequence length (default: 256)

### Python Components

**WikipediaParse.py**
- Extracts articles from compressed Wikipedia XML dumps
- Outputs structured JSON files (up to 10,000 articles per file)
- Skips redirect pages automatically
- Runs in the `Data/` directory

**Embedding.py**
- Exports the all-MiniLM-L6-v2 model to ONNX format
- Exports tokenizer vocabulary in the format expected by WordPieceTokenizer
- Creates the model files needed by the C++ application (see Data/README.md)

## Prerequisites

### System Requirements
- Windows 10+ (or Linux/macOS with modifications)
- Visual Studio 2022 with C++ support (for building)
- PostgreSQL server running locally or accessible via network
- At least 4GB RAM for embedding processing
- ~10GB disk space for models and indexes

### Software Dependencies

**C++ Dependencies** (managed via vcpkg):
- ONNX Runtime
- libpqxx (PostgreSQL C++ client)
- cpp-httplib

**Python Dependencies**:
- Python 3.10+
- sentence-transformers
- transformers
- pqxx (PostgreSQL connector)

## Quick Start

### 1. Prepare Data

See [Data/README.md](Data/README.md) for complete data preparation instructions.

**Summary**:
```bash
cd Data
# Download Wikipedia dump to Data/wikiarticles.xml.bz2
# Then run:
python WikipediaParse.py
```

This generates JSON files in `Data/output/` containing parsed articles.

### 2. Build and Run

**Build C++ Project**:
```bash
# In Visual Studio
# Build → Build Solution (or Ctrl+Shift+B)
```

**Configure and Run**:

Edit `main.cpp` to match your setup:
```cpp
pqxx::connection conn("host=localhost port=5432 dbname=vectorstore user=postgres password=YOUR_PASSWORD");

// Adjust these parameters as needed:
size_t batchSize = 250;        // Articles processed per batch
size_t maxThreads = 8;         // Concurrent workers
int maxPages = 5000;           // Total articles to process (-1 for all)
```

**Run Application**:
```bash
cd EngineDB\x64\Release
EngineDB.exe
```

### 4. Usage

The application presents an interactive menu:

```
Select an option:
1. Parse JSON files and store vectors
2. Search
3. Exit
```

**Option 1 - Parse and Store**:
- Reads JSON files from `Data/output/`
- Embeds each article using the ONNX model
- Stores embeddings and metadata in PostgreSQL
- Creates/updates HNSW index for fast search
- Shows progress and timing information

**Option 2 - Search**:
```
Search query (or 'exit'): neural networks in deep learning
```
- Enter search text (natural language)
- System finds most similar articles using semantic similarity
- Returns top results with scores

## Configuration

### ArticleParser Configuration (main.cpp)
- `parsedJSONpath`: Path to JSON files from WikipediaParse.py (default: `./Data/output`)
- `batchSize`: Articles per processing batch (default: 250, higher = faster but more memory)
- `maxThreads`: Concurrent embedding workers (default: 8, adjust based on CPU cores)
- `maxPages`: Limit total articles processed, -1 for all (default: 5000)

### VectorStorage Configuration (main.cpp)
- `DIM`: Embedding dimension (default: 384, matches all-MiniLM-L6-v2 output)
- `MAX_ELEMENTS`: Maximum HNSW index capacity (default: 2,000,000)

### Database Connection (main.cpp)
```cpp
pqxx::connection conn("host=localhost port=5432 dbname=vectorstore user=postgres password=YOUR_PASSWORD");
```

## How It Works

### Data Pipeline

1. **Wikipedia XML Dump** → WikipediaParse.py → **JSON Files** (10K articles/file)
2. **JSON Files** → ArticleParser → **Embedding Queue**
3. **Embedding Queue** → ONNXEmbedder → **384-dim Vectors**
4. **Vectors** → VectorStorage → **PostgreSQL + HNSW Index**

### Search Process

1. User enters search query
2. Query text is embedded using the same ONNX model
3. HNSW index performs approximate nearest neighbor search
4. Top K most similar articles are retrieved from PostgreSQL
5. Results displayed with similarity scores

## Performance Characteristics

- **Embedding**: ~100-200 articles/second (batch processing)
- **Search Latency**: <100ms for exact NN search against 2M vectors
- **Memory**: ~4GB for embeddings + 2M articles (~1.5GB for HNSW index + 2.5GB for PostgreSQL)
- **Model Size**: 91MB (all-MiniLM-L6-v2 ONNX format)

## Troubleshooting

### Database Connection Failures
- Verify PostgreSQL is running: `psql -U postgres`
- Check credentials in connection string
- Ensure database and table exist
- Test connectivity: `psql -h localhost -U postgres -d vectorstore`

### Embedding Model Not Found
- Verify `models/model.onnx` and `models/vocab.txt` exist
- Run Embedding.py to regenerate model files (see Data/README.md)
- Check relative paths in ONNXEmbedder constructor

### Memory Issues During Processing
- Reduce `batchSize` in main.cpp
- Process in multiple runs with `maxPages` limit
- Reduce `maxThreads` to lower peak memory usage

### Slow Embedding Speed
- Ensure batch processing is enabled
- Check CPU isn't bottlenecked by disk I/O (SSD recommended)
- Verify ONNX Runtime isn't set to GPU mode
