# Search Engine Tool

A command-line search engine for [quotes.toscrape.com](https://quotes.toscrape.com/). Crawls quote pages and author pages, builds an inverted index with TF-IDF scoring, and supports keyword search with ranked results.

Built for COMP3011 Web Services and Web Data, Coursework 2.

## Features

- **Web Crawler** - Follows pagination automatically, crawls author detail pages, respects a 6-second politeness window, retries with exponential backoff
- **Inverted Index** - Tokenization, stop word filtering, term frequency, document frequency, positional index, TF-IDF scoring
- **Search** - AND intersection queries, TF-IDF ranked results, context snippets with highlighted keywords, search timing
- **Query Suggestions** - Damerau-Levenshtein edit distance algorithm suggests similar words when no results are found
- **Dual Storage** - Save/load index in JSON (human-readable) or Pickle (compact binary)

## Setup

### Prerequisites

- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda
- Python 3.11
- Git

### Clone Repository

```bash
git clone https://github.com/guiyunyun/Web-Services-and-Web-Data-CW2.git
cd Web-Services-and-Web-Data-CW2
```

### Create Environment

```bash
conda create -n webcw2 python=3.11 -y
conda activate webcw2
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import requests; import bs4; import pytest; print('All dependencies OK!')"
```

## Usage

Run the interactive shell:

```bash
python -m src.main
```

### Available Commands

| Command | Description |
|---------|-------------|
| `build` | Crawl the website and build the inverted index |
| `build --format pickle` | Build and save index in Pickle format |
| `load` | Load index from file (default JSON) |
| `load --format pickle` | Load index from Pickle file |
| `print <word>` | Show inverted index details for a word |
| `find <word1> <word2> ...` | Find pages containing all words (AND query) |
| `help` | Show help message |
| `quit` | Exit the program |

### Example Session

```
$ python -m src.main

Welcome to Search Engine Tool
Type 'help' for available commands.

> build
Crawling https://quotes.toscrape.com/ ...
[Page 1] 10 quotes found. Waiting 6s...
[Page 2] 10 quotes found. Waiting 6s...
...
Building inverted index...
Index built successfully: 1500 terms across 60 pages.
Saved to data/index.json

> load
Index loaded: 1500 terms, 60 pages ready for search.

> find love
Found 5 results in 0.003s

  1. https://quotes.toscrape.com/page/1/ (TF-IDF: 0.85)
     "...that thing called LOVE changes your life..."

  2. https://quotes.toscrape.com/page/3/ (TF-IDF: 0.62)
     "...darkness cannot drive out hate, only LOVE can..."

> find lov
No results found for "lov".
Did you mean: love?

> print love
Term: "love"
Document Frequency: 5 (appears in 5 pages)

  1. https://quotes.toscrape.com/page/1/
     Term Frequency: 3
     Positions: [5, 12, 28]
     TF-IDF: 0.85

> quit
Goodbye!
```

## Project Structure

```
Web-Services-and-Web-Data-CW2/
├── src/
│   ├── crawler.py          # Web crawler with pagination and retry
│   ├── indexer.py           # Inverted index builder with TF-IDF
│   ├── search.py            # Search engine with ranking and suggestions
│   └── main.py              # Interactive CLI shell
├── tests/
│   ├── test_crawler.py      # Crawler unit tests (24 tests)
│   ├── test_indexer.py      # Indexer unit tests (33 tests)
│   ├── test_search.py       # Search unit tests (44 tests)
│   └── test_integration.py  # Integration tests (19 tests)
├── data/                    # Index storage directory
├── requirements.txt
└── README.md
```

## Architecture

```
User Input (CLI)
      │
      ▼
  main.py ─── Command Router
      │
      ├── build ──► crawler.py ──► indexer.py ──► data/index.json
      │              (crawl)        (index)        (save)
      │
      ├── load ───► indexer.py
      │              (load from file)
      │
      └── print/find ──► search.py
                          (query index)
```

## Technical Details

### Crawler

- Follows "Next" button pagination without hardcoding page numbers
- Crawls both quote pages and author detail pages
- Uses `requests.Session()` with custom User-Agent header
- 6-second delay between requests (politeness window)
- Retry with exponential backoff (up to 3 attempts)
- URL deduplication using a set

### Inverted Index

- Tokenization using regex (`[a-z0-9]+`)
- 80+ English stop words filtered via frozenset
- TF-IDF formula: `TF(t,d) = freq / doc_length`, `IDF(t) = log(N / df)`, `TF-IDF = TF * IDF`
- Stores term frequency, positions, and document frequency per term
- Supports JSON and Pickle serialization

### Search

- AND intersection query for multi-word searches
- Results ranked by combined TF-IDF score
- Context snippets with ANSI color-highlighted keywords
- Search timing display
- Query suggestions using Damerau-Levenshtein distance (supports transposition, max distance 2)

## Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

Run with coverage report:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Run individual test modules:

```bash
python -m pytest tests/test_crawler.py -v
python -m pytest tests/test_indexer.py -v
python -m pytest tests/test_search.py -v
python -m pytest tests/test_integration.py -v
```

### Test Summary

| Module | Tests | Code Coverage |
|--------|-------|---------------|
| Crawler | 24 | 70% |
| Indexer | 33 | 96% |
| Search | 44 | 100% |
| Integration | 19 | (cross-module) |
| **Total** | **120** | **86% overall** |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| requests | >= 2.31.0 | HTTP requests for web crawling |
| beautifulsoup4 | >= 4.12.0 | HTML parsing |
| pytest | >= 7.4.0 | Testing framework |
| pytest-cov | >= 4.1.0 | Test coverage reporting |

## GenAI Declaration

This project was developed with assistance from Claude (Anthropic). AI was used for code generation, debugging, and documentation. All code has been reviewed and understood by the author. A critical evaluation of AI usage is provided in the video demonstration.
