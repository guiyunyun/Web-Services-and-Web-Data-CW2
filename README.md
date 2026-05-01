# Search Engine Tool

A search engine tool for COMP3011 Web Services and Web Data (Coursework 2). Crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index with TF-IDF scoring, and supports keyword search.

## Setup

### Prerequisites

- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda
- Git

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

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| requests | >= 2.31.0 | HTTP requests for web crawling |
| beautifulsoup4 | >= 4.12.0 | HTML parsing |
| pytest | >= 7.4.0 | Testing framework |
| pytest-cov | >= 4.1.0 | Test coverage reporting |

## Usage

*Coming soon.*

## Testing

*Coming soon.*
