"""Integration tests for the full search engine pipeline."""

import os
import tempfile
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup
from src.crawler import Crawler
from src.indexer import Indexer
from src.search import SearchEngine
from src.main import main, parse_format_flag

MOCK_PAGE_1 = """
<html><body>
  <div class="quote">
    <span class="text">“The world as we have created it is a process of our thinking.”</span>
    <small class="author">Albert Einstein</small>
    <a href="/author/Albert-Einstein">(about)</a>
    <div class="tags">
      <a class="tag">change</a>
      <a class="tag">thinking</a>
    </div>
  </div>
  <div class="quote">
    <span class="text">“Love all, trust a few, do wrong to none.”</span>
    <small class="author">William Shakespeare</small>
    <a href="/author/William-Shakespeare">(about)</a>
    <div class="tags">
      <a class="tag">love</a>
    </div>
  </div>
  <nav><ul class="pager">
    <li class="next"><a href="/page/2/">Next</a></li>
  </ul></nav>
</body></html>
"""

MOCK_PAGE_2 = """
<html><body>
  <div class="quote">
    <span class="text">“Life is what happens when you're busy making other plans.”</span>
    <small class="author">John Lennon</small>
    <a href="/author/John-Lennon">(about)</a>
    <div class="tags">
      <a class="tag">life</a>
    </div>
  </div>
  <nav><ul class="pager">
    <li class="previous"><a href="/page/1/">Previous</a></li>
  </ul></nav>
</body></html>
"""

MOCK_AUTHOR = """
<html><body>
  <h3 class="author-title">Albert Einstein</h3>
  <span class="author-born-date">March 14, 1879</span>
  <span class="author-born-location">in Ulm, Germany</span>
  <div class="author-description">A theoretical physicist.</div>
</body></html>
"""


def simulate_crawl():
    """Build crawler data without network requests."""
    crawler = Crawler(delay=0)

    for url, html in [
        ("https://quotes.toscrape.com/", MOCK_PAGE_1),
        ("https://quotes.toscrape.com/page/2/", MOCK_PAGE_2),
    ]:
        soup = BeautifulSoup(html, "html.parser")
        quotes = crawler._parse_quotes(soup)
        page_text = crawler._get_page_text(soup)
        crawler.pages.append({
            "url": url,
            "page_num": len(crawler.pages) + 1,
            "text": page_text,
            "quotes": quotes,
        })

    for path in ["/author/Albert-Einstein", "/author/William-Shakespeare",
                 "/author/John-Lennon"]:
        soup = BeautifulSoup(MOCK_AUTHOR, "html.parser")
        author_data = crawler._parse_author_page(soup)
        full_url = f"https://quotes.toscrape.com{path}"
        author_data["url"] = full_url
        author_data["text"] = (
            f"{author_data['name']} {author_data['born_date']} "
            f"{author_data['born_location']} {author_data['bio']}"
        )
        crawler.authors[path] = author_data

    return crawler


class TestCrawlerToIndexPipeline:
    """Test crawler output feeds correctly into the indexer."""

    def test_crawl_and_build_index(self):
        crawler = simulate_crawl()
        documents = crawler.get_all_documents()

        assert len(documents) == 5

        indexer = Indexer()
        indexer.build(documents)

        assert indexer.metadata["total_documents"] == 5
        assert indexer.metadata["total_terms"] > 0

    def test_crawl_build_and_search(self):
        crawler = simulate_crawl()
        documents = crawler.get_all_documents()

        indexer = Indexer()
        indexer.build(documents)
        engine = SearchEngine(indexer)

        result = engine.find(["world"])
        assert "Found" in result

        result = engine.print_term("world")
        assert "Term:" in result
        assert "Document Frequency:" in result

    def test_search_not_found_gives_suggestion(self):
        crawler = simulate_crawl()
        documents = crawler.get_all_documents()

        indexer = Indexer()
        indexer.build(documents)
        engine = SearchEngine(indexer)

        result = engine.find(["wrold"])
        assert "No results found" in result
        assert "Did you mean:" in result


class TestSaveLoadSearch:
    """Test index persistence and search after reload."""

    def test_json_save_load_search(self):
        indexer = Indexer()
        docs = [
            {"url": "http://example.com/1", "text": "love makes life beautiful"},
            {"url": "http://example.com/2", "text": "life is wonderful"},
        ]
        indexer.build(docs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            indexer.save(path, fmt="json")

            indexer2 = Indexer()
            indexer2.load(path)
            engine = SearchEngine(indexer2)

            result = engine.find(["love"])
            assert "Found" in result
            assert "http://example.com/1" in result
        finally:
            os.remove(path)

    def test_pickle_save_load_search(self):
        indexer = Indexer()
        docs = [
            {"url": "http://example.com/1", "text": "love makes life beautiful"},
            {"url": "http://example.com/2", "text": "life is wonderful"},
        ]
        indexer.build(docs)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            indexer.save(path, fmt="pickle")

            indexer2 = Indexer()
            indexer2.load(path)
            engine = SearchEngine(indexer2)

            result = engine.find(["love"])
            assert "Found" in result
        finally:
            os.remove(path)

    def test_search_results_match_before_and_after_reload(self):
        indexer = Indexer()
        docs = [
            {"url": "http://example.com/1", "text": "cat dog bird"},
            {"url": "http://example.com/2", "text": "cat fish"},
        ]
        indexer.build(docs)
        engine1 = SearchEngine(indexer)
        result_before = engine1.print_term("cat")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            indexer.save(path, fmt="json")
            indexer2 = Indexer()
            indexer2.load(path)
            engine2 = SearchEngine(indexer2)
            result_after = engine2.print_term("cat")

            assert result_before == result_after
        finally:
            os.remove(path)


class TestParseFormatFlag:
    """Test CLI format flag parsing."""

    def test_default_json(self):
        assert parse_format_flag(["build"]) == "json"

    def test_pickle_format(self):
        assert parse_format_flag(["build", "--format", "pickle"]) == "pickle"

    def test_json_format(self):
        assert parse_format_flag(["load", "--format", "json"]) == "json"

    def test_missing_value(self):
        assert parse_format_flag(["build", "--format"]) == "json"


class TestCLICommands:
    """Test CLI command routing with mocked input."""

    @patch("builtins.input", side_effect=["help", "quit"])
    @patch("builtins.print")
    def test_help_command(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Available commands" in output

    @patch("builtins.input", side_effect=["exit"])
    @patch("builtins.print")
    def test_exit_command(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Goodbye" in output

    @patch("builtins.input", side_effect=["find love", "quit"])
    @patch("builtins.print")
    def test_find_without_index(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Index not loaded" in output

    @patch("builtins.input", side_effect=["print love", "quit"])
    @patch("builtins.print")
    def test_print_without_index(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Index not loaded" in output

    @patch("builtins.input", side_effect=["load", "quit"])
    @patch("builtins.print")
    def test_load_no_file(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "No index file found" in output or "Index loaded" in output

    @patch("builtins.input", side_effect=["unknown_cmd", "quit"])
    @patch("builtins.print")
    def test_unknown_command(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Unknown command" in output

    @patch("builtins.input", side_effect=["", "quit"])
    @patch("builtins.print")
    def test_empty_input(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Goodbye" in output

    @patch("builtins.input", side_effect=["find", "quit"])
    @patch("builtins.print")
    def test_find_no_args(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Usage" in output

    @patch("builtins.input", side_effect=["print", "quit"])
    @patch("builtins.print")
    def test_print_no_args(self, mock_print, mock_input):
        main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Usage" in output
