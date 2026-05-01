"""Unit tests for the crawler module."""

from unittest.mock import patch, MagicMock
import requests
from bs4 import BeautifulSoup
from src.crawler import Crawler

SAMPLE_QUOTE_PAGE = """
<html>
<body>
  <div class="quote">
    <span class="text">"Life is beautiful."</span>
    <small class="author">Someone Famous</small>
    <a href="/author/Someone-Famous">(about)</a>
    <div class="tags">
      <a class="tag" href="/tag/life/page/1/">life</a>
      <a class="tag" href="/tag/beauty/page/1/">beauty</a>
    </div>
  </div>
  <div class="quote">
    <span class="text">"The world is round."</span>
    <small class="author">Another Author</small>
    <a href="/author/Another-Author">(about)</a>
    <div class="tags">
      <a class="tag" href="/tag/world/page/1/">world</a>
    </div>
  </div>
  <nav>
    <ul class="pager">
      <li class="next"><a href="/page/2/">Next</a></li>
    </ul>
  </nav>
</body>
</html>
"""

SAMPLE_LAST_PAGE = """
<html>
<body>
  <div class="quote">
    <span class="text">"Last quote."</span>
    <small class="author">Final Author</small>
    <a href="/author/Final-Author">(about)</a>
    <div class="tags">
      <a class="tag" href="/tag/end/page/1/">end</a>
    </div>
  </div>
  <nav>
    <ul class="pager">
      <li class="previous"><a href="/page/9/">Previous</a></li>
    </ul>
  </nav>
</body>
</html>
"""

SAMPLE_AUTHOR_PAGE = """
<html>
<body>
  <h3 class="author-title">Albert Einstein</h3>
  <span class="author-born-date">March 14, 1879</span>
  <span class="author-born-location">in Ulm, Germany</span>
  <div class="author-description">
    A famous physicist who developed the theory of relativity.
  </div>
</body>
</html>
"""

SAMPLE_EMPTY_PAGE = """
<html><body></body></html>
"""

SAMPLE_MISSING_FIELDS = """
<html>
<body>
  <div class="quote">
    <span class="text">"Only text here."</span>
  </div>
  <div class="quote">
    <small class="author">Only author here</small>
  </div>
</body>
</html>
"""


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestParseQuotes:
    """Tests for _parse_quotes method."""

    def test_extracts_all_quotes(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert len(quotes) == 2

    def test_extracts_quote_text(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert "Life is beautiful." in quotes[0]["text"]

    def test_extracts_author_name(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert quotes[0]["author"] == "Someone Famous"

    def test_extracts_author_url(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert quotes[0]["author_url"] == "/author/Someone-Famous"

    def test_extracts_tags(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert quotes[0]["tags"] == ["life", "beauty"]
        assert quotes[1]["tags"] == ["world"]

    def test_empty_page_returns_empty_list(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_EMPTY_PAGE)
        quotes = crawler._parse_quotes(soup)
        assert quotes == []

    def test_skips_quotes_missing_text_or_author(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_MISSING_FIELDS)
        quotes = crawler._parse_quotes(soup)
        assert len(quotes) == 0


class TestGetNextPageUrl:
    """Tests for _get_next_page_url method."""

    def test_finds_next_page(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        url = crawler._get_next_page_url(soup)
        assert url == "https://quotes.toscrape.com/page/2/"

    def test_returns_none_on_last_page(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_LAST_PAGE)
        url = crawler._get_next_page_url(soup)
        assert url is None

    def test_returns_none_on_empty_page(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_EMPTY_PAGE)
        url = crawler._get_next_page_url(soup)
        assert url is None


class TestGetPageText:
    """Tests for _get_page_text method."""

    def test_extracts_text_from_quotes(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_QUOTE_PAGE)
        text = crawler._get_page_text(soup)
        assert "Life is beautiful" in text
        assert "Someone Famous" in text
        assert "The world is round" in text

    def test_empty_page_returns_empty_string(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_EMPTY_PAGE)
        text = crawler._get_page_text(soup)
        assert text == ""


class TestParseAuthorPage:
    """Tests for _parse_author_page method."""

    def test_extracts_author_name(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_AUTHOR_PAGE)
        author = crawler._parse_author_page(soup)
        assert author["name"] == "Albert Einstein"

    def test_extracts_born_date(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_AUTHOR_PAGE)
        author = crawler._parse_author_page(soup)
        assert author["born_date"] == "March 14, 1879"

    def test_extracts_born_location(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_AUTHOR_PAGE)
        author = crawler._parse_author_page(soup)
        assert author["born_location"] == "in Ulm, Germany"

    def test_extracts_bio(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_AUTHOR_PAGE)
        author = crawler._parse_author_page(soup)
        assert "theory of relativity" in author["bio"]

    def test_handles_missing_fields(self):
        crawler = Crawler()
        soup = make_soup(SAMPLE_EMPTY_PAGE)
        author = crawler._parse_author_page(soup)
        assert author["name"] == ""
        assert author["born_date"] == ""
        assert author["bio"] == ""


class TestFetch:
    """Tests for _fetch method with mocked HTTP requests."""

    @patch("src.crawler.requests.Session")
    def test_successful_fetch(self, mock_session_cls):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_QUOTE_PAGE
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        crawler = Crawler()
        crawler.session = mock_session
        result = crawler._fetch("https://quotes.toscrape.com/")

        assert result is not None
        mock_session.get.assert_called_once()

    @patch("src.crawler.requests.Session")
    def test_skips_already_visited_url(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        crawler = Crawler()
        crawler.session = mock_session
        crawler.visited_urls.add("https://quotes.toscrape.com/")

        result = crawler._fetch("https://quotes.toscrape.com/")
        assert result is None
        mock_session.get.assert_not_called()

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_retries_on_failure(self, mock_session_cls, mock_sleep):
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            requests.exceptions.ConnectionError("Connection error"),
            requests.exceptions.ConnectionError("Connection error"),
            MagicMock(text=SAMPLE_QUOTE_PAGE, raise_for_status=MagicMock()),
        ]
        mock_session_cls.return_value = mock_session

        crawler = Crawler(delay=1)
        crawler.session = mock_session
        result = crawler._fetch("https://quotes.toscrape.com/")

        assert result is not None
        assert mock_session.get.call_count == 3

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.Session")
    def test_returns_none_after_all_retries_fail(self, mock_session_cls, mock_sleep):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection error")
        mock_session_cls.return_value = mock_session

        crawler = Crawler(delay=1)
        crawler.session = mock_session
        result = crawler._fetch("https://example.com/fail")

        assert result is None
        assert mock_session.get.call_count == 3


class TestCollectAuthorUrls:
    """Tests for _collect_author_urls method."""

    def test_collects_unique_urls(self):
        crawler = Crawler()
        crawler.pages = [
            {
                "url": "http://example.com/page/1/",
                "quotes": [
                    {"author_url": "/author/A"},
                    {"author_url": "/author/B"},
                    {"author_url": "/author/A"},
                ],
            },
            {
                "url": "http://example.com/page/2/",
                "quotes": [
                    {"author_url": "/author/B"},
                    {"author_url": "/author/C"},
                ],
            },
        ]
        urls = crawler._collect_author_urls()
        assert urls == ["/author/A", "/author/B", "/author/C"]

    def test_empty_pages_returns_empty(self):
        crawler = Crawler()
        crawler.pages = []
        urls = crawler._collect_author_urls()
        assert urls == []


class TestGetAllDocuments:
    """Tests for get_all_documents method."""

    def test_returns_combined_documents(self):
        crawler = Crawler()
        crawler.pages = [
            {"url": "http://example.com/page/1/", "text": "some text", "quotes": []},
        ]
        crawler.authors = {
            "/author/A": {
                "url": "http://example.com/author/A",
                "text": "author bio",
            },
        }

        docs = crawler.get_all_documents()
        assert len(docs) == 2
        assert docs[0]["type"] == "quotes_page"
        assert docs[1]["type"] == "author_page"
