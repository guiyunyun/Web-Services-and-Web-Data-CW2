"""Web crawler for quotes.toscrape.com"""

import time
import logging
from urllib.parse import urljoin
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://quotes.toscrape.com/"
POLITENESS_DELAY = 6
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
USER_AGENT = "COMP3011-SearchEngine/1.0"


class Crawler:
    """Crawls quotes.toscrape.com and extracts page content."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        delay: int = POLITENESS_DELAY,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.base_url = base_url
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.visited_urls: set[str] = set()
        self.pages: list[dict] = []
        self.authors: dict[str, dict] = {}

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a URL with retry and exponential backoff."""
        if url in self.visited_urls:
            logger.info("Already visited %s, skipping", url)
            return None

        for attempt in range(1, self.max_retries + 1):
            logger.info("Fetching %s (attempt %d/%d)", url, attempt, self.max_retries)
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                self.visited_urls.add(url)
                return BeautifulSoup(response.text, "html.parser")
            except requests.exceptions.RequestException as e:
                logger.warning("Attempt %d failed for %s: %s", attempt, url, e)
                if attempt < self.max_retries:
                    backoff = self.delay * (2 ** (attempt - 1))
                    logger.info("Retrying in %ds...", backoff)
                    time.sleep(backoff)

        logger.error("All %d attempts failed for %s", self.max_retries, url)
        return None

    def _parse_quotes(self, soup: BeautifulSoup) -> list[dict]:
        """Extract all quotes from a page."""
        quotes = []
        for quote_div in soup.find_all("div", class_="quote"):
            text_tag = quote_div.find("span", class_="text")
            author_tag = quote_div.find("small", class_="author")
            author_link = quote_div.find("a", href=True)
            tag_links = quote_div.find_all("a", class_="tag")

            if not text_tag or not author_tag:
                continue

            quote = {
                "text": text_tag.get_text(strip=True),
                "author": author_tag.get_text(strip=True),
                "author_url": author_link["href"] if author_link else None,
                "tags": [tag.get_text(strip=True) for tag in tag_links],
            }
            quotes.append(quote)

        return quotes

    def _get_next_page_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the next page URL from pagination."""
        next_li = soup.find("li", class_="next")
        if next_li:
            next_link = next_li.find("a", href=True)
            if next_link:
                return urljoin(self.base_url, next_link["href"])
        return None

    def _get_page_text(self, soup: BeautifulSoup) -> str:
        """Extract all visible text from quote elements on the page."""
        texts = []
        for quote_div in soup.find_all("div", class_="quote"):
            texts.append(quote_div.get_text(separator=" ", strip=True))
        return " ".join(texts)

    def _parse_author_page(self, soup: BeautifulSoup) -> dict:
        """Extract author details from an author page."""
        name_tag = soup.find("h3", class_="author-title")
        born_date_tag = soup.find("span", class_="author-born-date")
        born_loc_tag = soup.find("span", class_="author-born-location")
        bio_tag = soup.find("div", class_="author-description")

        return {
            "name": name_tag.get_text(strip=True) if name_tag else "",
            "born_date": born_date_tag.get_text(strip=True) if born_date_tag else "",
            "born_location": born_loc_tag.get_text(strip=True) if born_loc_tag else "",
            "bio": bio_tag.get_text(strip=True) if bio_tag else "",
        }

    def _collect_author_urls(self) -> list[str]:
        """Collect unique author page URLs from all crawled quotes."""
        seen = set()
        urls = []
        for page in self.pages:
            for quote in page["quotes"]:
                author_url = quote.get("author_url")
                if author_url and author_url not in seen:
                    seen.add(author_url)
                    urls.append(author_url)
        return urls

    def crawl_quotes(self) -> list[dict]:
        """Crawl all quote pages following Next button pagination."""
        url = self.base_url
        page_num = 0

        while url:
            soup = self._fetch(url)
            if not soup:
                break

            page_num += 1
            quotes = self._parse_quotes(soup)
            page_text = self._get_page_text(soup)

            page_data = {
                "url": url,
                "page_num": page_num,
                "text": page_text,
                "quotes": quotes,
            }
            self.pages.append(page_data)
            logger.info("[Page %d] %d quotes found", page_num, len(quotes))

            url = self._get_next_page_url(soup)
            if url:
                logger.info("Waiting %ds before next request...", self.delay)
                time.sleep(self.delay)

        logger.info(
            "Quote crawling complete: %d pages, %d quotes total",
            len(self.pages),
            sum(len(p["quotes"]) for p in self.pages),
        )
        return self.pages

    def crawl_authors(self) -> dict[str, dict]:
        """Crawl all unique author detail pages."""
        author_urls = self._collect_author_urls()
        total = len(author_urls)
        logger.info("Found %d unique authors to crawl", total)

        for i, author_path in enumerate(author_urls, 1):
            full_url = urljoin(self.base_url, author_path)

            if i > 1:
                logger.info("Waiting %ds before next request...", self.delay)
                time.sleep(self.delay)

            soup = self._fetch(full_url)
            if not soup:
                continue

            author_data = self._parse_author_page(soup)
            author_data["url"] = full_url
            author_data["text"] = (
                f"{author_data['name']} {author_data['born_date']} "
                f"{author_data['born_location']} {author_data['bio']}"
            )
            self.authors[author_path] = author_data
            logger.info("[Author %d/%d] %s", i, total, author_data["name"])

        logger.info("Author crawling complete: %d authors", len(self.authors))
        return self.authors

    def crawl(self) -> dict:
        """Run full crawl of quotes and author pages."""
        self.crawl_quotes()
        self.crawl_authors()

        return {
            "pages": self.pages,
            "authors": self.authors,
        }

    def get_all_documents(self) -> list[dict]:
        """Return all crawled data as a flat list of documents for indexing."""
        documents = []

        for page in self.pages:
            documents.append({
                "url": page["url"],
                "text": page["text"],
                "type": "quotes_page",
            })

        for author_path, author_data in self.authors.items():
            documents.append({
                "url": author_data["url"],
                "text": author_data["text"],
                "type": "author_page",
            })

        return documents
