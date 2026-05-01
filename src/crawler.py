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
USER_AGENT = "COMP3011-SearchEngine/1.0"


class Crawler:
    """Crawls quotes.toscrape.com and extracts page content."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        delay: int = POLITENESS_DELAY,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.visited_urls: set[str] = set()
        self.pages: list[dict] = []

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a URL and return parsed BeautifulSoup object."""
        if url in self.visited_urls:
            logger.info("Already visited %s, skipping", url)
            return None

        logger.info("Fetching %s", url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return None

        self.visited_urls.add(url)
        return BeautifulSoup(response.text, "html.parser")

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
            "Crawling complete: %d pages, %d quotes total",
            len(self.pages),
            sum(len(p["quotes"]) for p in self.pages),
        )
        return self.pages

    def get_all_pages(self) -> list[dict]:
        """Return all crawled page data."""
        return self.pages
