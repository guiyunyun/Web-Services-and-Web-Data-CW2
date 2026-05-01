"""Inverted index builder for the search engine."""

import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class Indexer:
    """Builds and manages an inverted index from crawled documents."""

    def __init__(self) -> None:
        self.index: dict[str, dict] = {}
        self.metadata: dict = {}
        self.documents: dict[str, str] = {}

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercase alphanumeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def build(self, documents: list[dict]) -> dict:
        """Build inverted index from a list of documents.

        Each document should have 'url' and 'text' keys.
        """
        self.index = {}
        self.documents = {}

        for doc in documents:
            url = doc["url"]
            text = doc.get("text", "")
            self.documents[url] = text
            tokens = self.tokenize(text)

            term_positions: dict[str, list[int]] = {}
            for position, token in enumerate(tokens):
                if token not in term_positions:
                    term_positions[token] = []
                term_positions[token].append(position)

            for term, positions in term_positions.items():
                if term not in self.index:
                    self.index[term] = {"df": 0, "postings": {}}
                self.index[term]["df"] += 1
                self.index[term]["postings"][url] = {
                    "tf": len(positions),
                    "positions": positions,
                }

        self.metadata = {
            "total_documents": len(documents),
            "total_terms": len(self.index),
            "built_at": datetime.now().isoformat(),
            "document_urls": [doc["url"] for doc in documents],
        }

        logger.info(
            "Index built: %d terms across %d documents",
            len(self.index),
            len(documents),
        )
        return self.get_full_index()

    def get_full_index(self) -> dict:
        """Return the complete index with metadata."""
        return {
            "metadata": self.metadata,
            "index": self.index,
            "documents": self.documents,
        }

    def get_term(self, term: str) -> Optional[dict]:
        """Look up a single term in the index."""
        return self.index.get(term.lower())

    def get_document_text(self, url: str) -> str:
        """Return the stored text for a document URL."""
        return self.documents.get(url, "")
