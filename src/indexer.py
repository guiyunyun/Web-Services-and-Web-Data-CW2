"""Inverted index builder for the search engine."""

import re
import math
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "no", "nor",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "as", "until", "while",
    "if", "else", "also", "any", "every", "many", "much",
})


class Indexer:
    """Builds and manages an inverted index from crawled documents."""

    def __init__(self, use_stop_words: bool = True) -> None:
        self.index: dict[str, dict] = {}
        self.metadata: dict = {}
        self.documents: dict[str, str] = {}
        self.doc_token_counts: dict[str, int] = {}
        self.use_stop_words = use_stop_words

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercase alphanumeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def filter_stop_words(self, tokens: list[str]) -> list[str]:
        """Remove stop words from token list."""
        if not self.use_stop_words:
            return tokens
        return [t for t in tokens if t not in STOP_WORDS]

    def build(self, documents: list[dict]) -> dict:
        """Build inverted index from a list of documents.

        Each document should have 'url' and 'text' keys.
        """
        self.index = {}
        self.documents = {}
        self.doc_token_counts = {}

        for doc in documents:
            url = doc["url"]
            text = doc.get("text", "")
            self.documents[url] = text

            all_tokens = self.tokenize(text)
            tokens = self.filter_stop_words(all_tokens)
            self.doc_token_counts[url] = len(tokens)

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

        self._compute_tfidf(len(documents))

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

    def _compute_tfidf(self, total_docs: int) -> None:
        """Calculate TF-IDF scores for all terms in all documents."""
        for term, entry in self.index.items():
            df = entry["df"]
            idf = math.log(total_docs / df) if df > 0 else 0

            for url, posting in entry["postings"].items():
                doc_length = self.doc_token_counts.get(url, 1)
                tf = posting["tf"] / doc_length
                posting["tf_idf"] = round(tf * idf, 6)

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
