"""Search engine with print and find commands."""

import logging
from typing import Optional

from src.indexer import Indexer

logger = logging.getLogger(__name__)


class SearchEngine:
    """Provides search operations over an inverted index."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def is_ready(self) -> bool:
        """Check whether the index has been built or loaded."""
        return len(self.indexer.index) > 0

    def print_term(self, word: str) -> str:
        """Return formatted inverted index details for a single term."""
        if not self.is_ready():
            return "Index not loaded. Please run 'build' or 'load' first."

        term = word.lower().strip()
        if not term:
            return "Please provide a word to look up."

        entry = self.indexer.get_term(term)
        if entry is None:
            return f'Term "{term}" not found in the index.'

        lines = []
        lines.append(f'Term: "{term}"')
        lines.append(
            f"Document Frequency: {entry['df']} "
            f"(appears in {entry['df']} pages)"
        )
        lines.append("")

        sorted_postings = sorted(
            entry["postings"].items(),
            key=lambda item: item[1]["tf_idf"],
            reverse=True,
        )

        for rank, (url, posting) in enumerate(sorted_postings, 1):
            lines.append(f"  {rank}. {url}")
            lines.append(f"     Term Frequency: {posting['tf']}")
            lines.append(f"     Positions: {posting['positions']}")
            lines.append(f"     TF-IDF: {posting['tf_idf']}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def find(self, words: list[str]) -> str:
        """AND query across all words, ranked by combined TF-IDF."""
        if not self.is_ready():
            return "Index not loaded. Please run 'build' or 'load' first."

        terms = [w.lower().strip() for w in words if w.strip()]
        if not terms:
            return "Please provide at least one search term."

        posting_sets: list[Optional[set[str]]] = []
        term_entries: dict[str, dict] = {}

        for term in terms:
            entry = self.indexer.get_term(term)
            if entry is None:
                query_str = " ".join(terms)
                return f'No results found for "{query_str}".'
            posting_sets.append(set(entry["postings"].keys()))
            term_entries[term] = entry

        matched_urls = posting_sets[0]
        for s in posting_sets[1:]:
            matched_urls = matched_urls & s

        if not matched_urls:
            query_str = " ".join(terms)
            return f'No results found for "{query_str}".'

        scored_results: list[tuple[str, float]] = []
        for url in matched_urls:
            total_score = sum(
                term_entries[t]["postings"][url]["tf_idf"] for t in terms
            )
            scored_results.append((url, round(total_score, 6)))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        lines = []
        lines.append(f"Found {len(scored_results)} results")
        lines.append("")

        for rank, (url, score) in enumerate(scored_results, 1):
            lines.append(f"  {rank}. {url} (TF-IDF: {score})")

        return "\n".join(lines)
