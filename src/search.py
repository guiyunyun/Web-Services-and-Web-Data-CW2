"""Search engine with print and find commands."""

import re
import time
import logging

from src.indexer import Indexer

RESET = "\033[0m"
HIGHLIGHT = "\033[1;33m"

logger = logging.getLogger(__name__)


class SearchEngine:
    """Provides search operations over an inverted index."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def is_ready(self) -> bool:
        """Check whether the index has been built or loaded."""
        return len(self.indexer.index) > 0

    def _highlight(self, text: str, terms: list[str]) -> str:
        """Highlight search terms in text with ANSI bold yellow."""
        for term in terms:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            text = pattern.sub(
                lambda m: f"{HIGHLIGHT}{m.group().upper()}{RESET}", text
            )
        return text

    def _snippet(self, url: str, terms: list[str], context_chars: int = 40) -> str:
        """Extract a text snippet around the first matched term."""
        doc_text = self.indexer.get_document_text(url)
        if not doc_text:
            return ""

        best_pos = len(doc_text)
        for term in terms:
            match = re.search(r"\b" + re.escape(term) + r"\b", doc_text, re.IGNORECASE)
            if match and match.start() < best_pos:
                best_pos = match.start()

        if best_pos == len(doc_text):
            return ""

        start = max(0, best_pos - context_chars)
        end = min(len(doc_text), best_pos + context_chars + len(terms[0]))
        snippet = doc_text[start:end]

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(doc_text) else ""
        snippet = f"{prefix}{snippet}{suffix}"

        return self._highlight(snippet, terms)

    def _edit_distance(self, a: str, b: str) -> int:
        """Damerau-Levenshtein distance with transposition support."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
                if (
                    i > 1
                    and j > 1
                    and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]
                ):
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

        return dp[m][n]

    def _suggest(self, term: str, max_distance: int = 2) -> list[str]:
        """Find index terms within edit distance of the query term."""
        suggestions = []
        for word in self.indexer.index:
            dist = self._edit_distance(term, word)
            if 0 < dist <= max_distance:
                suggestions.append((word, dist))
        suggestions.sort(key=lambda x: x[1])
        return [word for word, _ in suggestions[:3]]

    def print_term(self, word: str) -> str:
        """Return formatted inverted index details for a single term."""
        if not self.is_ready():
            return "Index not loaded. Please run 'build' or 'load' first."

        term = word.lower().strip()
        if not term:
            return "Please provide a word to look up."

        entry = self.indexer.get_term(term)
        if entry is None:
            msg = f'Term "{term}" not found in the index.'
            suggestions = self._suggest(term)
            if suggestions:
                msg += f"\nDid you mean: {', '.join(suggestions)}?"
            return msg

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

        start_time = time.perf_counter()

        terms = [w.lower().strip() for w in words if w.strip()]
        if not terms:
            return "Please provide at least one search term."

        posting_sets: list[set[str]] = []
        term_entries: dict[str, dict] = {}

        missing_terms = []
        for term in terms:
            entry = self.indexer.get_term(term)
            if entry is None:
                missing_terms.append(term)
            else:
                posting_sets.append(set(entry["postings"].keys()))
                term_entries[term] = entry

        if missing_terms:
            query_str = " ".join(terms)
            msg = f'No results found for "{query_str}".'
            all_suggestions = []
            for t in missing_terms:
                all_suggestions.extend(self._suggest(t))
            if all_suggestions:
                unique = list(dict.fromkeys(all_suggestions))
                msg += f"\nDid you mean: {', '.join(unique[:3])}?"
            return msg

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

        elapsed = time.perf_counter() - start_time

        lines = []
        lines.append(f"Found {len(scored_results)} results in {elapsed:.3f}s")
        lines.append("")

        for rank, (url, score) in enumerate(scored_results, 1):
            lines.append(f"  {rank}. {url} (TF-IDF: {score})")
            snippet = self._snippet(url, terms)
            if snippet:
                lines.append(f'     "{snippet}"')
            lines.append("")

        return "\n".join(lines).rstrip()
