"""Unit tests for the search module."""

import pytest
from src.indexer import Indexer
from src.search import SearchEngine, HIGHLIGHT, RESET

SAMPLE_DOCS = [
    {"url": "http://example.com/1", "text": "Love is great. Love makes life beautiful."},
    {"url": "http://example.com/2", "text": "Life is good and wonderful."},
    {"url": "http://example.com/3", "text": "Love life and be happy every day."},
]


def build_engine(docs=None, use_stop_words=True):
    """Helper to create a SearchEngine with a built index."""
    idx = Indexer(use_stop_words=use_stop_words)
    idx.build(docs if docs is not None else SAMPLE_DOCS)
    return SearchEngine(idx)


def build_empty_engine():
    """Helper to create a SearchEngine with no index loaded."""
    idx = Indexer()
    return SearchEngine(idx)


class TestEditDistance:
    """Tests for Damerau-Levenshtein edit distance."""

    def test_identical_strings(self):
        se = build_engine()
        assert se._edit_distance("love", "love") == 0

    def test_single_insertion(self):
        se = build_engine()
        assert se._edit_distance("lov", "love") == 1

    def test_single_deletion(self):
        se = build_engine()
        assert se._edit_distance("loves", "love") == 1

    def test_single_replacement(self):
        se = build_engine()
        assert se._edit_distance("lova", "love") == 1

    def test_transposition(self):
        se = build_engine()
        assert se._edit_distance("lvoe", "love") == 1

    def test_completely_different(self):
        se = build_engine()
        assert se._edit_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        se = build_engine()
        assert se._edit_distance("", "") == 0

    def test_one_empty_string(self):
        se = build_engine()
        assert se._edit_distance("", "love") == 4
        assert se._edit_distance("love", "") == 4


class TestSuggest:
    """Tests for query suggestion."""

    def test_suggests_similar_term(self):
        se = build_engine()
        suggestions = se._suggest("lov")
        assert "love" in suggestions

    def test_transposition_suggestion(self):
        se = build_engine()
        suggestions = se._suggest("lvoe")
        assert "love" in suggestions

    def test_no_suggestion_for_distant_term(self):
        se = build_engine()
        suggestions = se._suggest("xyzabc")
        assert suggestions == []

    def test_exact_match_excluded(self):
        se = build_engine()
        suggestions = se._suggest("love")
        assert "love" not in suggestions

    def test_returns_max_three(self):
        se = build_engine()
        suggestions = se._suggest("a")
        assert len(suggestions) <= 3

    def test_sorted_by_distance(self):
        se = build_engine(use_stop_words=False)
        suggestions = se._suggest("lov")
        if len(suggestions) >= 2:
            dists = [se._edit_distance("lov", s) for s in suggestions]
            assert dists == sorted(dists)


class TestPrintTerm:
    """Tests for print_term command."""

    def test_shows_existing_term(self):
        se = build_engine()
        result = se.print_term("love")
        assert 'Term: "love"' in result
        assert "Document Frequency:" in result
        assert "Term Frequency:" in result
        assert "Positions:" in result
        assert "TF-IDF:" in result

    def test_case_insensitive(self):
        se = build_engine()
        assert se.print_term("LOVE") == se.print_term("love")

    def test_not_found(self):
        se = build_engine()
        result = se.print_term("xyznonexistent")
        assert "not found" in result

    def test_not_found_with_suggestion(self):
        se = build_engine()
        result = se.print_term("lov")
        assert "not found" in result
        assert "Did you mean:" in result
        assert "love" in result

    def test_empty_input(self):
        se = build_engine()
        result = se.print_term("")
        assert "Please provide a word" in result

    def test_whitespace_input(self):
        se = build_engine()
        result = se.print_term("   ")
        assert "Please provide a word" in result

    def test_index_not_loaded(self):
        se = build_empty_engine()
        result = se.print_term("love")
        assert "Index not loaded" in result

    def test_results_sorted_by_tfidf(self):
        se = build_engine()
        result = se.print_term("love")
        lines = result.split("\n")
        tfidf_values = []
        for line in lines:
            if "TF-IDF:" in line:
                score = float(line.strip().split("TF-IDF: ")[1])
                tfidf_values.append(score)
        assert tfidf_values == sorted(tfidf_values, reverse=True)


class TestFind:
    """Tests for find command."""

    def test_single_word_search(self):
        se = build_engine()
        result = se.find(["love"])
        assert "Found" in result
        assert "results in" in result
        assert "http://example.com/" in result

    def test_multi_word_and_query(self):
        se = build_engine()
        result = se.find(["love", "life"])
        assert "Found" in result

    def test_no_results(self):
        se = build_engine()
        result = se.find(["xyznonexistent"])
        assert "No results found" in result

    def test_no_results_with_suggestion(self):
        se = build_engine()
        result = se.find(["lov"])
        assert "No results found" in result
        assert "Did you mean:" in result

    def test_empty_query(self):
        se = build_engine()
        result = se.find([])
        assert "Please provide at least one search term" in result

    def test_whitespace_only_query(self):
        se = build_engine()
        result = se.find(["  ", ""])
        assert "Please provide at least one search term" in result

    def test_index_not_loaded(self):
        se = build_empty_engine()
        result = se.find(["love"])
        assert "Index not loaded" in result

    def test_results_ranked_by_tfidf(self):
        se = build_engine()
        result = se.find(["love"])
        lines = result.split("\n")
        scores = []
        for line in lines:
            if "TF-IDF:" in line:
                score = float(line.split("TF-IDF: ")[1].rstrip(")"))
                scores.append(score)
        assert scores == sorted(scores, reverse=True)

    def test_and_intersection(self):
        docs = [
            {"url": "http://example.com/a", "text": "cat dog"},
            {"url": "http://example.com/b", "text": "cat bird"},
            {"url": "http://example.com/c", "text": "dog bird"},
        ]
        se = build_engine(docs, use_stop_words=False)
        result = se.find(["cat", "dog"])
        assert "http://example.com/a" in result
        assert "http://example.com/b" not in result
        assert "http://example.com/c" not in result

    def test_no_common_documents(self):
        docs = [
            {"url": "http://example.com/a", "text": "cat"},
            {"url": "http://example.com/b", "text": "dog"},
        ]
        se = build_engine(docs, use_stop_words=False)
        result = se.find(["cat", "dog"])
        assert "No results found" in result

    def test_shows_timing(self):
        se = build_engine()
        result = se.find(["love"])
        assert "results in" in result
        assert "s" in result


class TestSnippet:
    """Tests for snippet extraction."""

    def test_snippet_contains_term(self):
        se = build_engine()
        snippet = se._snippet("http://example.com/1", ["love"])
        assert "LOVE" in snippet or "love" in snippet.lower()

    def test_snippet_has_ellipsis(self):
        long_doc = [{"url": "http://example.com/long",
                      "text": "a " * 50 + "love " + "b " * 50}]
        se = build_engine(long_doc)
        snippet = se._snippet("http://example.com/long", ["love"])
        assert "..." in snippet

    def test_snippet_empty_for_missing_doc(self):
        se = build_engine()
        snippet = se._snippet("http://nonexistent.com", ["love"])
        assert snippet == ""

    def test_snippet_empty_for_unmatched_term(self):
        se = build_engine()
        snippet = se._snippet("http://example.com/1", ["xyznonexistent"])
        assert snippet == ""

    def test_find_includes_snippet(self):
        se = build_engine()
        result = se.find(["love"])
        assert '"' in result


class TestHighlight:
    """Tests for ANSI highlighting."""

    def test_highlights_term(self):
        se = build_engine()
        result = se._highlight("I love you", ["love"])
        assert HIGHLIGHT in result
        assert RESET in result
        assert "LOVE" in result

    def test_case_insensitive_highlight(self):
        se = build_engine()
        result = se._highlight("Love and LOVE", ["love"])
        assert result.count(HIGHLIGHT) == 2

    def test_no_match_unchanged(self):
        se = build_engine()
        original = "hello world"
        result = se._highlight(original, ["xyz"])
        assert result == original

    def test_multiple_terms(self):
        se = build_engine()
        result = se._highlight("love life", ["love", "life"])
        assert "LOVE" in result
        assert "LIFE" in result


class TestIsReady:
    """Tests for is_ready method."""

    def test_ready_after_build(self):
        se = build_engine()
        assert se.is_ready() is True

    def test_not_ready_when_empty(self):
        se = build_empty_engine()
        assert se.is_ready() is False
