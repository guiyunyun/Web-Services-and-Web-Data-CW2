"""Unit tests for the indexer module."""

import os
import json
import pickle
import tempfile

import pytest
from src.indexer import Indexer, STOP_WORDS

SAMPLE_DOCS = [
    {"url": "http://example.com/1", "text": "Love is great. Love makes life beautiful."},
    {"url": "http://example.com/2", "text": "Life is good and wonderful."},
    {"url": "http://example.com/3", "text": "Love life and be happy every day."},
]


class TestTokenize:
    """Tests for the tokenize method."""

    def test_basic_tokenization(self):
        idx = Indexer()
        tokens = idx.tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_converts_to_lowercase(self):
        idx = Indexer()
        tokens = idx.tokenize("UPPER lower MiXeD")
        assert tokens == ["upper", "lower", "mixed"]

    def test_removes_punctuation(self):
        idx = Indexer()
        tokens = idx.tokenize("hello, world! it's a test.")
        assert tokens == ["hello", "world", "it", "s", "a", "test"]

    def test_handles_numbers(self):
        idx = Indexer()
        tokens = idx.tokenize("page 1 has 10 quotes")
        assert tokens == ["page", "1", "has", "10", "quotes"]

    def test_empty_text(self):
        idx = Indexer()
        tokens = idx.tokenize("")
        assert tokens == []

    def test_only_punctuation(self):
        idx = Indexer()
        tokens = idx.tokenize("!@#$%^&*()")
        assert tokens == []


class TestFilterStopWords:
    """Tests for stop words filtering."""

    def test_filters_stop_words(self):
        idx = Indexer(use_stop_words=True)
        tokens = ["love", "is", "the", "answer"]
        filtered = idx.filter_stop_words(tokens)
        assert "is" not in filtered
        assert "the" not in filtered
        assert "love" in filtered
        assert "answer" in filtered

    def test_disabled_stop_words(self):
        idx = Indexer(use_stop_words=False)
        tokens = ["love", "is", "the", "answer"]
        filtered = idx.filter_stop_words(tokens)
        assert filtered == tokens

    def test_all_stop_words(self):
        idx = Indexer(use_stop_words=True)
        tokens = ["the", "is", "a", "an"]
        filtered = idx.filter_stop_words(tokens)
        assert filtered == []

    def test_stop_words_list_is_nonempty(self):
        assert len(STOP_WORDS) > 50


class TestBuild:
    """Tests for the build method."""

    def test_builds_index(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        assert len(idx.index) > 0

    def test_term_frequency(self):
        idx = Indexer(use_stop_words=False)
        idx.build([
            {"url": "http://example.com/1", "text": "cat cat dog"},
        ])
        assert idx.index["cat"]["postings"]["http://example.com/1"]["tf"] == 2
        assert idx.index["dog"]["postings"]["http://example.com/1"]["tf"] == 1

    def test_positions_tracked(self):
        idx = Indexer(use_stop_words=False)
        idx.build([
            {"url": "http://example.com/1", "text": "a b c a b a"},
        ])
        positions = idx.index["a"]["postings"]["http://example.com/1"]["positions"]
        assert positions == [0, 3, 5]

    def test_document_frequency(self):
        idx = Indexer(use_stop_words=False)
        idx.build([
            {"url": "http://example.com/1", "text": "cat dog"},
            {"url": "http://example.com/2", "text": "cat bird"},
            {"url": "http://example.com/3", "text": "fish bird"},
        ])
        assert idx.index["cat"]["df"] == 2
        assert idx.index["dog"]["df"] == 1
        assert idx.index["bird"]["df"] == 2

    def test_stop_words_excluded_from_index(self):
        idx = Indexer(use_stop_words=True)
        idx.build([
            {"url": "http://example.com/1", "text": "the cat is on the mat"},
        ])
        assert "the" not in idx.index
        assert "is" not in idx.index
        assert "on" not in idx.index
        assert "cat" in idx.index
        assert "mat" in idx.index

    def test_metadata_populated(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        assert idx.metadata["total_documents"] == 3
        assert idx.metadata["total_terms"] == len(idx.index)
        assert "built_at" in idx.metadata
        assert len(idx.metadata["document_urls"]) == 3

    def test_documents_stored(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        assert idx.documents["http://example.com/1"] == SAMPLE_DOCS[0]["text"]

    def test_empty_documents(self):
        idx = Indexer()
        idx.build([])
        assert len(idx.index) == 0
        assert idx.metadata["total_documents"] == 0

    def test_document_with_empty_text(self):
        idx = Indexer()
        idx.build([{"url": "http://example.com/empty", "text": ""}])
        assert len(idx.index) == 0
        assert idx.metadata["total_documents"] == 1


class TestTfIdf:
    """Tests for TF-IDF calculation."""

    def test_tfidf_exists_in_postings(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        for term, entry in idx.index.items():
            for url, posting in entry["postings"].items():
                assert "tf_idf" in posting

    def test_rare_term_has_higher_tfidf(self):
        idx = Indexer(use_stop_words=False)
        idx.build([
            {"url": "http://example.com/1", "text": "cat dog"},
            {"url": "http://example.com/2", "text": "cat bird"},
            {"url": "http://example.com/3", "text": "cat fish"},
        ])
        # "cat" appears in all 3 docs (low IDF), "dog" in 1 doc (high IDF)
        cat_tfidf = idx.index["cat"]["postings"]["http://example.com/1"]["tf_idf"]
        dog_tfidf = idx.index["dog"]["postings"]["http://example.com/1"]["tf_idf"]
        assert dog_tfidf > cat_tfidf

    def test_term_in_all_docs_has_zero_idf(self):
        idx = Indexer(use_stop_words=False)
        idx.build([
            {"url": "http://example.com/1", "text": "hello"},
            {"url": "http://example.com/2", "text": "hello"},
        ])
        # log(2/2) = 0, so TF-IDF should be 0
        tfidf = idx.index["hello"]["postings"]["http://example.com/1"]["tf_idf"]
        assert tfidf == 0.0


class TestGetTerm:
    """Tests for the get_term method."""

    def test_finds_existing_term(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        result = idx.get_term("love")
        assert result is not None
        assert "df" in result
        assert "postings" in result

    def test_case_insensitive_lookup(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        assert idx.get_term("LOVE") == idx.get_term("love")

    def test_returns_none_for_missing_term(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)
        assert idx.get_term("xyznonexistent") is None


class TestSaveLoad:
    """Tests for save and load methods."""

    def test_save_and_load_json(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            idx.save(path, fmt="json")
            idx2 = Indexer()
            data = idx2.load(path)
            assert idx2.metadata["total_terms"] == idx.metadata["total_terms"]
            assert idx2.get_term("love") is not None
        finally:
            os.remove(path)

    def test_save_and_load_pickle(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            idx.save(path, fmt="pickle")
            idx2 = Indexer()
            data = idx2.load(path)
            assert idx2.metadata["total_terms"] == idx.metadata["total_terms"]
            assert idx2.get_term("love") is not None
        finally:
            os.remove(path)

    def test_auto_detect_json_format(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            idx.save(path, fmt="json")
            idx2 = Indexer()
            idx2.load(path)
            assert len(idx2.index) > 0
        finally:
            os.remove(path)

    def test_auto_detect_pickle_format(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            idx.save(path, fmt="pickle")
            idx2 = Indexer()
            idx2.load(path)
            assert len(idx2.index) > 0
        finally:
            os.remove(path)

    def test_validate_rejects_invalid_data(self):
        idx = Indexer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"metadata": {}, "index": {}}, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="missing keys"):
                idx.load(path)
        finally:
            os.remove(path)

    def test_load_file_not_found(self):
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load("/nonexistent/path/index.json")

    def test_json_is_human_readable(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            idx.save(path, fmt="json")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            data = json.loads(content)
            assert "metadata" in data
            assert "index" in data
        finally:
            os.remove(path)

    def test_pickle_smaller_than_json(self):
        idx = Indexer()
        idx.build(SAMPLE_DOCS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
            json_path = jf.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as pf:
            pkl_path = pf.name

        try:
            idx.save(json_path, fmt="json")
            idx.save(pkl_path, fmt="pickle")
            assert os.path.getsize(pkl_path) < os.path.getsize(json_path)
        finally:
            os.remove(json_path)
            os.remove(pkl_path)
