from unittest.mock import MagicMock, patch

from src.german_tutor.weaviate_client import chunk_text, extract_lemmas, fetch_url


def _make_token(lemma, pos, is_alpha=True, is_stop=False):
    t = MagicMock()
    t.lemma_ = lemma
    t.pos_ = pos
    t.is_alpha = is_alpha
    t.is_stop = is_stop
    return t


class TestChunkText:
    def test_single_chunk_when_short(self):
        text = "Hello world"
        assert chunk_text(text, chunk_words=10) == ["Hello world"]

    def test_splits_into_multiple_chunks(self):
        words = ["word"] * 500
        chunks = chunk_text(" ".join(words), chunk_words=200, overlap_words=30)
        assert len(chunks) > 1

    def test_chunks_overlap(self):
        words = [str(i) for i in range(250)]
        chunks = chunk_text(" ".join(words), chunk_words=200, overlap_words=50)
        last_of_first = chunks[0].split()[-1]
        assert last_of_first in chunks[1].split()

    def test_last_chunk_contains_final_words(self):
        words = [str(i) for i in range(250)]
        chunks = chunk_text(" ".join(words), chunk_words=200, overlap_words=30)
        assert "249" in chunks[-1]

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []


class TestFetchUrl:
    def _make_response(self, html: str, status: int = 200):
        resp = MagicMock()
        resp.status_code = status
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_title_and_text(self):
        html = "<html><head><title>Test Article</title></head><body><p>Hello world.</p></body></html>"
        with patch(
            "src.german_tutor.weaviate_client.requests.get",
            return_value=self._make_response(html),
        ):
            title, text = fetch_url("http://example.com")
        assert title == "Test Article"
        assert "Hello world" in text

    def test_falls_back_to_url_when_no_title(self):
        html = "<html><body><p>Some text.</p></body></html>"
        with patch(
            "src.german_tutor.weaviate_client.requests.get",
            return_value=self._make_response(html),
        ):
            title, _ = fetch_url("http://example.com/article")
        assert title == "http://example.com/article"

    def test_http_error_raises_runtime_error(self):
        import requests as req

        resp = MagicMock()
        resp.raise_for_status.side_effect = req.HTTPError("404")
        with patch("src.german_tutor.weaviate_client.requests.get", return_value=resp):
            import pytest

            with pytest.raises(RuntimeError, match="Failed to fetch URL"):
                fetch_url("http://example.com")

    def test_only_paragraph_text_extracted(self):
        html = (
            "<html><body>"
            "<nav>Skip nav</nav>"
            "<p>Article paragraph.</p>"
            "<footer>Footer</footer>"
            "</body></html>"
        )
        with patch(
            "src.german_tutor.weaviate_client.requests.get",
            return_value=self._make_response(html),
        ):
            _, text = fetch_url("http://example.com")
        assert "Article paragraph" in text
        assert "Skip nav" not in text
        assert "Footer" not in text


class TestExtractLemmas:
    def _mock_nlp(self, tokens):
        mock_nlp = MagicMock(return_value=tokens)
        return mock_nlp

    def test_returns_content_word_lemmas(self):
        tokens = [
            _make_token("Regierung", "NOUN"),
            _make_token("beschließen", "VERB"),
        ]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("Die Regierung beschloss.")
        assert "regierung" in result
        assert "beschließen" in result

    def test_excludes_stopwords(self):
        tokens = [_make_token("die", "DET", is_stop=True)]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("Die")
        assert result == ""

    def test_excludes_non_alpha_tokens(self):
        tokens = [_make_token("123", "NUM", is_alpha=False)]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("123")
        assert result == ""

    def test_excludes_non_content_pos(self):
        tokens = [_make_token("schnell", "ADV")]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("schnell")
        assert result == ""

    def test_deduplicates_lemmas(self):
        tokens = [
            _make_token("Regierung", "NOUN"),
            _make_token("Regierung", "NOUN"),
        ]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("Regierung Regierung")
        assert result.count("regierung") == 1

    def test_result_is_sorted(self):
        tokens = [
            _make_token("Zeitung", "NOUN"),
            _make_token("Artikel", "NOUN"),
        ]
        with patch(
            "src.german_tutor.weaviate_client._get_nlp",
            return_value=self._mock_nlp(tokens),
        ):
            result = extract_lemmas("Zeitung Artikel")
        assert result == "artikel zeitung"
