import json
import urllib.error
from unittest.mock import MagicMock, patch

from src.german_tutor.dictionary import get_frequency, get_synonyms, lookup_frequencies


def make_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestGetFrequency:
    def test_returns_hits_from_api(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response({"hits": 42, "total": "1000", "frequency": 3}),
        ):
            assert get_frequency("müde") == 42

    def test_null_hits_returns_zero(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response({"hits": None}),
        ):
            assert get_frequency("müde") == 0

    def test_missing_hits_key_returns_zero(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response({}),
        ):
            assert get_frequency("müde") == 0

    def test_network_error_returns_zero(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            assert get_frequency("müde") == 0

    def test_malformed_json_returns_zero(self):
        resp = MagicMock()
        resp.read.return_value = b"not-json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=resp,
        ):
            assert get_frequency("müde") == 0

    def test_word_is_url_encoded_in_request(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response({"count": 0}),
        ) as mock:
            get_frequency("sich freuen")
        url = mock.call_args[0][0]
        assert "sich+freuen" in url or "sich%20freuen" in url


class TestGetSynonyms:
    def test_returns_synonym_groups(self):
        data = {
            "synsets": [
                {
                    "terms": [
                        {"term": "müde"},
                        {"term": "dösig", "level": "umgangssprachlich"},
                    ]
                }
            ]
        }
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response(data),
        ):
            result = get_synonyms("müde")
        assert result == [
            [{"term": "müde"}, {"term": "dösig", "level": "umgangssprachlich"}]
        ]

    def test_network_error_returns_empty(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            assert get_synonyms("müde") == []

    def test_malformed_json_returns_empty(self):
        resp = MagicMock()
        resp.read.return_value = b"not-json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=resp,
        ):
            assert get_synonyms("müde") == []

    def test_empty_synsets_returns_empty(self):
        with patch(
            "src.german_tutor.dictionary.urllib.request.urlopen",
            return_value=make_response({"synsets": []}),
        ):
            assert get_synonyms("müde") == []


class TestLookupFrequencies:
    def test_returns_dict_of_all_words(self):
        counts = {"Trauer": 10, "Kummer": 5}
        with patch(
            "src.german_tutor.dictionary.get_frequency",
            side_effect=lambda w: counts.get(w, 0),
        ):
            result = lookup_frequencies(["Trauer", "Kummer"])
        assert result == {"Trauer": 10, "Kummer": 5}

    def test_empty_list_returns_empty_dict(self):
        result = lookup_frequencies([])
        assert result == {}
