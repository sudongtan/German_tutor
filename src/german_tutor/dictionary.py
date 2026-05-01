"""
DWDS corpus frequency lookup + OpenThesaurus synonym lookup.

Exposes raw helpers (get_frequency, get_synonyms, lookup_frequencies) and
LangChain @tool wrappers (frequency_lookup, synonyms_lookup) for use in
LangGraph tool nodes.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_DWDS_API = "https://www.dwds.de/api/frequency/"
_OPENTHESAURUS_API = "https://www.openthesaurus.de/synonyme/search"


def get_frequency(word: str) -> int:
    url = _DWDS_API + "?q=" + urllib.parse.quote(word)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return int(data.get("hits") or 0)
    except (
        urllib.error.URLError,
        json.JSONDecodeError,
        ValueError,
        KeyError,
        OSError,
    ) as e:
        logger.warning("DWDS lookup failed for %r: %s", word, e)
        return 0


def get_synonyms(word: str) -> list[list[dict]]:
    url = (
        _OPENTHESAURUS_API
        + "?q="
        + urllib.parse.quote(word)
        + "&format=application/json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GermanTutor/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [synset["terms"] for synset in data.get("synsets", [])]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning("OpenThesaurus lookup failed for %r: %s", word, e)
        return []


def lookup_frequencies(words: list[str]) -> dict[str, int]:
    return {word: get_frequency(word) for word in words}


@tool
def frequency_lookup(word: str) -> str:
    """Look up how often a German word appears in the DWDS corpus (a large collection of written German texts). Useful for comparing how common words are in everyday written German."""
    hits = get_frequency(word)
    return f"'{word}' appears {hits:,} times in the DWDS corpus."


@tool
def synonyms_lookup(word: str) -> str:
    """Look up synonyms of a German word from OpenThesaurus, grouped by meaning. Each synonym may include a register label such as 'umgangssprachlich' (colloquial), 'gehoben' (elevated/formal), or 'derb' (crude). Useful for understanding the stylistic level and semantic range of a word."""
    groups = get_synonyms(word)
    if not groups:
        return f"No synonyms found for '{word}'."
    lines = []
    for group in groups:
        parts = [
            f"{t['term']} ({t['level']})" if "level" in t else t["term"] for t in group
        ]
        lines.append(", ".join(parts))
    return "Synonym groups for '{}':\n{}".format(
        word, "\n".join(f"  • {line}" for line in lines)
    )
