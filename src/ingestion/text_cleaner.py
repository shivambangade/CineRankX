"""NLTK-based text cleaning: lowercase, stop-word removal, stemming."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _ensure_nltk_data() -> None:
    for resource in ("corpora/stopwords",):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


_ensure_nltk_data()
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    tokens = _TOKEN_RE.findall(text.lower())
    stemmed = [_STEMMER.stem(tok) for tok in tokens if tok not in _STOPWORDS]
    return " ".join(stemmed)
