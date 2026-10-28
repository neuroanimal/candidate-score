"""Text similarity: TF-IDF (word stems and character n-grams), BM25 and optional
multilingual sentence embeddings.

Embeddings (``sentence-transformers``, a small multilingual encoder - not an LLM)
are used only when the package is installed **and** the model is already in the
local cache, or when explicitly enabled with ``CVSCORE_EMBEDDINGS=download``. The
tool therefore never needs network access to run.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np

from . import textnorm

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

EMBEDDING_MODEL = os.environ.get("CVSCORE_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def _analyzer(text: str) -> list[str]:
    return textnorm.content_stems(text)


def _cos(a, b) -> float:
    num = a.multiply(b).sum() if hasattr(a, "multiply") else float(np.dot(a, b))
    da = math.sqrt(a.multiply(a).sum()) if hasattr(a, "multiply") else float(np.linalg.norm(a))
    db = math.sqrt(b.multiply(b).sum()) if hasattr(b, "multiply") else float(np.linalg.norm(b))
    return float(num / (da * db)) if da and db else 0.0


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of stemmed-word TF-IDF vectors.

    IDF is learned from the sentences of both texts, so frequent words of the pair
    (e.g. "praca") weigh less than the distinctive ones.
    """
    if TfidfVectorizer is None:
        return token_overlap(text_a, text_b)
    corpus = textnorm.sentences(text_a) + textnorm.sentences(text_b)
    if len(corpus) < 2:
        return token_overlap(text_a, text_b)
    vec = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True, min_df=1)
    try:
        vec.fit(corpus)
    except ValueError:
        return 0.0
    mat = vec.transform([text_a, text_b])
    return _cos(mat[0], mat[1])


def char_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of character 3-5-gram TF-IDF vectors (robust to inflection/typos)."""
    if TfidfVectorizer is None:
        return token_overlap(text_a, text_b)
    corpus = [textnorm.fold(s) for s in textnorm.sentences(text_a) + textnorm.sentences(text_b)]
    if len(corpus) < 2:
        return token_overlap(text_a, text_b)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    try:
        vec.fit(corpus)
    except ValueError:
        return 0.0
    mat = vec.transform([textnorm.fold(text_a), textnorm.fold(text_b)])
    return _cos(mat[0], mat[1])


def token_overlap(query: str, text: str) -> float:
    """Share of the query's content stems present in the text (0..1)."""
    q = set(textnorm.content_stems(query, textnorm.GENERIC_REQUIREMENT_WORDS))
    if not q:
        return 0.0
    t = set(textnorm.content_stems(text))
    return len(q & t) / len(q)


class LineIndex:
    """BM25 index over lines/statements of a document, for requirement matching."""

    def __init__(self, lines: list[str]):
        self.lines = [ln for ln in lines if ln.strip()]
        self.tokens = [textnorm.content_stems(ln) for ln in self.lines]
        self.bm25 = BM25Okapi(self.tokens) if BM25Okapi and any(self.tokens) else None
        # document frequency of stems across lines: a word on every line ("prowadzilam")
        # says less about a match than a rare one ("dokumentacja")
        self.df: dict[str, int] = {}
        for toks in self.tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, stem_: str) -> float:
        """Smoothed inverse document frequency; unseen stems get the highest weight."""
        n = max(1, len(self.tokens))
        return math.log((n + 1) / (self.df.get(stem_, 0) + 0.5))

    def best(self, query: str, top: int = 3) -> list[tuple[str, float, float]]:
        """Best lines for a query: ``(line, bm25_normalised, overlap)``.

        BM25 scores are normalised by the score the query would get against itself
        (an upper bound), giving roughly 0..1.
        """
        if not self.lines:
            return []
        q = textnorm.content_stems(query, textnorm.GENERIC_REQUIREMENT_WORDS)
        if not q:
            return []
        qset = set(q)
        weights = {s: self.idf(s) for s in qset}
        total = sum(weights.values()) or 1.0
        # IDF-weighted share of the query found in the line
        overlaps = [sum(weights[s] for s in qset & set(t)) / total for t in self.tokens]
        if self.bm25 is not None:
            scores = self.bm25.get_scores(q)
            ceiling = max(float(np.max(scores)), 1e-9)
            # normalise against the best line, damped by overlap so a single rare
            # shared word does not look like a perfect match
            norm = [float(s) / ceiling * min(1.0, 0.35 + ov) if ceiling > 0 else 0.0
                    for s, ov in zip(scores, overlaps)]
        else:
            norm = overlaps
        order = sorted(range(len(self.lines)), key=lambda i: (-overlaps[i], -norm[i]))[:top]
        return [(self.lines[i], norm[i], overlaps[i]) for i in order]


# ----------------------------------------------------------------- embeddings


@lru_cache(maxsize=1)
def _embedder():
    mode = os.environ.get("CVSCORE_EMBEDDINGS", "auto").lower()
    if mode in ("0", "off", "no", "false"):
        return None
    # quiet, and strictly offline unless a download was requested explicitly
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if mode != "download":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=mode != "download", device="cpu")
    except Exception:  # pylint: disable=broad-exception-caught
        return None


_VEC_CACHE: dict[str, np.ndarray] = {}


def _encode(model, texts: list[str]) -> np.ndarray:
    """Encode with a per-process cache (CV lines are compared with every requirement)."""
    missing = [t for t in dict.fromkeys(texts) if t not in _VEC_CACHE]
    if missing:
        vecs = model.encode(missing, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        for text, vec in zip(missing, vecs):
            _VEC_CACHE[text] = vec
        if len(_VEC_CACHE) > 50_000:
            _VEC_CACHE.clear()
    return np.array([_VEC_CACHE[t] for t in texts])


def embeddings_available() -> bool:
    """True when a sentence-embedding model can be used offline."""
    return _embedder() is not None


def embedding_similarity(text_a: str, text_b: str) -> float | None:
    """Cosine similarity of mean sentence embeddings, or ``None`` if unavailable."""
    model = _embedder()
    if model is None:
        return None
    sa = textnorm.sentences(text_a)[:200] or [text_a]
    sb = textnorm.sentences(text_b)[:200] or [text_b]
    va = _encode(model, sa).mean(axis=0)
    vb = _encode(model, sb).mean(axis=0)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def embedding_best_match(query: str, lines: list[str]) -> tuple[str, float] | None:
    """Most similar line to the query by embeddings, or ``None`` if unavailable."""
    model = _embedder()
    if model is None or not lines:
        return None
    vecs = _encode(model, [query] + lines)
    sims = vecs[1:] @ vecs[0]
    idx = int(np.argmax(sims))
    return lines[idx], float(sims[idx])
