"""Hybrid retrieval: TF-IDF (dense-ish) + BM25 (lexical), fused via RRF.

Why this shape:
  - BM25 is the right default for short, exact-term queries ("Snell's law",
    "discriminant") — it rewards rare terms and handles length normalisation.
  - TF-IDF with cosine over L2-normalised vectors captures overlap at a
    different scale (document-level term presence) and acts as a second
    opinion. For this corpus size (~12 documents, ~200 unique terms) a
    custom TF-IDF in ~30 lines of numpy is plenty — no torch, no downloads.
  - Reciprocal Rank Fusion is a tuning-free way to merge the two rankings,
    robust to the fact that the two scorers have different score scales.

Upgrade path: swap TfidfIndex for a dense-embedding backend
(sentence-transformers or Voyage) without changing the RRF layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .data import Material

RRF_K = 60  # standard RRF constant
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalHit:
    material: Material
    score: float
    dense_rank: int
    bm25_rank: int


class BM25Index:
    """Minimal BM25 (Okapi) over a fixed corpus. Standard k1=1.5, b=0.75."""

    def __init__(self, tokenized: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        n_docs = len(tokenized)
        self.doc_lens = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if n_docs else 0.0
        self.term_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in tokenized:
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        # Okapi BM25 idf with +1 floor to keep common terms non-negative.
        self.idf = {
            t: float(np.log((n_docs - c + 0.5) / (c + 0.5) + 1.0))
            for t, c in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        n = len(self.doc_lens)
        scores = np.zeros(n, dtype=np.float32)
        if not query_tokens:
            return scores
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf_map in enumerate(self.term_freqs):
                f = tf_map.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class TfidfIndex:
    """Minimal numpy TF-IDF with cosine similarity. ~30 lines, zero heavy deps."""

    def __init__(self, texts: list[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for toks in tokenized:
            for tok in toks:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        n_docs = len(texts)
        n_terms = len(vocab)
        tf = np.zeros((n_docs, n_terms), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            for tok in toks:
                tf[i, vocab[tok]] += 1.0
        df = (tf > 0).sum(axis=0)
        # Smoothed idf; +1 avoids div-by-zero and zero-weights on ubiquitous terms.
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        weighted = tf * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = weighted / norms  # L2-normalised rows => cosine == dot

    def scores(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in toks:
            idx = self.vocab.get(tok)
            if idx is not None:
                q[idx] += 1.0
        q *= self.idf
        n = float(np.linalg.norm(q))
        if n == 0.0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return self.matrix @ (q / n)


class HybridRetriever:
    def __init__(self, materials: list[Material]):
        if not materials:
            raise ValueError("HybridRetriever requires at least one material")
        self.materials = materials
        texts = [m.searchable_text() for m in materials]
        self.tfidf = TfidfIndex(texts)
        self.bm25 = BM25Index([tokenize(t) for t in texts])

    def search(
        self,
        query: str,
        top_k: int = 3,
        topic_filter: list[str] | None = None,
    ) -> list[RetrievalHit]:
        q_tokens = tokenize(query)
        dense_scores = self.tfidf.scores(query)
        bm25_scores = self.bm25.scores(q_tokens)

        dense_order = np.argsort(-dense_scores)
        bm25_order = np.argsort(-bm25_scores)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order)}
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        n = len(self.materials)
        rrf = np.zeros(n, dtype=np.float32)
        for idx, rank in dense_rank.items():
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        for idx, rank in bm25_rank.items():
            rrf[idx] += 1.0 / (RRF_K + rank + 1)

        allowed: set[int] | None = None
        if topic_filter:
            wanted = {t.strip().lower() for t in topic_filter}
            allowed = {i for i, m in enumerate(self.materials) if m.topic.lower() in wanted}
            if not allowed:
                return []

        candidates = sorted(
            (i for i in range(n) if allowed is None or i in allowed),
            key=lambda i: -rrf[i],
        )

        hits: list[RetrievalHit] = []
        for i in candidates[:top_k]:
            # Drop zero-signal hits (neither dense nor BM25 scored anything) —
            # but only when no topic filter was given. An explicit topic filter
            # is itself a strong signal of intent; the student asked for *this*
            # topic even if their phrasing doesn't lexically overlap.
            if topic_filter is None and dense_scores[i] == 0 and bm25_scores[i] == 0:
                continue
            hits.append(
                RetrievalHit(
                    material=self.materials[i],
                    score=float(rrf[i]),
                    dense_rank=dense_rank[i],
                    bm25_rank=bm25_rank[i],
                )
            )
        return hits


__all__ = ["HybridRetriever", "RetrievalHit", "TfidfIndex", "tokenize"]
