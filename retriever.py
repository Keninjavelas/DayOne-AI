"""Hybrid retrieval for DayOne AI.

Architecture:
    1. Dense retrieval  — pgvector cosine ANN
  2. Sparse retrieval — BM25Okapi on same corpus
  3. Fusion           — Reciprocal Rank Fusion (RRF, k=60)
  4. Reranking        — cross-encoder/ms-marco-MiniLM-L-6-v2 (toggleable)
  5. Confidence       — sigmoid(top_reranker_score) when reranker ON;
                        normalised BM25 score when reranker OFF

Design trade-offs documented inline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from sqlalchemy import text

from backend.services.auth_db import require_engine

# ---------------------------------------------------------------------------
# Config — override via environment variable
# ---------------------------------------------------------------------------
# Trade-off: reranker adds ~200-400 ms CPU latency but improves precision by
# ~10-16 pp on our benchmark (see eval.py results). Default ON for correctness.
USE_RERANKER: bool = os.getenv("DAYONE_USE_RERANKER", "1") != "0"
PGVECTOR_PROBES: int = int(os.getenv("DAYONE_PGVECTOR_PROBES", "10"))
EMBEDDING_DIM: int = int(os.getenv("DAYONE_EMBEDDING_DIM", "384"))
ASSERT_TENANT_ISOLATION: bool = os.getenv("DAYONE_ASSERT_TENANT_ISOLATION", "0") == "1"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Retrieve this many candidates before reranking. Wider net = better recall
# at the cost of reranker compute. 12 balances recall vs latency on CPU.
CANDIDATE_K: int = 12
FINAL_K: int = 4       # Docs passed to the LLM
RRF_K: int = 60        # RRF smoothing constant (standard value from literature)

# Confidence thresholds
CONF_LOW: float = 0.40   # Below this → show "low confidence" warning in UI
CONF_HIGH: float = 0.70  # Above this → high confidence

_cross_encoder_cache: Dict[str, Any] = {}


@dataclass
class RetrievalResult:
    """Structured result from HybridRetriever.retrieve().

    Carries everything needed for the Answer Justification layer:
    - final_docs / final_scores  : what the LLM sees
    - candidates / candidate_scores : what existed before reranking
    - rank_changes               : index delta per final doc
    """
    final_docs: List[Any]          # Top final_k docs sent to LLM
    final_scores: List[float]      # CE scores (or BM25 proxies) for final docs
    confidence: float              # [0, 1] confidence estimate
    candidates: List[Any]          # All candidate_k docs before reranking
    candidate_scores: List[float]  # Scores of candidates (BM25 or CE pre-sort)
    latency_ms: float
    used_reranker: bool
    dense_topk: List[Dict[str, Any]] = field(default_factory=list)
    sparse_topk: List[Dict[str, Any]] = field(default_factory=list)
    fused_topk: List[Dict[str, Any]] = field(default_factory=list)
    reranked_topk: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def rank_changes(self) -> List[int]:
        """For each final doc, how many positions it moved up during reranking.

        Positive = moved up (promoted by reranker). Zero = same position.
        Allows UI to show ↑N next to each chunk.
        """
        changes: List[int] = []
        for doc in self.final_docs:
            try:
                candidate_pos = next(
                    i for i, c in enumerate(self.candidates)
                    if c.page_content == doc.page_content
                )
                final_pos = self.final_docs.index(doc)
                changes.append(candidate_pos - final_pos)
            except StopIteration:
                changes.append(0)
        return changes


def _get_cross_encoder() -> Any:
    """Lazy-load and cache the cross-encoder (one instance per process)."""
    if RERANKER_MODEL not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        _cross_encoder_cache[RERANKER_MODEL] = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder_cache[RERANKER_MODEL]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _normalize_vector(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {arr.shape[0]}"
        )
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Zero-norm embedding encountered")
    arr = arr / norm
    return arr.tolist()


class HybridRetriever:
    """BM25 + pgvector hybrid retriever with optional cross-encoder reranking."""

    def __init__(
        self,
        embeddings: Any,
        docs: List[Any],
        dense_indices_fn: Callable[[str, int], List[Tuple[int, float]]],
        use_reranker: bool = USE_RERANKER,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.embeddings = embeddings
        self.use_reranker = use_reranker
        self._dense_indices_fn = dense_indices_fn
        # Optional per-source reputation weights from FeedbackStore.
        # Keys are bare filenames (e.g. "handbook.pdf"); values centred at 1.0.
        self._source_weights: Dict[str, float] = source_weights or {}

        self._docs = docs

        if self._docs:
            corpus = [doc.page_content.lower().split() for doc in self._docs]
            self._bm25: Optional[BM25Okapi] = BM25Okapi(corpus)
        else:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_hits(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return dense retrieval hits for the top-k results."""
        return self._dense_indices_fn(query, k)

    def _sparse_indices(self, query: str, k: int) -> Tuple[List[int], "np.ndarray"]:
        """Return BM25-ranked indices and the full score array."""
        if self._bm25 is None:
            return [], np.array([], dtype=np.float32)
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indices = list(np.argsort(scores)[::-1][:k])
        return indices, scores

    def _apply_source_weights(self, docs: List[Any], scores: List[float]) -> List[float]:
        """Multiply each score by its source's reputation weight.

        Looks up each doc's source filename in self._source_weights.
        Unknown sources get weight 1.0 (neutral).
        """
        if not self._source_weights:
            return scores
        from pathlib import Path as _Path
        weighted: List[float] = []
        for doc, score in zip(docs, scores):
            source_name = _Path(str(doc.metadata.get("source", ""))).name
            weight = self._source_weights.get(source_name, 1.0)
            weighted.append(score * weight)
        return weighted

    @staticmethod
    def _rrf_score_map(
        dense: List[int], sparse: List[int], k: int = RRF_K
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
        for rank, idx in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
        return scores

    @classmethod
    def _rrf_fuse(
        cls, dense: List[int], sparse: List[int], k: int = RRF_K
    ) -> List[int]:
        """Reciprocal Rank Fusion across two ranked lists.

        score(d) = Σ  1 / (rank(d, list) + k)
        k=60 is the standard value from Cormack et al. (2009).
        """
        scores = cls._rrf_score_map(dense, sparse, k=k)
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def _build_trace_items(
        self,
        indices: List[int],
        scores: List[float],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[:limit]):
            if idx >= len(self._docs):
                continue
            doc = self._docs[idx]
            metadata = getattr(doc, "metadata", {}) or {}
            items.append(
                {
                    "rank": rank + 1,
                    "index": idx,
                    "source": str(metadata.get("source", "unknown")),
                    "page": metadata.get("page"),
                    "row": metadata.get("row"),
                    "score": round(float(scores[rank]) if rank < len(scores) else 0.0, 6),
                    "snippet": str(getattr(doc, "page_content", ""))[:400].strip(),
                }
            )
        return items

    def _doc_trace_item(self, doc: Any, *, rank: int, score: float, raw_score: float) -> Dict[str, Any]:
        metadata = getattr(doc, "metadata", {}) or {}
        return {
            "rank": rank,
            "source": str(metadata.get("source", "unknown")),
            "page": metadata.get("page"),
            "row": metadata.get("row"),
            "score": round(float(score), 6),
            "raw_score": round(float(raw_score), 6),
            "snippet": str(getattr(doc, "page_content", ""))[:400].strip(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        final_k: int = FINAL_K,
        candidate_k: int = CANDIDATE_K,
    ) -> RetrievalResult:
        """Run hybrid retrieval and optional reranking.

        Returns a RetrievalResult with both pre- and post-reranking data,
        enabling the Answer Justification layer in the UI.
        """
        t0 = time.perf_counter()

        if not self._docs:
            return RetrievalResult(
                final_docs=[], final_scores=[], confidence=0.0,
                candidates=[], candidate_scores=[], latency_ms=0.0,
                used_reranker=self.use_reranker,
            )

        dense_hits = self._dense_hits(query, candidate_k)
        dense_idx = [idx for idx, _ in dense_hits]
        dense_scores = [float(score) for _, score in dense_hits]
        sparse_idx, bm25_scores = self._sparse_indices(query, candidate_k)
        sparse_scores = [float(bm25_scores[i]) for i in sparse_idx if i < len(self._docs)]
        fusion_scores = self._rrf_score_map(dense_idx, sparse_idx)
        fused = sorted(fusion_scores.keys(), key=lambda i: fusion_scores[i], reverse=True)[:candidate_k]
        fused_scores = [float(fusion_scores[i]) for i in fused if i < len(self._docs)]
        candidates = [self._docs[i] for i in fused if i < len(self._docs)]
        cand_bm25 = [float(bm25_scores[i]) for i in fused if i < len(self._docs)]

        if not candidates:
            return RetrievalResult(
                final_docs=[], final_scores=[], confidence=0.0,
                candidates=[], candidate_scores=[], latency_ms=0.0,
                used_reranker=self.use_reranker,
                dense_topk=[], sparse_topk=[], fused_topk=[], reranked_topk=[],
            )

        if self.use_reranker:
            ce = _get_cross_encoder()
            pairs = [(query, doc.page_content) for doc in candidates]
            raw_scores: List[float] = ce.predict(pairs).tolist()
            # Convert logits to probability
            probs = [_sigmoid(s) for s in raw_scores]
            # Apply source-reputation weights before final ranking
            weighted_scores = self._apply_source_weights(candidates, probs)
            ranked = sorted(
                zip(candidates, weighted_scores, raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            final_docs = [d for d, _, _ in ranked[:final_k]]
            final_scores = [ws for _, ws, _ in ranked[:final_k]]  # weighted scores shown in UI
            confidence = _sigmoid(raw_scores[candidates.index(final_docs[0])]) if final_docs else 0.0
            reranked_topk = []
            for i, (doc, weighted_score, raw_score) in enumerate(ranked[:final_k]):
                reranked_topk.append(
                    self._doc_trace_item(
                        doc,
                        rank=i + 1,
                        score=weighted_score,
                        raw_score=raw_score,
                    )
                )
        else:
            weighted_fused = self._apply_source_weights(candidates, fused_scores)
            ranked_pairs = sorted(
                zip(candidates, weighted_fused, cand_bm25),
                key=lambda x: x[1],
                reverse=True,
            )
            final_docs = [d for d, _, _ in ranked_pairs[:final_k]]
            final_scores = [ws for _, ws, _ in ranked_pairs[:final_k]]
            top_bm25 = ranked_pairs[0][2] if ranked_pairs else 0.0
            confidence = min(0.30 + (top_bm25 / 15.0) * 0.60, 0.95)
            raw_scores = cand_bm25  # for candidate_scores below
            reranked_topk = []
            for i, (doc, weighted_score, raw_score) in enumerate(ranked_pairs[:final_k]):
                reranked_topk.append(
                    self._doc_trace_item(
                        doc,
                        rank=i + 1,
                        score=weighted_score,
                        raw_score=raw_score,
                    )
                )

        latency_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            final_docs=final_docs,
            final_scores=final_scores,
            confidence=confidence,
            candidates=candidates,
            candidate_scores=fused_scores,
            latency_ms=latency_ms,
            used_reranker=self.use_reranker,
            dense_topk=self._build_trace_items(dense_idx, dense_scores, limit=candidate_k),
            sparse_topk=self._build_trace_items(sparse_idx, sparse_scores, limit=candidate_k),
            fused_topk=self._build_trace_items(fused, fused_scores, limit=candidate_k),
            reranked_topk=reranked_topk,
        )


def confidence_label(score: float) -> str:
    """Human-readable confidence label for UI display."""
    if score >= CONF_HIGH:
        return "high"
    if score >= CONF_LOW:
        return "medium"
    return "low"


def build_pgvector_hybrid_retriever(
    *,
    organization: str,
    tenant_id: Optional[str],
    embeddings: Any,
    use_reranker: bool = USE_RERANKER,
    source_weights: Optional[Dict[str, float]] = None,
) -> HybridRetriever:
    """Build HybridRetriever backed by pgvector dense search and BM25 sparse search.

    Retrieval contract remains unchanged; only dense storage backend differs.
    """
    engine = require_engine()

    if tenant_id:
        tenant_where = "e.tenant_id = CAST(:tenant_id AS uuid)"
        tenant_params: Dict[str, Any] = {"tenant_id": tenant_id.strip()}
    else:
        # Eval and non-JWT contexts can still resolve tenant by name.
        tenant_where = "e.tenant_id = (SELECT id FROM tenants WHERE lower(name)=lower(:organization) LIMIT 1)"
        tenant_params = {"organization": organization.strip()}

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT e.id, e.chunk_text, e.metadata, e.tenant_id
                FROM embeddings e
                WHERE {tenant_where}
                """
            ),
            tenant_params,
        ).mappings().all()

    if ASSERT_TENANT_ISOLATION and tenant_id:
        expected_tenant = tenant_id.strip()
        for row in rows:
            row_tenant = str(row.get("tenant_id") or "").strip()
            assert row_tenant == expected_tenant, "tenant leakage detected in pgvector corpus preload"

    docs: List[Any] = []
    id_to_index: Dict[str, int] = {}
    for i, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append(Document(page_content=str(row["chunk_text"]), metadata=metadata))
        id_to_index[str(row["id"])] = i

    def _dense_indices_pgvector(query: str, k: int) -> List[Tuple[int, float]]:
        query_vec = _normalize_vector(embeddings.embed_query(query))
        qv = "[" + ",".join(f"{float(v):.8f}" for v in query_vec) + "]"
        with engine.connect() as conn:
            # Apply ANN probe count for each connection used by retrieval.
            conn.execute(text(f"SET ivfflat.probes = {max(PGVECTOR_PROBES, 1)}"))
            hit_rows = conn.execute(
                text(
                    f"""
                    SELECT e.id, e.tenant_id, (e.embedding <=> CAST(:query_vector AS vector)) AS distance
                    FROM embeddings e
                    WHERE {tenant_where}
                    ORDER BY e.embedding <=> CAST(:query_vector AS vector)
                    LIMIT :k
                    """
                ),
                {
                    **tenant_params,
                    "query_vector": qv,
                    "k": int(k),
                },
            ).mappings().all()
        if ASSERT_TENANT_ISOLATION and tenant_id:
            expected_tenant = tenant_id.strip()
            for row in hit_rows:
                row_tenant = str(row.get("tenant_id") or "").strip()
                assert row_tenant == expected_tenant, "tenant leakage detected in pgvector hit set"
        hits: List[Tuple[int, float]] = []
        for row in hit_rows:
            idx = id_to_index.get(str(row["id"]))
            if idx is not None:
                distance = float(row.get("distance") or 0.0)
                score = 1.0 / (1.0 + max(distance, 0.0))
                hits.append((idx, score))
        return hits

    return HybridRetriever(
        embeddings=embeddings,
        docs=docs,
        dense_indices_fn=_dense_indices_pgvector,
        use_reranker=use_reranker,
        source_weights=source_weights,
    )
