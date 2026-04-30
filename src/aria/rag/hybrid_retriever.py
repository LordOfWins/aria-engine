"""ARIA Engine - Hybrid Retriever

벡터 시맨틱 검색 + BM25 키워드 검색 병합
- Reciprocal Rank Fusion (RRF) 기반 스코어 통합
- 벡터 검색: 의미적 유사도 (임베딩 기반)
- BM25 검색: 키워드 매칭 (토큰 기반)
- 두 검색의 장점을 결합하여 검색 품질 향상

RRF 공식: score(d) = Σ 1 / (k + rank_i(d))
- k: 상수 (기본 60 — MS 논문 권장)
- rank_i(d): i번째 검색 시스템에서 문서 d의 순위

참고: "Reciprocal Rank Fusion outperforms Condorcet and individual
       Rank Learning Methods" (Cormack et al., 2009)
"""

from __future__ import annotations

from typing import Any

import structlog

from aria.rag.bm25_index import BM25Index
from aria.rag.vector_store import VectorStore

logger = structlog.get_logger()

# RRF 상수 (k 값이 클수록 순위 차이에 둔감)
_RRF_K = 60


class HybridRetriever:
    """벡터 + BM25 하이브리드 검색기

    사용법:
        retriever = HybridRetriever(vector_store, bm25_index)
        results = retriever.search("psychology_kb", "회피형 애착 패턴", top_k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        *,
        rrf_k: int = _RRF_K,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        """
        Args:
            vector_store: Qdrant 벡터 저장소
            bm25_index: BM25 인메모리 인덱스
            rrf_k: RRF 상수 (기본 60)
            vector_weight: 벡터 검색 가중치 (기본 1.0)
            bm25_weight: BM25 검색 가중치 (기본 1.0)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 5,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """하이브리드 검색 (벡터 + BM25 → RRF 병합)

        Args:
            collection_name: 컬렉션 이름
            query: 검색 쿼리
            top_k: 최종 반환 결과 수
            vector_top_k: 벡터 검색 후보 수
            bm25_top_k: BM25 검색 후보 수
            score_threshold: 벡터 검색 최소 유사도

        Returns:
            [{"text": str, "score": float, "metadata": dict,
              "vector_rank": int|None, "bm25_rank": int|None}, ...]
            score: RRF 통합 점수
        """
        # 1. 벡터 시맨틱 검색
        vector_results = self._search_vector(
            collection_name, query,
            top_k=vector_top_k,
            score_threshold=score_threshold,
        )

        # 2. BM25 키워드 검색
        bm25_results = self._search_bm25(
            collection_name, query,
            top_k=bm25_top_k,
        )

        # 3. RRF 병합
        merged = self._rrf_merge(vector_results, bm25_results)

        # 4. top_k 잘라서 반환
        final = merged[:top_k]

        logger.info(
            "hybrid_search_completed",
            collection=collection_name,
            query=query[:50],
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            merged_hits=len(merged),
            returned=len(final),
        )

        return final

    def _search_vector(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """벡터 검색 (에러 시 빈 결과 반환 — BM25만으로 대체)"""
        try:
            return self.vector_store.search(
                collection_name=collection_name,
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.warning(
                "hybrid_vector_search_failed",
                collection=collection_name,
                error=str(e)[:200],
            )
            return []

    def _search_bm25(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """BM25 검색 (인덱스 없으면 빈 결과)"""
        if not self.bm25_index.has_collection(collection_name):
            logger.debug(
                "bm25_index_not_available",
                collection=collection_name,
            )
            return []

        return self.bm25_index.search(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
        )

    def _rrf_merge(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion 병합

        두 검색 결과를 문서 텍스트 기준으로 매칭하여
        RRF 점수로 재정렬합니다

        Returns:
            RRF 점수 내림차순 정렬된 결과
        """
        # 문서 텍스트 → 통합 데이터 맵
        doc_map: dict[str, dict[str, Any]] = {}

        # 벡터 결과 등록 (순위 = 1부터)
        for rank, result in enumerate(vector_results, start=1):
            text_key = result["text"][:200]  # 텍스트 앞 200자로 키 생성
            if text_key not in doc_map:
                doc_map[text_key] = {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            doc_map[text_key]["vector_rank"] = rank
            doc_map[text_key]["vector_score"] = result.get("score", 0.0)

        # BM25 결과 등록
        for rank, result in enumerate(bm25_results, start=1):
            text_key = result["text"][:200]
            if text_key not in doc_map:
                doc_map[text_key] = {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            doc_map[text_key]["bm25_rank"] = rank
            doc_map[text_key]["bm25_score"] = result.get("score", 0.0)

        # RRF 점수 계산
        for doc in doc_map.values():
            rrf = 0.0
            if doc["vector_rank"] is not None:
                rrf += self.vector_weight * (1.0 / (self.rrf_k + doc["vector_rank"]))
            if doc["bm25_rank"] is not None:
                rrf += self.bm25_weight * (1.0 / (self.rrf_k + doc["bm25_rank"]))
            doc["rrf_score"] = rrf

        # RRF 점수 내림차순 정렬
        sorted_docs = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 출력 포맷 정리
        return [
            {
                "text": doc["text"],
                "score": doc["rrf_score"],
                "metadata": doc["metadata"],
                "vector_rank": doc["vector_rank"],
                "bm25_rank": doc["bm25_rank"],
                "vector_score": doc["vector_score"],
                "bm25_score": doc["bm25_score"],
            }
            for doc in sorted_docs
        ]
