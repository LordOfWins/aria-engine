"""ARIA Engine - BM25 In-Memory Index

키워드 기반 검색 (벡터 검색 보완용)
- rank-bm25 (BM25Okapi) 기반
- 경량 한국어 토크나이저 (공백 + 한글 문자 단위 + 불용어)
- 컬렉션별 인덱스 분리 관리
- 문서 추가/삭제 시 자동 재빌드

설계 결정:
- 인메모리 전용 (현재 문서 수 수십~수백 건 수준)
- 서버 재시작 시 Qdrant에서 재로딩 필요 → warm_up() 메서드 제공
- 한국어 형태소 분석기(mecab)는 추후 정확도 필요 시 교체
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

logger = structlog.get_logger()

# 한국어 불용어 (조사/어미/접속사 등 — 검색 노이즈 제거용)
_KOREAN_STOPWORDS: set[str] = {
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "부터", "까지", "에게", "한테", "께",
    "이다", "하다", "있다", "없다", "되다", "않다",
    "그", "그리고", "그러나", "하지만", "또는", "및", "등",
    "것", "수", "때", "중", "더", "잘", "매우", "아주",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "then", "so",
    "this", "that", "these", "those", "it", "its",
}

# 최소 토큰 길이 (1자 토큰은 노이즈가 많음)
_MIN_TOKEN_LENGTH = 2


def tokenize_korean(text: str) -> list[str]:
    """경량 한국어 + 영문 토크나이저

    처리 과정:
    1. 소문자 변환 + 정규화 (NFC)
    2. 특수문자 제거 (한글/영문/숫자만 유지)
    3. 공백 기준 분리
    4. 불용어 제거
    5. 최소 길이 필터

    향후 정확도 개선 시 mecab 토크나이저로 교체 가능
    """
    # NFC 정규화 → 소문자
    text = unicodedata.normalize("NFC", text.lower())

    # 한글/영문/숫자/공백만 유지
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣa-z0-9]", " ", text)

    # 공백 기준 분리 → 불용어 제거 → 최소 길이 필터
    tokens = [
        token
        for token in text.split()
        if token not in _KOREAN_STOPWORDS and len(token) >= _MIN_TOKEN_LENGTH
    ]

    return tokens


@dataclass
class _IndexEntry:
    """BM25 인덱스 내부 문서 엔트리"""
    doc_id: str          # Qdrant point ID (결정적 UUID)
    text: str            # 원본 텍스트
    tokens: list[str]    # 토큰화된 텍스트
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """컬렉션별 BM25 인메모리 인덱스

    사용법:
        index = BM25Index()
        index.add_documents("psychology_kb", [
            {"id": "abc-123", "text": "회피형 애착은...", "metadata": {"source": "dsm5"}}
        ])
        results = index.search("psychology_kb", "회피형 애착 패턴", top_k=5)

    재빌드:
        index.rebuild("psychology_kb", documents)  # 전체 교체
    """

    def __init__(self) -> None:
        # 컬렉션명 → 인덱스 데이터
        self._indices: dict[str, list[_IndexEntry]] = {}
        # 컬렉션명 → BM25 모델 (문서 추가 시 재빌드)
        self._models: dict[str, BM25Okapi | None] = {}

    def _rebuild_model(self, collection_name: str) -> None:
        """BM25 모델 재빌드 (문서 변경 시 호출)"""
        entries = self._indices.get(collection_name, [])
        if not entries:
            self._models[collection_name] = None
            return

        corpus = [entry.tokens for entry in entries]
        self._models[collection_name] = BM25Okapi(corpus)
        logger.debug(
            "bm25_model_rebuilt",
            collection=collection_name,
            doc_count=len(entries),
        )

    def add_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> int:
        """문서 추가 → BM25 인덱스 재빌드

        Args:
            collection_name: 컬렉션 이름
            documents: [{"id": str, "text": str, "metadata": dict}, ...]
                       id는 Qdrant point ID와 동일해야 함

        Returns:
            추가된 문서 수
        """
        if not documents:
            return 0

        if collection_name not in self._indices:
            self._indices[collection_name] = []

        # 기존 ID 세트 (중복 방지)
        existing_ids = {e.doc_id for e in self._indices[collection_name]}
        added = 0

        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")

            if not doc_id or not text:
                continue

            # 중복 문서는 업데이트 (기존 제거 후 재추가)
            if doc_id in existing_ids:
                self._indices[collection_name] = [
                    e for e in self._indices[collection_name] if e.doc_id != doc_id
                ]

            tokens = tokenize_korean(text)
            if not tokens:
                continue

            self._indices[collection_name].append(
                _IndexEntry(
                    doc_id=doc_id,
                    text=text,
                    tokens=tokens,
                    metadata=doc.get("metadata", {}),
                )
            )
            added += 1

        # 문서 변경 시 BM25 모델 재빌드
        if added > 0:
            self._rebuild_model(collection_name)
            logger.info(
                "bm25_documents_added",
                collection=collection_name,
                added=added,
                total=len(self._indices[collection_name]),
            )

        return added

    def rebuild(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> int:
        """컬렉션 인덱스 전체 교체 (warm_up / 재색인 용)

        Args:
            collection_name: 컬렉션 이름
            documents: [{"id": str, "text": str, "metadata": dict}, ...]

        Returns:
            인덱싱된 문서 수
        """
        self._indices[collection_name] = []
        self._models[collection_name] = None
        return self.add_documents(collection_name, documents)

    def search(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """BM25 키워드 검색

        Args:
            collection_name: 컬렉션 이름
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수

        Returns:
            [{"text": str, "score": float, "metadata": dict, "doc_id": str}, ...]
            score: BM25 스코어 (정규화되지 않은 원시 값)
        """
        model = self._models.get(collection_name)
        entries = self._indices.get(collection_name, [])

        if model is None or not entries:
            return []

        query_tokens = tokenize_korean(query)
        if not query_tokens:
            return []

        scores = model.get_scores(query_tokens)

        # 점수 > 0인 것만 필터 → 정렬 → top_k
        scored_entries = [
            (entries[i], float(scores[i]))
            for i in range(len(entries))
            if scores[i] > 0
        ]
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry, score in scored_entries[:top_k]:
            results.append({
                "text": entry.text,
                "score": score,
                "metadata": entry.metadata,
                "doc_id": entry.doc_id,
            })

        logger.debug(
            "bm25_search_completed",
            collection=collection_name,
            query=query[:50],
            results=len(results),
        )
        return results

    def has_collection(self, collection_name: str) -> bool:
        """컬렉션에 BM25 인덱스가 있는지 확인"""
        entries = self._indices.get(collection_name, [])
        return len(entries) > 0

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """컬렉션 BM25 인덱스 통계"""
        entries = self._indices.get(collection_name, [])
        return {
            "collection": collection_name,
            "document_count": len(entries),
            "has_model": self._models.get(collection_name) is not None,
        }

    def remove_collection(self, collection_name: str) -> None:
        """컬렉션 BM25 인덱스 제거"""
        self._indices.pop(collection_name, None)
        self._models.pop(collection_name, None)
        logger.info("bm25_collection_removed", collection=collection_name)
