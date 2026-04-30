"""ARIA Engine - Hybrid Retrieval 단위 테스트

테스트 대상:
1. tokenize_korean: 한국어 토크나이저
2. BM25Index: 인메모리 BM25 인덱스
3. HybridRetriever._rrf_merge: RRF 병합 로직
"""

from __future__ import annotations

import pytest

from aria.rag.bm25_index import BM25Index, tokenize_korean
from aria.rag.hybrid_retriever import HybridRetriever


# === tokenize_korean 테스트 ===

class TestTokenizeKorean:
    """한국어 토크나이저 테스트"""

    def test_basic_korean(self) -> None:
        """기본 한국어 문장 토큰화"""
        tokens = tokenize_korean("회피형 애착 스타일은 거리두기를 선호합니다")
        assert "회피형" in tokens
        assert "애착" in tokens
        assert "스타일은" in tokens  # 형태소 분석 없으므로 조사 포함
        assert len(tokens) > 0

    def test_stopword_removal(self) -> None:
        """불용어 제거 확인"""
        tokens = tokenize_korean("그리고 이 것은 그 하지만 또는")
        # "것은"은 "것" + 조사 "은"이 붙어서 불용어로 안 잡힘 (형태소 분석 미적용)
        # 불용어에 정확히 매칭되는 "그리고" "하지만" "또는"은 제거
        # 1자 토큰 "이" "그"는 최소 길이 필터로 제거
        assert "그리고" not in tokens
        assert "하지만" not in tokens
        assert "또는" not in tokens
        assert "것은" in tokens  # 조사 붙은 형태 — 불용어 미매칭 정상

    def test_english_text(self) -> None:
        """영문 토큰화"""
        tokens = tokenize_korean("Attachment Theory by John Bowlby")
        assert "attachment" in tokens
        assert "theory" in tokens
        assert "john" in tokens
        # 불용어 "by" 제거됨
        assert "by" not in tokens

    def test_mixed_language(self) -> None:
        """한영 혼합 텍스트"""
        tokens = tokenize_korean("회피형 attachment style 분석")
        assert "회피형" in tokens
        assert "attachment" in tokens
        assert "style" in tokens
        assert "분석" in tokens

    def test_special_characters_removed(self) -> None:
        """특수문자 제거 확인"""
        tokens = tokenize_korean("회피형(avoidant) — 애착 #패턴!")
        # 특수문자가 공백으로 치환되어 토큰에 포함되지 않음
        for token in tokens:
            assert "#" not in token
            assert "!" not in token
            assert "(" not in token

    def test_empty_input(self) -> None:
        """빈 입력"""
        assert tokenize_korean("") == []
        assert tokenize_korean("   ") == []

    def test_min_token_length(self) -> None:
        """최소 토큰 길이 필터 (2자 미만 제거)"""
        tokens = tokenize_korean("a b c 테 스 트")
        # 1자 토큰 모두 제거됨
        assert len(tokens) == 0

    def test_unicode_normalization(self) -> None:
        """유니코드 NFC 정규화"""
        # 결합형 vs 완성형 한글
        tokens1 = tokenize_korean("가나다")
        tokens2 = tokenize_korean("가나다")  # NFC 정규화 후 동일해야 함
        assert tokens1 == tokens2


# === BM25Index 테스트 ===

class TestBM25Index:
    """BM25 인메모리 인덱스 테스트"""

    def _sample_docs(self) -> list[dict]:
        return [
            {
                "id": "doc-1",
                "text": "회피형 애착 스타일은 친밀한 관계에서 거리를 두려는 경향이 있습니다",
                "metadata": {"source": "psychology"},
            },
            {
                "id": "doc-2",
                "text": "불안형 애착은 관계에서 과도한 걱정과 집착을 보이는 패턴입니다",
                "metadata": {"source": "psychology"},
            },
            {
                "id": "doc-3",
                "text": "안전형 애착은 건강한 관계 형성의 기초가 되는 애착 유형입니다",
                "metadata": {"source": "psychology"},
            },
            {
                "id": "doc-4",
                "text": "인지행동치료는 사고 패턴을 변화시켜 행동을 개선하는 심리치료 기법입니다",
                "metadata": {"source": "therapy"},
            },
        ]

    def test_add_and_search(self) -> None:
        """문서 추가 후 검색"""
        index = BM25Index()
        docs = self._sample_docs()
        added = index.add_documents("test_kb", docs)

        assert added == 4
        assert index.has_collection("test_kb")

        results = index.search("test_kb", "회피형 애착", top_k=3)
        assert len(results) > 0
        # "회피형 애착" 키워드가 포함된 doc-1이 상위에 있어야 함
        assert "회피형" in results[0]["text"]

    def test_search_empty_collection(self) -> None:
        """빈 컬렉션 검색 → 빈 결과"""
        index = BM25Index()
        results = index.search("nonexistent", "테스트")
        assert results == []

    def test_search_no_match(self) -> None:
        """매칭 없는 쿼리"""
        index = BM25Index()
        index.add_documents("test_kb", self._sample_docs())
        results = index.search("test_kb", "블록체인 암호화폐 비트코인")
        # 관련 없는 쿼리 → 결과 없거나 매우 적음
        assert len(results) == 0

    def test_duplicate_document_update(self) -> None:
        """중복 문서 추가 시 업데이트"""
        index = BM25Index()
        index.add_documents("test_kb", [
            {"id": "doc-1", "text": "원본 텍스트", "metadata": {}},
        ])
        index.add_documents("test_kb", [
            {"id": "doc-1", "text": "수정된 텍스트", "metadata": {}},
        ])

        stats = index.get_collection_stats("test_kb")
        assert stats["document_count"] == 1

    def test_rebuild(self) -> None:
        """전체 재빌드"""
        index = BM25Index()
        index.add_documents("test_kb", self._sample_docs())
        assert index.get_collection_stats("test_kb")["document_count"] == 4

        # 2개 문서로 재빌드
        new_docs = self._sample_docs()[:2]
        rebuilt = index.rebuild("test_kb", new_docs)
        assert rebuilt == 2
        assert index.get_collection_stats("test_kb")["document_count"] == 2

    def test_remove_collection(self) -> None:
        """컬렉션 제거"""
        index = BM25Index()
        index.add_documents("test_kb", self._sample_docs())
        assert index.has_collection("test_kb")

        index.remove_collection("test_kb")
        assert not index.has_collection("test_kb")

    def test_empty_documents_skipped(self) -> None:
        """빈 텍스트 / 빈 ID 문서 건너뛰기"""
        index = BM25Index()
        added = index.add_documents("test_kb", [
            {"id": "", "text": "텍스트", "metadata": {}},
            {"id": "doc-1", "text": "", "metadata": {}},
            {"id": "doc-2", "text": "정상 문서 텍스트입니다", "metadata": {}},
        ])
        assert added == 1  # 유효한 문서 1개만

    def test_korean_keyword_matching(self) -> None:
        """한국어 키워드 매칭 품질 확인

        공백 토크나이저 한계: "인지행동치료" vs "인지행동치료는" → 별도 토큰
        doc-1에만 "인지행동치료" 단독 토큰 존재 → 1개 매칭 정상
        형태소 분석기 도입 시 개선 가능
        """
        index = BM25Index()
        index.add_documents("test_kb", [
            {"id": "doc-1", "text": "인지행동치료 CBT는 우울증 치료에 효과적입니다", "metadata": {}},
            {"id": "doc-2", "text": "정신분석 치료는 무의식적 갈등을 탐구합니다", "metadata": {}},
            {"id": "doc-3", "text": "인지행동치료는 불안장애에도 널리 사용됩니다", "metadata": {}},
        ])

        results = index.search("test_kb", "인지행동치료", top_k=3)
        assert len(results) >= 1
        # "인지행동치료" 단독 토큰이 있는 doc-1이 상위
        assert "인지행동치료" in results[0]["text"]


# === HybridRetriever RRF 병합 테스트 ===

class TestRRFMerge:
    """Reciprocal Rank Fusion 병합 로직 테스트"""

    def _make_retriever(self) -> HybridRetriever:
        """테스트용 retriever (vector_store/bm25_index는 mock 불필요 — _rrf_merge만 테스트)"""
        # _rrf_merge는 인스턴스 메서드이므로 dummy 객체 필요
        return HybridRetriever.__new__(HybridRetriever)

    def test_both_results_merged(self) -> None:
        """벡터 + BM25 결과가 모두 있을 때 RRF 병합"""
        retriever = self._make_retriever()
        retriever.rrf_k = 60
        retriever.vector_weight = 1.0
        retriever.bm25_weight = 1.0

        vector_results = [
            {"text": "문서 A는 벡터 1위", "score": 0.9, "metadata": {}},
            {"text": "문서 B는 벡터 2위", "score": 0.7, "metadata": {}},
        ]
        bm25_results = [
            {"text": "문서 B는 벡터 2위", "score": 5.2, "metadata": {}},  # BM25 1위
            {"text": "문서 C는 BM25만", "score": 3.1, "metadata": {}},
        ]

        merged = retriever._rrf_merge(vector_results, bm25_results)

        assert len(merged) == 3  # A + B + C

        # 문서 B는 벡터 2위 + BM25 1위 → 양쪽에 존재하므로 RRF 점수 최고
        texts = [m["text"] for m in merged]
        assert merged[0]["text"] == "문서 B는 벡터 2위"
        assert merged[0]["vector_rank"] == 2
        assert merged[0]["bm25_rank"] == 1

    def test_vector_only(self) -> None:
        """벡터 결과만 있을 때"""
        retriever = self._make_retriever()
        retriever.rrf_k = 60
        retriever.vector_weight = 1.0
        retriever.bm25_weight = 1.0

        vector_results = [
            {"text": "문서 A", "score": 0.9, "metadata": {}},
        ]
        merged = retriever._rrf_merge(vector_results, [])

        assert len(merged) == 1
        assert merged[0]["vector_rank"] == 1
        assert merged[0]["bm25_rank"] is None

    def test_bm25_only(self) -> None:
        """BM25 결과만 있을 때"""
        retriever = self._make_retriever()
        retriever.rrf_k = 60
        retriever.vector_weight = 1.0
        retriever.bm25_weight = 1.0

        bm25_results = [
            {"text": "문서 A", "score": 4.5, "metadata": {}},
        ]
        merged = retriever._rrf_merge([], bm25_results)

        assert len(merged) == 1
        assert merged[0]["vector_rank"] is None
        assert merged[0]["bm25_rank"] == 1

    def test_empty_both(self) -> None:
        """양쪽 모두 빈 결과"""
        retriever = self._make_retriever()
        retriever.rrf_k = 60
        retriever.vector_weight = 1.0
        retriever.bm25_weight = 1.0

        merged = retriever._rrf_merge([], [])
        assert merged == []

    def test_weight_affects_ranking(self) -> None:
        """가중치가 RRF 점수에 영향"""
        retriever = self._make_retriever()
        retriever.rrf_k = 60
        retriever.vector_weight = 2.0  # 벡터 가중치 2배
        retriever.bm25_weight = 1.0

        vector_results = [
            {"text": "벡터 전용 문서", "score": 0.9, "metadata": {}},
        ]
        bm25_results = [
            {"text": "BM25 전용 문서", "score": 5.0, "metadata": {}},
        ]

        merged = retriever._rrf_merge(vector_results, bm25_results)

        # 벡터 가중치가 2배이므로 벡터 전용 문서가 상위
        assert merged[0]["text"] == "벡터 전용 문서"
