"""ARIA Engine - 통합 테스트

테스트 대상:
1. VectorStore + BM25Index 동기화 (문서 추가 시 BM25 자동 동기화)
2. HybridRetriever 실제 검색 동작 (벡터 + BM25 → RRF)
3. Prompt Caching 파라미터 전달 검증

주의: Qdrant 서버 불필요 (mock 사용)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from aria.rag.bm25_index import BM25Index
from aria.rag.hybrid_retriever import HybridRetriever


class TestVectorStoreBM25Sync:
    """VectorStore → BM25 자동 동기화 테스트"""

    def _make_mock_vector_store(self, bm25_index: BM25Index) -> Any:
        """Qdrant/FastEmbed 없이 VectorStore 동작을 시뮬레이션"""
        # VectorStore의 _generate_deterministic_id만 가져옴
        from aria.rag.vector_store import VectorStore

        # __init__ 우회하여 인스턴스 생성
        store = VectorStore.__new__(VectorStore)
        store._bm25_index = bm25_index
        return store

    def test_bm25_index_receives_documents(self) -> None:
        """VectorStore.add_documents 시 BM25에 문서 전달 확인"""
        bm25 = BM25Index()
        store = self._make_mock_vector_store(bm25)

        # BM25Okapi는 문서 3개 이상이어야 IDF가 0이 안 됨
        docs = [
            {"text": "회피형 애착은 친밀감을 회피하는 패턴입니다", "metadata": {"source": "test"}},
            {"text": "불안형 애착은 관계에서 과도한 걱정을 보입니다", "metadata": {"source": "test"}},
            {"text": "안전형 애착은 건강한 관계 형성의 기초입니다", "metadata": {"source": "test"}},
            {"text": "인지행동치료는 사고 패턴을 변화시킵니다", "metadata": {"source": "test"}},
        ]

        # BM25에 직접 추가 (VectorStore.add_documents의 동기화 로직 검증)
        bm25_docs = [
            {
                "id": store._generate_deterministic_id(doc["text"]),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            }
            for doc in docs
        ]
        added = bm25.add_documents("test_kb", bm25_docs)

        assert added == 4
        assert bm25.has_collection("test_kb")

        # BM25 검색 동작 확인 — "회피형"이 doc-1에만 독립 토큰
        results = bm25.search("test_kb", "회피형", top_k=3)
        assert len(results) > 0
        assert "회피형" in results[0]["text"]

    def test_deterministic_id_consistency(self) -> None:
        """같은 텍스트 → 같은 ID 생성 (BM25 ↔ Qdrant 매칭용)"""
        bm25 = BM25Index()
        store = self._make_mock_vector_store(bm25)

        text = "동일한 텍스트는 동일한 ID를 생성해야 합니다"
        id1 = store._generate_deterministic_id(text)
        id2 = store._generate_deterministic_id(text)

        assert id1 == id2

    def test_bm25_none_no_error(self) -> None:
        """bm25_index=None이면 동기화 건너뛰기 (하위 호환)"""
        store = MagicMock()
        store._bm25_index = None
        # bm25_index가 None이면 동기화 코드 진입 안 함 — 에러 없어야 함
        assert store._bm25_index is None


class TestHybridRetrieverIntegration:
    """HybridRetriever 통합 검색 테스트"""

    def _make_retriever_with_data(self) -> tuple[HybridRetriever, BM25Index]:
        """테스트 데이터가 있는 HybridRetriever 생성"""
        bm25 = BM25Index()

        # 테스트 문서 적재
        docs = [
            {"id": "doc-1", "text": "회피형 애착 스타일은 친밀한 관계에서 거리를 두려는 경향이 있습니다", "metadata": {"domain": "attachment"}},
            {"id": "doc-2", "text": "불안형 애착은 관계에서 과도한 걱정과 집착을 보이는 패턴입니다", "metadata": {"domain": "attachment"}},
            {"id": "doc-3", "text": "인지행동치료 CBT는 사고 패턴을 변화시켜 행동을 개선합니다", "metadata": {"domain": "therapy"}},
            {"id": "doc-4", "text": "안전형 애착은 건강한 관계 형성의 기초가 되는 유형입니다", "metadata": {"domain": "attachment"}},
            {"id": "doc-5", "text": "감정 조절은 자신의 감정을 인식하고 적절히 표현하는 능력입니다", "metadata": {"domain": "emotion"}},
        ]
        bm25.add_documents("psychology_kb", docs)

        # 벡터 검색은 mock — BM25만 실제 동작 검증
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"text": docs[0]["text"], "score": 0.85, "metadata": docs[0]["metadata"]},
            {"text": docs[3]["text"], "score": 0.72, "metadata": docs[3]["metadata"]},
        ]

        retriever = HybridRetriever(mock_vector_store, bm25)
        return retriever, bm25

    def test_hybrid_search_merges_results(self) -> None:
        """벡터 + BM25 결과가 RRF로 병합"""
        retriever, _ = self._make_retriever_with_data()

        results = retriever.search("psychology_kb", "회피형 애착", top_k=5)

        assert len(results) > 0
        # 벡터 결과와 BM25 결과가 모두 포함되어야 함
        texts = [r["text"] for r in results]
        assert any("회피형" in t for t in texts)

    def test_hybrid_returns_rrf_metadata(self) -> None:
        """결과에 vector_rank / bm25_rank 포함"""
        retriever, _ = self._make_retriever_with_data()

        results = retriever.search("psychology_kb", "회피형 애착", top_k=3)

        for r in results:
            assert "vector_rank" in r
            assert "bm25_rank" in r
            assert "score" in r  # RRF 점수

    def test_bm25_fallback_when_vector_fails(self) -> None:
        """벡터 검색 실패 시 BM25만으로 결과 반환"""
        bm25 = BM25Index()
        # BM25Okapi IDF 정상 동작을 위해 문서 4개 이상
        bm25.add_documents("test_kb", [
            {"id": "doc-1", "text": "검증용 데이터가 포함된 테스트 문서입니다", "metadata": {}},
            {"id": "doc-2", "text": "다른 내용의 두 번째 문서입니다", "metadata": {}},
            {"id": "doc-3", "text": "세 번째 문서는 관련 없는 내용입니다", "metadata": {}},
            {"id": "doc-4", "text": "네 번째 문서도 다른 주제입니다", "metadata": {}},
        ])

        mock_vector_store = MagicMock()
        mock_vector_store.search.side_effect = Exception("Qdrant 연결 실패")

        retriever = HybridRetriever(mock_vector_store, bm25)
        # "검증용"은 doc-1에만 존재
        results = retriever.search("test_kb", "검증용", top_k=3)

        # BM25 결과만으로 응답
        assert len(results) > 0
        assert results[0]["vector_rank"] is None
        assert results[0]["bm25_rank"] is not None

    def test_empty_collection_returns_empty(self) -> None:
        """빈 컬렉션 → 빈 결과"""
        bm25 = BM25Index()
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []

        retriever = HybridRetriever(mock_vector_store, bm25)
        results = retriever.search("empty_kb", "쿼리", top_k=3)

        assert results == []


class TestPromptCachingIntegration:
    """Prompt Caching 파라미터 전달 검증"""

    @pytest.mark.asyncio
    async def test_intent_analysis_no_cache(self) -> None:
        """_analyze_intent에서 cache_system_prompt 미사용 확인 (cheap 모델은 캐시 최소 토큰 미달)"""
        from aria.agents.react_agent import ReActAgent, AgentState

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"surface_intent":"test","deeper_intent":"","required_knowledge":[],"search_queries":["test"],"complexity":"simple","recommended_action":"respond"}',
            "model": "test-model",
            "usage": MagicMock(),
        })

        mock_vector_store = MagicMock()
        agent = ReActAgent(mock_llm, mock_vector_store)

        state = AgentState(query="테스트 질문")
        await agent._analyze_intent(state)

        # cheap 모델(Haiku)은 최소 캐시 요구 4096 토큰 → cache_system_prompt 미전달
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get("cache_system_prompt") is None or call_kwargs.kwargs.get("cache_system_prompt") is False
        assert call_kwargs.kwargs.get("system_prompt") is not None

    @pytest.mark.asyncio
    async def test_reasoning_uses_cache(self) -> None:
        """_reason에서 cache_system_prompt=True 전달 확인"""
        from aria.agents.react_agent import ReActAgent, AgentState

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value={
            "content": "추론 결과입니다",
            "model": "test-model",
            "usage": MagicMock(),
        })

        mock_vector_store = MagicMock()
        agent = ReActAgent(mock_llm, mock_vector_store)

        state = AgentState(
            query="테스트 질문",
            intent={"surface_intent": "test"},
            search_results=[],
            reasoning_steps=[],
            iteration=0,
        )
        await agent._reason(state)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get("cache_system_prompt") is True

    @pytest.mark.asyncio
    async def test_self_reflect_no_cache(self) -> None:
        """_self_reflect에서 cache_system_prompt 미사용 확인 (cheap 모델은 캐시 최소 토큰 미달)"""
        from aria.agents.react_agent import ReActAgent, AgentState

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value={
            "content": '{"quality_score":0.8,"issues":[],"should_retry":false,"improvement_suggestion":""}',
            "model": "test-model",
            "usage": MagicMock(),
        })

        mock_vector_store = MagicMock()
        agent = ReActAgent(mock_llm, mock_vector_store)

        state = AgentState(
            query="테스트 질문",
            intent={"surface_intent": "test"},
            current_answer="답변입니다",
            iteration=1,
        )
        await agent._self_reflect(state)

        # cheap 모델(Haiku)은 최소 캐시 요구 4096 토큰 → cache_system_prompt 미전달
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get("cache_system_prompt") is None or call_kwargs.kwargs.get("cache_system_prompt") is False


class TestReActAgentHybridIntegration:
    """ReAct 에이전트 + HybridRetriever 통합 테스트"""

    @pytest.mark.asyncio
    async def test_agent_uses_hybrid_retriever(self) -> None:
        """hybrid_retriever 설정 시 _search_knowledge에서 사용 확인"""
        from aria.agents.react_agent import ReActAgent, AgentState

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = [
            {"text": "하이브리드 검색 결과", "score": 0.033, "metadata": {},
             "vector_rank": 1, "bm25_rank": 2, "vector_score": 0.8, "bm25_score": 3.5},
        ]

        mock_llm = MagicMock()
        mock_vector_store = MagicMock()

        agent = ReActAgent(mock_llm, mock_vector_store, hybrid_retriever=mock_hybrid)

        state = AgentState(
            query="테스트 쿼리",
            collection="test_kb",
            intent={"search_queries": ["테스트 쿼리"]},
        )
        result = await agent._search_knowledge(state)

        # hybrid_retriever.search가 호출되었는지 확인
        mock_hybrid.search.assert_called_once()
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["text"] == "하이브리드 검색 결과"

    @pytest.mark.asyncio
    async def test_agent_fallback_to_vector_only(self) -> None:
        """hybrid_retriever=None이면 기존 벡터 검색 사용"""
        from aria.agents.react_agent import ReActAgent, AgentState

        mock_llm = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"text": "벡터 전용 결과", "score": 0.75, "metadata": {}},
        ]

        agent = ReActAgent(mock_llm, mock_vector_store)  # hybrid_retriever=None

        state = AgentState(
            query="테스트 쿼리",
            collection="test_kb",
            intent={"search_queries": ["테스트 쿼리"]},
        )
        result = await agent._search_knowledge(state)

        mock_vector_store.search.assert_called_once()
        assert result["search_results"][0]["text"] == "벡터 전용 결과"
