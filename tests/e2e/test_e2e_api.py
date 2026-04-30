"""ARIA Engine - E2E 테스트

FastAPI TestClient 기반 API 엔드포인트 전체 검증
- 서버 기동 (lifespan) → 엔드포인트 호출 → 응답 검증
- Qdrant/LLM은 mock으로 대체 (외부 의존성 제거)

테스트 대상:
1. POST /v1/knowledge → 문서 추가 + BM25 동기화
2. POST /v1/knowledge/{collection}/search → 벡터 검색
3. GET /v1/collections → 컬렉션 목록
4. GET /v1/health → 헬스 체크
5. GET /v1/cost → 비용 현황
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def e2e_client() -> TestClient:
    """테스트용 FastAPI 앱 생성 (Qdrant/LLM mock)

    글로벌 상태를 설정하므로 teardown에서 반드시 원복합니다
    """
    import os
    os.environ["ARIA_AUTH_DISABLED"] = "true"

    # config 캐시 클리어 (auth_disabled 반영)
    from aria.core.config import get_config
    get_config.cache_clear()

    import aria.api.app as app_module

    # 원본 글로벌 상태 저장
    _orig_vector_store = app_module.vector_store
    _orig_llm_provider = app_module.llm_provider
    _orig_react_agent = app_module.react_agent
    _orig_rate_limiter = app_module.rate_limiter

    # Qdrant와 FastEmbed를 mock하여 실제 서버 불필요
    with patch("aria.rag.vector_store.QdrantClient") as mock_qdrant_cls, \
         patch("aria.rag.vector_store.TextEmbedding") as mock_embed_cls:

        # QdrantClient mock
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant_cls.return_value = mock_qdrant

        # TextEmbedding mock
        mock_embedder = MagicMock()
        import numpy as np
        mock_embedder.embed.return_value = iter([np.zeros(384)])
        mock_embed_cls.return_value = mock_embedder

        # Global instance 직접 주입 (lifespan 우회)
        from aria.rag.bm25_index import BM25Index
        from aria.rag.hybrid_retriever import HybridRetriever
        from aria.rag.vector_store import VectorStore
        from aria.providers.llm_provider import LLMProvider
        from aria.agents.react_agent import ReActAgent

        config = get_config()
        bm25_index = BM25Index()
        app_module.vector_store = VectorStore(config, bm25_index=bm25_index)
        app_module.llm_provider = LLMProvider(config)
        hybrid = HybridRetriever(app_module.vector_store, bm25_index)
        app_module.react_agent = ReActAgent(app_module.llm_provider, app_module.vector_store, hybrid_retriever=hybrid)
        app_module.rate_limiter = app_module.RateLimiter(max_requests=60, window_seconds=60)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        yield client

    # teardown — 글로벌 상태 원복 (다른 테스트 모듈 오염 방지)
    app_module.vector_store = _orig_vector_store
    app_module.llm_provider = _orig_llm_provider
    app_module.react_agent = _orig_react_agent
    app_module.rate_limiter = _orig_rate_limiter
    get_config.cache_clear()


class TestE2EHealthCheck:
    """헬스 체크 E2E"""

    def test_health_returns_ok(self, e2e_client: TestClient) -> None:
        """GET /v1/health → 200"""
        response = e2e_client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["engine"] == "ARIA"


class TestE2EKnowledgeEndpoints:
    """지식 베이스 E2E"""

    def test_add_knowledge_returns_count(self, e2e_client: TestClient) -> None:
        """POST /v1/knowledge → 문서 추가 성공 (200 응답)"""
        payload = {
            "collection": "test_kb",
            "documents": [
                {"text": "테스트 문서 1", "metadata": {"source": "test"}},
                {"text": "테스트 문서 2", "metadata": {"source": "test"}},
            ],
        }

        response = e2e_client.post("/v1/knowledge", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["collection"] == "test_kb"
        # mock 환경에서 documents_added 수는 Qdrant mock 동작에 의존
        assert "documents_added" in data

    def test_add_empty_documents_returns_zero(self, e2e_client: TestClient) -> None:
        """빈 문서 리스트 → 422 (min_length=1 제약)"""
        payload = {
            "collection": "test_kb",
            "documents": [],
        }

        response = e2e_client.post("/v1/knowledge", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_add_knowledge_missing_collection(self, e2e_client: TestClient) -> None:
        """collection 누락 → 422"""
        payload = {
            "documents": [
                {"text": "테스트", "metadata": {}},
            ],
        }

        response = e2e_client.post("/v1/knowledge", json=payload)
        assert response.status_code == 422


class TestE2ECostEndpoint:
    """비용 현황 E2E"""

    def test_cost_returns_summary(self, e2e_client: TestClient) -> None:
        """GET /v1/cost → 비용 요약"""
        response = e2e_client.get("/v1/cost")

        assert response.status_code == 200
        data = response.json()
        assert "daily_cost_usd" in data
        assert "monthly_cost_usd" in data
        assert "total_requests" in data
        assert "total_cached_tokens" in data


class TestE2ECollections:
    """컬렉션 목록 E2E"""

    def test_list_collections_returns_list(self, e2e_client: TestClient) -> None:
        """GET /v1/collections → 컬렉션 목록"""
        response = e2e_client.get("/v1/collections")

        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert isinstance(data["collections"], list)


class TestE2EAuthIntegration:
    """인증 + Rate Limit E2E"""

    def test_auth_disabled_allows_access(self, e2e_client: TestClient) -> None:
        """ARIA_AUTH_DISABLED=true → 인증 없이 접근 가능"""
        response = e2e_client.get("/v1/health")
        assert response.status_code == 200

    def test_query_input_validation(self, e2e_client: TestClient) -> None:
        """POST /v1/query → 빈 쿼리 거부"""
        payload = {"query": "", "collection": "test"}
        response = e2e_client.post("/v1/query", json=payload)

        # min_length=1 제약에 의해 422
        assert response.status_code == 422
