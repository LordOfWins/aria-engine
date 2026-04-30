"""Task 1: 에러 핸들링 강화 테스트

테스트 범위:
1. 커스텀 예외 클래스 구조
2. LLM fallback 체인 동작
3. VectorStore 구조화된 에러
4. ReAct 에이전트 에러 전파
5. FastAPI 글로벌 에러 핸들러
6. Rate limiter
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.core.exceptions import (
    AriaError,
    KillSwitchError,
    LLMAllProvidersExhaustedError,
    LLMProviderError,
    NoAPIKeyError,
    CollectionNotFoundError,
    VectorStoreError,
    AgentError,
)


# === 1. Custom Exception Tests ===

class TestCustomExceptions:
    def test_aria_error_base(self) -> None:
        err = AriaError("test error", code="TEST", details={"key": "val"})
        assert err.message == "test error"
        assert err.code == "TEST"
        assert err.details == {"key": "val"}
        assert str(err) == "test error"

    def test_killswitch_error(self) -> None:
        err = KillSwitchError("일일 상한", daily_cost=5.0, monthly_cost=20.0)
        assert err.code == "KILLSWITCH_TRIGGERED"
        assert err.details["daily_cost_usd"] == 5.0
        assert err.details["monthly_cost_usd"] == 20.0

    def test_all_providers_exhausted(self) -> None:
        attempts = [
            {"model": "claude-sonnet-4-20250514", "error_type": "AuthenticationError", "error": "invalid key"},
            {"model": "gpt-4o-mini", "error_type": "RateLimitError", "error": "rate limited"},
        ]
        err = LLMAllProvidersExhaustedError(attempts)
        assert err.code == "ALL_PROVIDERS_EXHAUSTED"
        assert "claude-sonnet-4-20250514" in err.message
        assert "gpt-4o-mini" in err.message
        assert len(err.details["attempts"]) == 2

    def test_collection_not_found(self) -> None:
        err = CollectionNotFoundError("my_collection")
        assert err.code == "COLLECTION_NOT_FOUND"
        assert "my_collection" in err.message
        assert err.details["collection"] == "my_collection"

    def test_agent_error_truncates_query(self) -> None:
        long_query = "x" * 500
        err = AgentError("failed", query=long_query, iteration=2)
        assert len(err.details["query"]) == 200  # 200자 truncation
        assert err.details["iteration"] == 2

    def test_exception_hierarchy(self) -> None:
        """모든 ARIA 예외는 AriaError를 상속"""
        assert issubclass(KillSwitchError, AriaError)
        assert issubclass(LLMProviderError, AriaError)
        assert issubclass(LLMAllProvidersExhaustedError, LLMProviderError)
        assert issubclass(VectorStoreError, AriaError)
        assert issubclass(CollectionNotFoundError, VectorStoreError)
        assert issubclass(AgentError, AriaError)
        assert issubclass(NoAPIKeyError, AriaError)


# === 2. JSON Safe Parser Tests ===

class TestSafeParseJson:
    def setup_method(self) -> None:
        from aria.agents.react_agent import _safe_parse_json
        self.parse = _safe_parse_json

    def test_pure_json(self) -> None:
        result = self.parse('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_in_code_block(self) -> None:
        content = '```json\n{"key": "value"}\n```'
        result = self.parse(content)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self) -> None:
        content = '```\n{"key": "value"}\n```'
        result = self.parse(content)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        content = 'Here is the result:\n{"quality_score": 0.8, "should_retry": false}\nDone.'
        result = self.parse(content)
        assert result is not None
        assert result["quality_score"] == 0.8

    def test_invalid_json_returns_none(self) -> None:
        result = self.parse("This is not JSON at all")
        assert result is None

    def test_empty_string(self) -> None:
        result = self.parse("")
        assert result is None

    def test_nested_json(self) -> None:
        content = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = self.parse(content)
        assert result is not None
        assert result["outer"]["inner"] == [1, 2, 3]


# === 3. Rate Limiter Tests ===

class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        from aria.api.app import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed("client1") is True

    def test_blocks_over_limit(self) -> None:
        from aria.api.app import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            assert limiter.is_allowed("client1") is True

        assert limiter.is_allowed("client1") is False

    def test_different_clients_independent(self) -> None:
        from aria.api.app import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # client2는 별도
        assert limiter.is_allowed("client2") is True

    def test_window_expiration(self) -> None:
        from aria.api.app import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # 윈도우 만료 대기
        time.sleep(1.1)
        assert limiter.is_allowed("client1") is True


# === 4. LLM Provider Fallback Chain Tests ===

class TestLLMProviderFallback:
    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.llm.default_model = "claude-sonnet-4-20250514"
        config.llm.cheap_model = "gpt-4o-mini"
        config.llm.fallback_model = "deepseek/deepseek-chat"
        config.llm.embedding_model = "BAAI/bge-small-en-v1.5"
        config.llm.max_tokens_per_request = 4096
        config.cost_control.daily_cost_limit_usd = 10.0
        config.cost_control.monthly_cost_limit_usd = 100.0
        return config

    def test_fallback_chain_structure(self, mock_config: MagicMock) -> None:
        from aria.providers.llm_provider import LLMProvider
        provider = LLMProvider(mock_config)

        assert provider._fallback_chains["default"] == [
            "claude-sonnet-4-20250514",
            "deepseek/deepseek-chat",
        ]
        assert provider._fallback_chains["cheap"] == [
            "gpt-4o-mini",
            "deepseek/deepseek-chat",
        ]
        # fallback 티어는 자기 자신만
        assert provider._fallback_chains["fallback"] == [
            "deepseek/deepseek-chat",
        ]

    def test_killswitch_raises_custom_error(self, mock_config: MagicMock) -> None:
        from aria.providers.llm_provider import LLMProvider
        provider = LLMProvider(mock_config)

        # 비용 상한 초과 시뮬레이션
        provider.cost_tracker.daily_cost = 15.0
        provider.cost_tracker.current_date = "2026-04-22"

        with pytest.raises(KillSwitchError) as exc_info:
            provider._check_killswitch()

        assert exc_info.value.details["daily_cost_usd"] == 15.0


# === 5. FastAPI Error Handler Tests ===

class TestFastAPIErrorHandlers:
    """API 키 인증 활성화 상태에서의 에러 핸들러 테스트"""

    # 기본 개발 키 (config 기본값)
    _DEFAULT_KEY = "aria-dev-key-change-me"
    _AUTH_HEADER = {"X-API-Key": _DEFAULT_KEY}

    @pytest.fixture
    def client(self):  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient
        from aria.api.app import app
        return TestClient(app)

    def test_health_check(self, client) -> None:  # type: ignore[no-untyped-def]
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"

    def test_query_without_agent_returns_503(self, client) -> None:  # type: ignore[no-untyped-def]
        """에이전트 미초기화 상태에서 query 호출 → 503"""
        import aria.api.app as app_module
        original = app_module.react_agent
        app_module.react_agent = None
        try:
            response = client.post(
                "/v1/query",
                json={"query": "테스트 질문"},
                headers=self._AUTH_HEADER,
            )
            assert response.status_code == 503
        finally:
            app_module.react_agent = original

    def test_empty_query_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        """빈 쿼리 → Pydantic validation error"""
        response = client.post(
            "/v1/query",
            json={"query": ""},
            headers=self._AUTH_HEADER,
        )
        assert response.status_code == 422

    def test_search_without_vector_store_returns_503(self, client) -> None:  # type: ignore[no-untyped-def]
        import aria.api.app as app_module
        original = app_module.vector_store
        app_module.vector_store = None  # 명시적으로 None 설정
        try:
            response = client.post(
                "/v1/knowledge/test_col/search",
                json={"query": "테스트"},
                headers=self._AUTH_HEADER,
            )
            assert response.status_code == 503
        finally:
            app_module.vector_store = original
