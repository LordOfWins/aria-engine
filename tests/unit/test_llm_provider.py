"""LLM Provider 단위 테스트"""

import pytest
from aria.core.config import AriaConfig
from aria.providers.llm_provider import LLMProvider, CostTracker, UsageRecord


class TestCostTracker:
    def test_add_record(self):
        tracker = CostTracker()
        record = UsageRecord(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=200.0,
        )
        tracker.add(record)
        assert tracker.daily_cost == 0.001
        assert tracker.monthly_cost == 0.001
        assert len(tracker.records) == 1

    def test_killswitch_daily_limit(self):
        config = AriaConfig()
        config.cost_control.daily_cost_limit_usd = 0.01

        tracker = CostTracker()
        record = UsageRecord(
            model="test", input_tokens=100, output_tokens=50,
            cost_usd=0.02, latency_ms=100.0,
        )
        tracker.add(record)

        allowed, reason = tracker.check_limits(config)
        assert not allowed
        assert "일일 비용 상한" in reason

    def test_killswitch_allows_within_limit(self):
        config = AriaConfig()
        config.cost_control.daily_cost_limit_usd = 10.0

        tracker = CostTracker()
        record = UsageRecord(
            model="test", input_tokens=100, output_tokens=50,
            cost_usd=0.001, latency_ms=100.0,
        )
        tracker.add(record)

        allowed, reason = tracker.check_limits(config)
        assert allowed
        assert reason == "OK"


class TestLLMProvider:
    def test_model_tiers(self):
        provider = LLMProvider()
        assert "default" in provider._model_tiers
        assert "cheap" in provider._model_tiers
        assert "heavy" in provider._model_tiers
        assert "fallback" in provider._model_tiers

    def test_cost_summary(self):
        provider = LLMProvider()
        summary = provider.get_cost_summary()
        assert "daily_cost_usd" in summary
        assert "monthly_cost_usd" in summary
        assert summary["total_requests"] == 0

    def test_fallback_model_default_differs_from_default_model(self):
        """config.py Field 기본값 검증: fallback_model ≠ default_model (다른 rate limit 풀)

        .env 설정과 무관하게 코드의 기본값이 올바른지 확인.
        실제 운영 환경에서는 .env의 ARIA_FALLBACK_MODEL 값도 변경 필요.
        """
        from aria.core.config import LLMConfig

        # .env/환경변수 무시 → Field default만 검증
        defaults = LLMConfig.model_fields
        default_model_default = defaults["default_model"].default
        fallback_model_default = defaults["fallback_model"].default
        assert default_model_default != fallback_model_default, (
            f"config.py Field 기본값: default({default_model_default})와 "
            f"fallback({fallback_model_default})이 동일하면 "
            "rate limit 공유로 fallback이 무의미합니다"
        )

    def test_max_retries_configured(self):
        """최대 재시도 횟수가 1보다 큰지 확인"""
        assert LLMProvider._MAX_RETRIES_PER_MODEL >= 2


class TestGetRetryAfter:
    """_get_retry_after 헤더 파싱 테스트"""

    def test_with_retry_after_header(self):
        """Retry-After 헤더가 있으면 해당 값 반환"""
        from unittest.mock import MagicMock
        import httpx

        error = MagicMock()
        error.response = httpx.Response(
            status_code=429,
            headers={"retry-after": "30"},
            request=httpx.Request(method="POST", url="https://api.example.com"),
        )
        result = LLMProvider._get_retry_after(error)
        assert result == 30.0

    def test_without_retry_after_header(self):
        """Retry-After 헤더가 없으면 0.0 반환"""
        from unittest.mock import MagicMock
        import httpx

        error = MagicMock()
        error.response = httpx.Response(
            status_code=429,
            headers={},
            request=httpx.Request(method="POST", url="https://api.example.com"),
        )
        result = LLMProvider._get_retry_after(error)
        assert result == 0.0

    def test_without_response(self):
        """response 없으면 0.0 반환"""
        from unittest.mock import MagicMock

        error = MagicMock(spec=[])  # response 속성 없음
        result = LLMProvider._get_retry_after(error)
        assert result == 0.0

    def test_invalid_retry_after_value(self):
        """Retry-After 값이 숫자가 아니면 0.0 반환"""
        from unittest.mock import MagicMock
        import httpx

        error = MagicMock()
        error.response = httpx.Response(
            status_code=429,
            headers={"retry-after": "invalid"},
            request=httpx.Request(method="POST", url="https://api.example.com"),
        )
        result = LLMProvider._get_retry_after(error)
        assert result == 0.0


class TestRetryLogic:
    """재시도 로직 통합 검증"""

    @pytest.mark.asyncio
    async def test_rate_limit_retries_before_fallback(self):
        """RateLimitError 시 같은 모델에서 최대 3회 재시도 후 fallback"""
        from unittest.mock import AsyncMock, MagicMock, patch
        import litellm
        from aria.core.config import AriaConfig, LLMConfig

        # default ≠ fallback 보장하는 config 직접 생성 (.env/lru_cache 의존 제거)
        config = AriaConfig()
        config.llm = LLMConfig(
            default_model="claude-sonnet-4-20250514",
            fallback_model="claude-haiku-4-5-20251001",
            cheap_model="claude-haiku-4-5-20251001",
        )
        config.anthropic_api_key = "test-key"  # noqa: S105 — 테스트용 더미
        provider = LLMProvider(config=config)

        call_count = 0
        called_models: list[str] = []

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            called_models.append(kwargs["model"])
            if call_count <= 4:  # 첫 모델 4회(1+3) 모두 실패
                raise litellm.RateLimitError(
                    message="rate limit exceeded",
                    llm_provider="anthropic",
                    model=kwargs["model"],
                )
            # 5번째 호출 (fallback 모델) 성공
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "success"
            mock_response.choices[0].message.tool_calls = None
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_response.usage.cache_read_input_tokens = 0
            mock_response.model = kwargs["model"]
            return mock_response

        with patch("litellm.acompletion", side_effect=mock_acompletion), \
             patch.object(provider, "_record_usage", return_value=MagicMock()), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            result = await provider._call_llm_with_fallback(
                {"messages": [{"role": "user", "content": "test"}], "temperature": 0.7, "max_tokens": 100},
                model_tier="default",
            )
            assert result is not None
            # 첫 모델(Sonnet) 4회(1+3 retries) + fallback(Haiku) 1회 = 5회
            assert call_count == 5
            # fallback에서 다른 모델 사용 확인
            assert called_models[-1] == "claude-haiku-4-5-20251001"
            assert called_models[0] == "claude-sonnet-4-20250514"
