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
