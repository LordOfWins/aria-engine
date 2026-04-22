"""Configuration 단위 테스트"""

from aria.core.config import get_config, AriaConfig, Environment


class TestConfig:
    def test_default_config(self):
        config = AriaConfig()
        assert config.llm.default_model == "claude-sonnet-4-20250514"
        assert config.api.port == 8100
        assert config.api.env == Environment.DEVELOPMENT

    def test_cost_control_defaults(self):
        config = AriaConfig()
        assert config.cost_control.daily_cost_limit_usd == 10.0
        assert config.cost_control.monthly_cost_limit_usd == 300.0

    def test_qdrant_defaults(self):
        config = AriaConfig()
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
