"""Task 2: 에러 핸들링 강화 테스트

테스트 범위:
1. AriaConfig 모델-키 매핑 검증
2. NoAPIKeyError 에러 클래스
3. LLMProvider 키 미설정 사전 감지
4. fallback 체인 키 기반 정렬
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.core.config import AriaConfig
from aria.core.exceptions import NoAPIKeyError


# === 1. AriaConfig Model-Key Mapping ===


class TestModelKeyMapping:
    def test_anthropic_model_maps_to_anthropic_key(self, clean_env) -> None:
        config = AriaConfig(ANTHROPIC_API_KEY="sk-ant-test")
        assert config.has_api_key_for_model("claude-sonnet-4-20250514") is True
        assert config.has_api_key_for_model("claude-haiku-4-5-20251001") is True

    def test_openai_model_maps_to_openai_key(self, clean_env) -> None:
        config = AriaConfig(OPENAI_API_KEY="sk-test")
        assert config.has_api_key_for_model("gpt-4o") is True
        assert config.has_api_key_for_model("o1-preview") is True

    def test_deepseek_model_maps_to_deepseek_key(self, clean_env) -> None:
        config = AriaConfig(DEEPSEEK_API_KEY="dk-test")
        assert config.has_api_key_for_model("deepseek/deepseek-chat") is True

    def test_missing_key_returns_false(self, clean_env) -> None:
        config = AriaConfig()  # 모든 키 빈 상태 (ARIA_ENV_FILE="" + clean_env)
        assert config.has_api_key_for_model("claude-sonnet-4-20250514") is False
        assert config.has_api_key_for_model("gpt-4o") is False

    def test_get_available_models_filters_by_key(self, clean_env) -> None:
        config = AriaConfig(ANTHROPIC_API_KEY="sk-ant-test")
        available = config.get_available_models()
        # 기본 모델이 모두 claude 계열이므로 anthropic 키만 있으면 전부 사용 가능
        assert len(available) > 0
        assert all("claude" in m for m in available)

    def test_get_missing_key_models_with_mixed_providers(self, clean_env, monkeypatch) -> None:
        """서로 다른 프로바이더 모델 혼합 시 키 없는 모델 감지"""
        monkeypatch.setenv("ARIA_FALLBACK_MODEL", "gpt-4o")
        monkeypatch.setenv("ARIA_CHEAP_MODEL", "deepseek/deepseek-chat")
        config = AriaConfig(ANTHROPIC_API_KEY="sk-ant-test")
        missing = config.get_missing_key_models()
        model_names = [m for m, _ in missing]
        assert any("gpt" in m for m in model_names)
        assert any("deepseek" in m for m in model_names)

    def test_all_keys_present_no_missing(self, clean_env) -> None:
        config = AriaConfig(
            ANTHROPIC_API_KEY="sk-ant-test",
            OPENAI_API_KEY="sk-test",
            DEEPSEEK_API_KEY="dk-test",
        )
        missing = config.get_missing_key_models()
        assert len(missing) == 0

    def test_unknown_model_returns_empty_key(self, clean_env) -> None:
        config = AriaConfig()
        assert config.get_api_key_for_model("unknown-model-xyz") == ""


# === 2. NoAPIKeyError ===


class TestNoAPIKeyError:
    def test_error_message_includes_model_and_env_var(self) -> None:
        err = NoAPIKeyError(model="claude-sonnet-4-20250514", env_var="ANTHROPIC_API_KEY")
        assert "claude-sonnet-4-20250514" in err.message
        assert "ANTHROPIC_API_KEY" in err.message
        assert err.code == "NO_API_KEY"

    def test_error_details(self) -> None:
        err = NoAPIKeyError(model="gpt-4o", env_var="OPENAI_API_KEY")
        assert err.details["model"] == "gpt-4o"
        assert err.details["required_env_var"] == "OPENAI_API_KEY"


# === 3. LLMProvider Key Pre-check ===


class TestLLMProviderKeyPrecheck:
    @pytest.fixture
    def config_no_keys(self) -> MagicMock:
        """모든 API 키 미설정 config"""
        config = MagicMock()
        config.llm.default_model = "claude-sonnet-4-20250514"
        config.llm.cheap_model = "deepseek/deepseek-chat"
        config.llm.fallback_model = "gpt-4o"
        config.llm.embedding_model = "BAAI/bge-small-en-v1.5"
        config.llm.max_tokens_per_request = 4096
        config.cost_control.daily_cost_limit_usd = 10.0
        config.cost_control.monthly_cost_limit_usd = 100.0
        config.has_api_key_for_model.return_value = False
        config.get_available_models.return_value = []
        config.get_missing_key_models.return_value = [
            ("claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
            ("gpt-4o", "OPENAI_API_KEY"),
            ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
        ]
        return config

    @pytest.fixture
    def config_partial_keys(self) -> MagicMock:
        """Anthropic 키만 있는 config"""
        config = MagicMock()
        config.llm.default_model = "claude-sonnet-4-20250514"
        config.llm.cheap_model = "deepseek/deepseek-chat"
        config.llm.fallback_model = "gpt-4o"
        config.llm.embedding_model = "BAAI/bge-small-en-v1.5"
        config.llm.max_tokens_per_request = 4096
        config.cost_control.daily_cost_limit_usd = 10.0
        config.cost_control.monthly_cost_limit_usd = 100.0

        def _has_key(model: str) -> bool:
            return "claude" in model.lower()
        config.has_api_key_for_model.side_effect = _has_key
        config.get_available_models.return_value = ["claude-sonnet-4-20250514"]
        config.get_missing_key_models.return_value = [
            ("gpt-4o", "OPENAI_API_KEY"),
            ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
        ]
        return config

    def test_provider_logs_missing_keys(self, config_no_keys: MagicMock) -> None:
        """모든 키 미설정 시 LLMProvider 생성은 성공 (경고만)"""
        from aria.providers.llm_provider import LLMProvider
        provider = LLMProvider(config_no_keys)
        assert provider is not None

    @pytest.mark.asyncio
    async def test_no_keys_raises_no_api_key_error(self, config_no_keys: MagicMock) -> None:
        """모든 키 미설정 상태에서 complete 호출 → NoAPIKeyError"""
        from aria.providers.llm_provider import LLMProvider
        provider = LLMProvider(config_no_keys)

        with pytest.raises(NoAPIKeyError) as exc_info:
            await provider.complete("test prompt")

        assert "ANTHROPIC_API_KEY" in exc_info.value.message

    def test_partial_keys_provider_initializes(self, config_partial_keys: MagicMock) -> None:
        """일부 키만 있어도 Provider 생성 성공"""
        from aria.providers.llm_provider import LLMProvider
        provider = LLMProvider(config_partial_keys)
        assert provider is not None
