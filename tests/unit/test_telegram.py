"""ARIA Engine - Telegram Module 단위 테스트

테스트 범위:
- TelegramConfig: 설정 로딩 / 기본값
- ARIAClient: API 호출 / 에러 핸들링 / 타임아웃
- Notifier: 메시지 분할 / send_message / send_confirmation
- Handlers: 인증 / 명령어 / 메시지 처리
- Bot: create_bot 검증
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.core.config import TelegramConfig
from aria.telegram.client import ARIAClient
from aria.telegram.notifier import _split_message


# === TelegramConfig Tests ===


class TestTelegramConfig:
    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARIA_TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_ARIA_BASE_URL", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_ARIA_API_KEY", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_DEFAULT_SCOPE", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_DEFAULT_COLLECTION", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_REQUEST_TIMEOUT", raising=False)
        monkeypatch.setenv("ARIA_ENV_FILE", "")
        config = TelegramConfig()
        assert config.bot_token == ""
        assert config.chat_id == ""
        assert config.aria_base_url == "http://localhost:8100"
        assert config.request_timeout == 120
        assert config.default_scope == "global"
        assert config.default_collection == "default"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARIA_TELEGRAM_BOT_TOKEN", "test-token-123")
        monkeypatch.setenv("ARIA_TELEGRAM_CHAT_ID", "999888777")
        monkeypatch.setenv("ARIA_TELEGRAM_ARIA_BASE_URL", "http://aria:8100")
        config = TelegramConfig()
        assert config.bot_token == "test-token-123"
        assert config.chat_id == "999888777"
        assert config.aria_base_url == "http://aria:8100"


# === ARIAClient Tests ===


class TestARIAClient:
    def test_init_with_api_key(self) -> None:
        client = ARIAClient("http://localhost:8100", api_key="test-key")
        assert client.base_url == "http://localhost:8100"
        assert client._headers["X-API-Key"] == "test-key"

    def test_init_without_api_key(self) -> None:
        client = ARIAClient("http://localhost:8100")
        assert "X-API-Key" not in client._headers

    def test_trailing_slash_stripped(self) -> None:
        client = ARIAClient("http://localhost:8100/")
        assert client.base_url == "http://localhost:8100"

    @pytest.mark.asyncio
    async def test_query_success(self) -> None:
        """query() 성공 — httpx 목"""
        client = ARIAClient("http://localhost:8100", api_key="key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "테스트 답변",
            "confidence": 0.8,
            "tool_calls_made": 1,
        }

        with patch("aria.telegram.client.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await client.query("테스트 질문")

        assert result["answer"] == "테스트 답변"
        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_query_connection_error(self) -> None:
        """ARIA 서버 연결 실패"""
        import httpx as httpx_module
        client = ARIAClient("http://localhost:9999")

        with patch("aria.telegram.client.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=httpx_module.ConnectError("refused"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await client.query("test")

        assert result["error"] == "CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_query_timeout(self) -> None:
        """ARIA 서버 타임아웃"""
        import httpx as httpx_module
        client = ARIAClient("http://localhost:8100", timeout=5)

        with patch("aria.telegram.client.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(side_effect=httpx_module.TimeoutException("timeout"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await client.query("test")

        assert result["error"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_query_api_error(self) -> None:
        """ARIA API 에러 응답 (429 등)"""
        client = ARIAClient("http://localhost:8100", api_key="key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": "KILLSWITCH_TRIGGERED",
            "message": "비용 상한 초과",
        }

        with patch("aria.telegram.client.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await client.query("test")

        assert result["error"] == "KILLSWITCH_TRIGGERED"

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        client = ARIAClient("http://localhost:8100")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "0.2.0"}

        with patch("aria.telegram.client.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client_instance

            result = await client.health_check()

        assert result["status"] == "ok"


# === Notifier Tests ===


class TestMessageSplitting:
    def test_short_message_no_split(self) -> None:
        result = _split_message("짧은 메시지")
        assert len(result) == 1
        assert result[0] == "짧은 메시지"

    def test_long_message_split(self) -> None:
        # 5000자 메시지 → 2개로 분할 (한도 4096)
        long_msg = "x" * 5000
        result = _split_message(long_msg)
        assert len(result) == 2
        assert len(result[0]) <= 4096
        # 모든 문자 보존
        assert "".join(result) == long_msg

    def test_split_at_newline(self) -> None:
        # 줄바꿈 기준으로 분할
        lines = ["line " + str(i) for i in range(500)]
        long_msg = "\n".join(lines)
        result = _split_message(long_msg)
        for chunk in result:
            assert len(chunk) <= 4096

    def test_exact_limit(self) -> None:
        msg = "x" * 4096
        result = _split_message(msg)
        assert len(result) == 1


# === Handlers Tests ===


class TestHandlersAuth:
    def test_authorized_chat(self) -> None:
        from aria.telegram.handlers import ARIAHandlers
        client = MagicMock()
        handlers = ARIAHandlers(client, allowed_chat_id="12345")

        update = MagicMock()
        update.effective_chat.id = 12345

        assert handlers._is_authorized(update) is True

    def test_unauthorized_chat(self) -> None:
        from aria.telegram.handlers import ARIAHandlers
        client = MagicMock()
        handlers = ARIAHandlers(client, allowed_chat_id="12345")

        update = MagicMock()
        update.effective_chat.id = 99999

        assert handlers._is_authorized(update) is False

    def test_no_chat(self) -> None:
        from aria.telegram.handlers import ARIAHandlers
        client = MagicMock()
        handlers = ARIAHandlers(client, allowed_chat_id="12345")

        update = MagicMock()
        update.effective_chat = None

        assert handlers._is_authorized(update) is False


class TestHandlersMessage:
    @pytest.mark.asyncio
    async def test_handle_message_authorized(self) -> None:
        """인증된 메시지 → ARIA 호출 → 응답"""
        from aria.telegram.handlers import ARIAHandlers

        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value={
            "answer": "ARIA 답변입니다",
            "confidence": 0.85,
            "tool_calls_made": 0,
        })

        handlers = ARIAHandlers(mock_client, allowed_chat_id="12345")

        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.text = "테스트 질문"

        thinking_msg = MagicMock()
        thinking_msg.delete = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=thinking_msg)

        context = MagicMock()
        await handlers.handle_message(update, context)

        mock_client.query.assert_called_once_with(
            "테스트 질문",
            scope="global",
            collection="default",
        )

    @pytest.mark.asyncio
    async def test_handle_message_with_scope(self) -> None:
        """@testorum 스코프 지정 메시지"""
        from aria.telegram.handlers import ARIAHandlers

        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value={
            "answer": "Testorum 답변",
            "confidence": 0.9,
            "tool_calls_made": 0,
        })

        handlers = ARIAHandlers(mock_client, allowed_chat_id="12345")

        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.text = "@testorum 사용자 통계 분석"

        thinking_msg = MagicMock()
        thinking_msg.delete = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=thinking_msg)

        context = MagicMock()
        await handlers.handle_message(update, context)

        mock_client.query.assert_called_once_with(
            "사용자 통계 분석",
            scope="testorum",
            collection="default",
        )


# === Bot Creation Tests ===


class TestBotCreation:
    def test_create_bot_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aria.telegram.bot import create_bot
        monkeypatch.delenv("ARIA_TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_ARIA_API_KEY", raising=False)
        monkeypatch.setenv("ARIA_ENV_FILE", "")
        config = TelegramConfig()  # bot_token=""

        with pytest.raises(ValueError, match="BOT_TOKEN"):
            create_bot(config)

    def test_create_bot_missing_chat_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aria.telegram.bot import create_bot
        monkeypatch.delenv("ARIA_TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("ARIA_TELEGRAM_ARIA_API_KEY", raising=False)
        monkeypatch.setenv("ARIA_ENV_FILE", "")
        monkeypatch.setenv("ARIA_TELEGRAM_BOT_TOKEN", "fake-token")
        config = TelegramConfig()  # chat_id=""

        with pytest.raises(ValueError, match="CHAT_ID"):
            create_bot(config)
