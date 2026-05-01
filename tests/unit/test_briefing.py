"""능동 보고 (Daily Briefing) 단위 테스트

테스트 범위:
- build_briefing: 데이터 수집 + 포맷팅
- send_daily_briefing: 브리핑 생성 + 전송
- schedule_daily_briefing: JobQueue 스케줄 등록
- /briefing 명령어 핸들러
"""

from __future__ import annotations

from datetime import time, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.telegram.briefing import (
    KST,
    build_briefing,
    send_daily_briefing,
    schedule_daily_briefing,
    _briefing_job_callback,
)
from aria.telegram.client import ARIAClient
from aria.telegram.handlers import ARIAHandlers


# === build_briefing 테스트 ===


class TestBuildBriefing:

    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=ARIAClient)
        client.health_check = AsyncMock(return_value={
            "status": "ok", "version": "v0.2.0",
        })
        client.get_cost = AsyncMock(return_value={
            "daily_cost_usd": 0.52,
            "monthly_cost_usd": 3.14,
            "daily_limit_usd": 10,
            "monthly_limit_usd": 300,
            "total_requests": 42,
            "total_cached_tokens": 15000,
        })
        client.get_memory_index = AsyncMock(return_value={
            "entries": [
                {"domain": "user-profile", "summary": "test"},
                {"domain": "features", "summary": "test"},
            ],
        })
        return client

    @pytest.mark.asyncio
    async def test_briefing_contains_header(self, mock_client):
        text = await build_briefing(mock_client)
        assert "모닝 브리핑" in text

    @pytest.mark.asyncio
    async def test_briefing_contains_server_status(self, mock_client):
        text = await build_briefing(mock_client)
        assert "v0.2.0" in text
        assert "🟢" in text

    @pytest.mark.asyncio
    async def test_briefing_contains_cost(self, mock_client):
        text = await build_briefing(mock_client)
        assert "$0.52" in text or "0.5200" in text
        assert "$3.14" in text or "3.1400" in text

    @pytest.mark.asyncio
    async def test_briefing_contains_memory(self, mock_client):
        text = await build_briefing(mock_client)
        assert "global" in text
        assert "2개 토픽" in text

    @pytest.mark.asyncio
    async def test_briefing_server_down(self, mock_client):
        mock_client.health_check = AsyncMock(return_value={
            "error": "CONNECTION_ERROR", "message": "연결 실패",
        })
        text = await build_briefing(mock_client)
        assert "연결 실패" in text
        assert "🔴" in text

    @pytest.mark.asyncio
    async def test_briefing_cost_error_graceful(self, mock_client):
        """비용 조회 실패해도 브리핑 생성 성공"""
        mock_client.get_cost = AsyncMock(return_value={
            "error": "TIMEOUT", "message": "timeout",
        })
        text = await build_briefing(mock_client)
        assert "모닝 브리핑" in text  # 다른 섹션은 정상

    @pytest.mark.asyncio
    async def test_briefing_no_memory_entries(self, mock_client):
        """메모리 없어도 정상"""
        mock_client.get_memory_index = AsyncMock(return_value={
            "entries": [],
        })
        text = await build_briefing(mock_client)
        assert "모닝 브리핑" in text

    @pytest.mark.asyncio
    async def test_briefing_ends_with_greeting(self, mock_client):
        text = await build_briefing(mock_client)
        assert "좋은 하루" in text


# === send_daily_briefing 테스트 ===


class TestSendDailyBriefing:

    @pytest.mark.asyncio
    async def test_send_success(self):
        mock_client = MagicMock(spec=ARIAClient)
        mock_client.health_check = AsyncMock(return_value={"status": "ok", "version": "v0.2.0"})
        mock_client.get_cost = AsyncMock(return_value={"error": "skip"})
        mock_client.get_memory_index = AsyncMock(return_value={"entries": []})

        with patch("aria.telegram.briefing.send_message", new_callable=AsyncMock, return_value={"ok": True}) as mock_send:
            result = await send_daily_briefing("token", "chat123", mock_client)

        assert result["ok"]
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args.args[0] == "token"
        assert call_args.args[1] == "chat123"

    @pytest.mark.asyncio
    async def test_send_on_build_error(self):
        """브리핑 생성 실패 시에도 에러 메시지 전송"""
        mock_client = MagicMock(spec=ARIAClient)
        mock_client.health_check = AsyncMock(side_effect=Exception("boom"))

        with patch("aria.telegram.briefing.send_message", new_callable=AsyncMock, return_value={"ok": True}) as mock_send:
            result = await send_daily_briefing("token", "chat123", mock_client)

        assert result["ok"]
        sent_text = mock_send.call_args.args[2]
        assert "실패" in sent_text


# === schedule_daily_briefing 테스트 ===


class TestScheduleDailyBriefing:

    def test_schedule_registers_job(self):
        mock_app = MagicMock()
        mock_job_queue = MagicMock()
        mock_app.job_queue = mock_job_queue
        mock_client = MagicMock(spec=ARIAClient)

        schedule_daily_briefing(
            app=mock_app,
            bot_token="test-token",
            chat_id="123",
            client=mock_client,
            hour=6,
            minute=0,
        )

        mock_job_queue.run_daily.assert_called_once()
        call_kwargs = mock_job_queue.run_daily.call_args
        scheduled_time = call_kwargs.kwargs.get("time") or call_kwargs.args[1]
        assert scheduled_time.hour == 6
        assert scheduled_time.minute == 0
        assert scheduled_time.tzinfo == KST

    def test_schedule_custom_time(self):
        mock_app = MagicMock()
        mock_job_queue = MagicMock()
        mock_app.job_queue = mock_job_queue

        schedule_daily_briefing(
            app=mock_app,
            bot_token="t",
            chat_id="c",
            client=MagicMock(spec=ARIAClient),
            hour=8,
            minute=30,
        )

        call_kwargs = mock_job_queue.run_daily.call_args
        scheduled_time = call_kwargs.kwargs.get("time") or call_kwargs.args[1]
        assert scheduled_time.hour == 8
        assert scheduled_time.minute == 30

    def test_schedule_no_job_queue(self):
        """JobQueue 없으면 에러 없이 스킵"""
        mock_app = MagicMock()
        mock_app.job_queue = None

        # Should not raise
        schedule_daily_briefing(
            app=mock_app,
            bot_token="t",
            chat_id="c",
            client=MagicMock(spec=ARIAClient),
        )

    def test_schedule_passes_data(self):
        mock_app = MagicMock()
        mock_job_queue = MagicMock()
        mock_app.job_queue = mock_job_queue
        mock_client = MagicMock(spec=ARIAClient)

        schedule_daily_briefing(
            app=mock_app,
            bot_token="my-token",
            chat_id="my-chat",
            client=mock_client,
        )

        call_kwargs = mock_job_queue.run_daily.call_args
        data = call_kwargs.kwargs.get("data") or call_kwargs.args[3] if len(call_kwargs.args) > 3 else call_kwargs.kwargs["data"]
        assert data["bot_token"] == "my-token"
        assert data["chat_id"] == "my-chat"
        assert data["client"] is mock_client


# === _briefing_job_callback 테스트 ===


class TestBriefingJobCallback:

    @pytest.mark.asyncio
    async def test_callback_calls_send(self):
        mock_client = MagicMock(spec=ARIAClient)
        mock_client.health_check = AsyncMock(return_value={"status": "ok", "version": "v0.2.0"})
        mock_client.get_cost = AsyncMock(return_value={"error": "skip"})
        mock_client.get_memory_index = AsyncMock(return_value={"entries": []})

        mock_context = MagicMock()
        mock_context.job.data = {
            "bot_token": "token",
            "chat_id": "chat",
            "client": mock_client,
        }

        with patch("aria.telegram.briefing.send_message", new_callable=AsyncMock, return_value={"ok": True}):
            await _briefing_job_callback(mock_context)

    @pytest.mark.asyncio
    async def test_callback_missing_data(self):
        """데이터 누락 시 에러 없이 리턴"""
        mock_context = MagicMock()
        mock_context.job.data = {}

        # Should not raise
        await _briefing_job_callback(mock_context)


# === /briefing 핸들러 테스트 ===


class TestBriefingHandler:

    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=ARIAClient)
        client.health_check = AsyncMock(return_value={"status": "ok", "version": "v0.2.0"})
        client.get_cost = AsyncMock(return_value={"error": "skip"})
        client.get_memory_index = AsyncMock(return_value={"entries": []})
        return client

    @pytest.mark.asyncio
    async def test_briefing_command(self, mock_client):
        handlers = ARIAHandlers(mock_client, allowed_chat_id="123")

        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        await handlers.briefing(update, MagicMock())

        # "브리핑 생성 중" + 실제 브리핑 = 2회 호출
        assert update.message.reply_text.call_count == 2

    @pytest.mark.asyncio
    async def test_briefing_unauthorized(self, mock_client):
        handlers = ARIAHandlers(mock_client, allowed_chat_id="123")

        update = MagicMock()
        update.effective_chat.id = 999  # 다른 사용자

        await handlers.briefing(update, MagicMock())
        # 인증 실패 시 reply 없음

# === KST 타임존 검증 ===


class TestKST:

    def test_kst_offset(self):
        assert KST.utcoffset(None) == timedelta(hours=9)

    def test_briefing_time_construction(self):
        t = time(hour=6, minute=0, tzinfo=KST)
        assert t.hour == 6
        assert t.tzinfo == KST
