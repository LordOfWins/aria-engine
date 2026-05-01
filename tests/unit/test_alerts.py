"""ARIA Engine - Proactive Alert Tests

능동 알림 시스템 단위 테스트
- AlertType / AlertLevel / Alert 스키마
- AlertManager 비용/confidence/에러/메모리/서버 알림
- 쿨다운 동작
- 비활성화 동작
- API 통합
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from aria.alerts.alert_types import (
    Alert,
    AlertLevel,
    AlertType,
    ALERT_EMOJI,
    DEFAULT_COOLDOWNS,
)
from aria.alerts.alert_manager import AlertManager


# === Fixtures ===


@pytest.fixture
def manager():
    """활성화된 AlertManager (텔레그램 mock)"""
    return AlertManager(
        bot_token="test-token",
        chat_id="123456",
        enabled=True,
        cost_warning_threshold=0.7,
        cost_critical_threshold=0.9,
        confidence_threshold=0.3,
        consecutive_error_threshold=3,
    )


@pytest.fixture
def disabled_manager():
    """비활성화된 AlertManager"""
    return AlertManager(
        bot_token="test-token",
        chat_id="123456",
        enabled=False,
    )


@pytest.fixture
def no_token_manager():
    """토큰 없는 AlertManager"""
    return AlertManager(
        bot_token="",
        chat_id="123456",
        enabled=True,
    )


# === AlertType / AlertLevel Tests ===


class TestAlertEnums:
    """알림 enum 테스트"""

    def test_alert_level_values(self):
        assert AlertLevel.INFO == "info"
        assert AlertLevel.WARNING == "warning"
        assert AlertLevel.CRITICAL == "critical"

    def test_alert_type_values(self):
        assert AlertType.COST_WARNING == "cost_warning"
        assert AlertType.KILLSWITCH == "killswitch"
        assert AlertType.LOW_CONFIDENCE == "low_confidence"

    def test_all_types_have_emoji(self):
        for alert_type in AlertType:
            assert alert_type in ALERT_EMOJI

    def test_all_types_have_cooldown(self):
        for alert_type in AlertType:
            assert alert_type in DEFAULT_COOLDOWNS


# === Alert Model Tests ===


class TestAlert:
    """Alert 모델 테스트"""

    def test_create_alert(self):
        alert = Alert(
            alert_type=AlertType.COST_WARNING,
            level=AlertLevel.WARNING,
            title="비용 경고",
            message="일일 비용 70% 도달",
        )
        assert alert.alert_type == AlertType.COST_WARNING
        assert alert.timestamp  # 자동 생성

    def test_to_telegram_warning(self):
        alert = Alert(
            alert_type=AlertType.COST_WARNING,
            level=AlertLevel.WARNING,
            title="비용 70% 도달",
            message="일일: $7.00 / $10.00",
        )
        text = alert.to_telegram()
        assert "💰" in text
        assert "[주의]" in text
        assert "$7.00" in text

    def test_to_telegram_critical(self):
        alert = Alert(
            alert_type=AlertType.KILLSWITCH,
            level=AlertLevel.CRITICAL,
            title="KillSwitch",
            message="차단됨",
        )
        text = alert.to_telegram()
        assert "🚨" in text
        assert "[긴급]" in text

    def test_to_telegram_with_data(self):
        alert = Alert(
            alert_type=AlertType.COST_WARNING,
            level=AlertLevel.WARNING,
            title="테스트",
            message="본문",
            data={"key1": "value1", "key2": 42},
        )
        text = alert.to_telegram()
        assert "key1: value1" in text
        assert "key2: 42" in text


# === AlertManager Initialization Tests ===


class TestAlertManagerInit:
    """AlertManager 초기화 테스트"""

    def test_enabled_with_token_and_chat_id(self, manager):
        assert manager.enabled is True

    def test_disabled_explicitly(self, disabled_manager):
        assert disabled_manager.enabled is False

    def test_disabled_without_token(self, no_token_manager):
        assert no_token_manager.enabled is False

    def test_initial_stats(self, manager):
        stats = manager.get_stats()
        assert stats["enabled"] is True
        assert stats["sent_count"] == 0
        assert stats["consecutive_errors"] == 0


# === Cost Alert Tests ===


class TestCostAlerts:
    """비용 알림 테스트"""

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            result = await manager.check_cost(
                daily=3.0, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is None
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_warning_at_70_percent(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_cost(
                daily=7.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is not None
            assert result.alert_type == AlertType.COST_WARNING
            assert result.level == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_critical_at_90_percent(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is not None
            assert result.alert_type == AlertType.COST_CRITICAL
            assert result.level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_monthly_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_cost(
                daily=1.0, monthly=270.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is not None
            assert result.alert_type == AlertType.COST_CRITICAL  # 270/300 = 90%

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, disabled_manager):
        result = await disabled_manager.check_cost(
            daily=9.5, monthly=290.0,
            daily_limit=10.0, monthly_limit=300.0,
        )
        assert result is None


# === KillSwitch Alert Tests ===


class TestKillSwitchAlerts:
    """KillSwitch 알림 테스트"""

    @pytest.mark.asyncio
    async def test_killswitch_alert(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_killswitch(daily_cost=10.5, monthly_cost=305.0)
            assert result is not None
            assert result.alert_type == AlertType.KILLSWITCH
            assert result.level == AlertLevel.CRITICAL


# === Confidence Alert Tests ===


class TestConfidenceAlerts:
    """Confidence 알림 테스트"""

    @pytest.mark.asyncio
    async def test_no_alert_above_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock):
            result = await manager.check_confidence(confidence=0.5, query="테스트 질문")
            assert result is None

    @pytest.mark.asyncio
    async def test_alert_below_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_confidence(confidence=0.2, query="테스트 질문")
            assert result is not None
            assert result.alert_type == AlertType.LOW_CONFIDENCE

    @pytest.mark.asyncio
    async def test_long_query_truncated(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            long_query = "a" * 200
            result = await manager.check_confidence(confidence=0.1, query=long_query)
            assert result is not None
            assert "..." in result.message


# === Consecutive Error Alert Tests ===


class TestErrorAlerts:
    """연속 에러 알림 테스트"""

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock):
            result = await manager.check_error("AgentError", "에이전트 실패")
            assert result is None  # 1회 → 아직 임계치 미도달
            assert manager.consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_alert_at_threshold(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            # 3회 연속
            await manager.check_error("AgentError", "1차 실패")
            await manager.check_error("AgentError", "2차 실패")
            result = await manager.check_error("AgentError", "3차 실패")
            assert result is not None
            assert result.alert_type == AlertType.CONSECUTIVE_ERRORS
            assert manager.consecutive_errors == 3

    @pytest.mark.asyncio
    async def test_reset_error_counter(self, manager):
        manager._consecutive_errors = 2
        manager.reset_error_counter()
        assert manager.consecutive_errors == 0


# === Memory Conflict Alert Tests ===


class TestMemoryConflictAlerts:
    """메모리 충돌 알림 테스트"""

    @pytest.mark.asyncio
    async def test_memory_conflict_alert(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_memory_conflict(
                scope="global", domain="daily-trends",
                expected_version=3, actual_version=5,
            )
            assert result is not None
            assert result.alert_type == AlertType.MEMORY_CONFLICT


# === Server Error Alert Tests ===


class TestServerErrorAlerts:
    """서버 에러 알림 테스트"""

    @pytest.mark.asyncio
    async def test_server_error_alert(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            result = await manager.check_server_error(
                path="/v1/query",
                error="RuntimeError: 예상 못한 에러",
            )
            assert result is not None
            assert result.alert_type == AlertType.SERVER_ERROR
            assert result.level == AlertLevel.CRITICAL


# === Cooldown Tests ===


class TestCooldown:
    """쿨다운 동작 테스트"""

    @pytest.mark.asyncio
    async def test_cooldown_blocks_duplicate(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}

            # 첫 번째: 발송
            result1 = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result1 is not None

            # 두 번째: 쿨다운 → 미발송
            result2 = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result2 is None

            assert mock_send.call_count == 1  # 1회만 발송

    @pytest.mark.asyncio
    async def test_cooldown_expires(self, manager):
        """쿨다운 만료 후 재발송"""
        # 쿨다운을 0초로 설정
        manager._cooldowns[AlertType.COST_CRITICAL] = 0

        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}

            result1 = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result1 is not None

            # 쿨다운 0초 → 즉시 재발송 가능
            result2 = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result2 is not None

    @pytest.mark.asyncio
    async def test_different_types_independent(self, manager):
        """다른 알림 유형은 독립 쿨다운"""
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}

            # 비용 경고
            await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )

            # KillSwitch — 다른 유형이므로 쿨다운 영향 없음
            result = await manager.check_killswitch(daily_cost=10.5, monthly_cost=305.0)
            assert result is not None

            assert mock_send.call_count == 2


# === Send Failure Handling Tests ===


class TestSendFailure:
    """발송 실패 처리 테스트"""

    @pytest.mark.asyncio
    async def test_send_failure_returns_none(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": False, "error": "텔레그램 오류"}
            result = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is None
            assert manager.sent_count == 0  # 실패 → 카운트 안 됨

    @pytest.mark.asyncio
    async def test_send_exception_returns_none(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("네트워크 에러")
            result = await manager.check_cost(
                daily=9.5, monthly=100.0,
                daily_limit=10.0, monthly_limit=300.0,
            )
            assert result is None  # 예외 발생해도 None (메인 로직 영향 없음)


# === Stats Tests ===


class TestAlertStats:
    """알림 통계 테스트"""

    @pytest.mark.asyncio
    async def test_stats_after_send(self, manager):
        with patch("aria.alerts.alert_manager.send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"ok": True}
            await manager.check_killswitch(daily_cost=10.5, monthly_cost=305.0)

            stats = manager.get_stats()
            assert stats["sent_count"] == 1
            assert "killswitch" in stats["seconds_since_last"]


# === AlertConfig Tests ===


class TestAlertConfig:
    """AlertConfig 설정 테스트"""

    def test_default_values(self):
        from aria.core.config import AlertConfig
        config = AlertConfig()
        assert config.enabled is True
        assert config.cost_warning_threshold == 0.7
        assert config.cost_critical_threshold == 0.9
        assert config.confidence_threshold == 0.3
        assert config.consecutive_error_threshold == 3

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ARIA_ALERT_ENABLED", "false")
        monkeypatch.setenv("ARIA_ALERT_COST_WARNING_THRESHOLD", "0.5")
        from aria.core.config import AlertConfig
        config = AlertConfig()
        assert config.enabled is False
        assert config.cost_warning_threshold == 0.5
