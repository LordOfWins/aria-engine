"""HITL (Human-in-the-Loop) 실행 연동 단위 테스트

테스트 범위:
- PendingAction: 생성 / 만료 / TTL
- PendingStore: add / get / remove / cleanup / TTL
- ToolRegistry: execute_pending / deny_pending / PendingStore 연동
- ARIAClient: execute_pending / deny_pending HTTP 메서드
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.tools.pending_store import PendingAction, PendingStore, DEFAULT_TTL_SECONDS
from aria.tools.tool_types import (
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
    ToolCategory,
    SafetyLevelHint,
)
from aria.tools.tool_registry import ToolRegistry, ToolNotFoundError
from aria.telegram.client import ARIAClient


# === Test Helpers ===


class EchoExecutor(ToolExecutor):
    """테스트용 에코 도구"""

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name="echo_tool",
            success=True,
            output=f"echo: {parameters.get('message', '')}",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo_tool",
            description="메시지를 그대로 반환합니다",
            parameters=[
                ToolParameter(name="message", type="string", required=True),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.READ_ONLY,
        )


class FailExecutor(ToolExecutor):
    """항상 실패하는 테스트 도구"""

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        raise RuntimeError("의도적 실패")

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fail_tool",
            description="항상 실패",
        )


# === PendingAction 테스트 ===


class TestPendingAction:

    def test_create(self):
        action = PendingAction(
            confirmation_id="confirm-abc123",
            tool_name="echo_tool",
            parameters={"message": "hello"},
        )
        assert action.confirmation_id == "confirm-abc123"
        assert action.tool_name == "echo_tool"
        assert not action.is_expired

    def test_not_expired_within_ttl(self):
        action = PendingAction(
            confirmation_id="x",
            tool_name="y",
            parameters={},
            ttl_seconds=3600,
        )
        assert not action.is_expired

    def test_expired_after_ttl(self):
        action = PendingAction(
            confirmation_id="x",
            tool_name="y",
            parameters={},
            created_at=datetime.now(timezone.utc) - timedelta(seconds=1900),
            ttl_seconds=1800,
        )
        assert action.is_expired

    def test_default_ttl(self):
        action = PendingAction(confirmation_id="x", tool_name="y", parameters={})
        assert action.ttl_seconds == DEFAULT_TTL_SECONDS


# === PendingStore 테스트 ===


class TestPendingStore:

    def test_add_and_get(self):
        store = PendingStore()
        action = PendingAction(
            confirmation_id="abc",
            tool_name="echo_tool",
            parameters={"message": "hi"},
        )
        store.add(action)
        assert store.get("abc") is action
        assert store.count == 1

    def test_get_nonexistent(self):
        store = PendingStore()
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self):
        store = PendingStore()
        action = PendingAction(
            confirmation_id="old",
            tool_name="x",
            parameters={},
            created_at=datetime.now(timezone.utc) - timedelta(seconds=3600),
            ttl_seconds=1800,
        )
        store.add(action)
        assert store.get("old") is None
        assert store.count == 0  # 자동 삭제됨

    def test_remove(self):
        store = PendingStore()
        store.add(PendingAction(confirmation_id="abc", tool_name="x", parameters={}))
        assert store.remove("abc")
        assert store.get("abc") is None

    def test_remove_nonexistent(self):
        store = PendingStore()
        assert not store.remove("nonexistent")

    def test_cleanup_expired(self):
        store = PendingStore()
        old_time = datetime.now(timezone.utc) - timedelta(seconds=3600)

        store.add(PendingAction(confirmation_id="old1", tool_name="x", parameters={}, created_at=old_time, ttl_seconds=1800))
        store.add(PendingAction(confirmation_id="old2", tool_name="x", parameters={}, created_at=old_time, ttl_seconds=1800))
        store.add(PendingAction(confirmation_id="new1", tool_name="x", parameters={}))

        cleaned = store.cleanup_expired()
        assert cleaned == 2
        assert store.count == 1
        assert store.get("new1") is not None

    def test_active_count(self):
        store = PendingStore()
        old_time = datetime.now(timezone.utc) - timedelta(seconds=3600)

        store.add(PendingAction(confirmation_id="old", tool_name="x", parameters={}, created_at=old_time, ttl_seconds=1800))
        store.add(PendingAction(confirmation_id="new", tool_name="x", parameters={}))

        assert store.count == 2
        assert store.active_count == 1


# === ToolRegistry HITL 테스트 ===


class TestToolRegistryHITL:

    @pytest.fixture
    def registry(self):
        reg = ToolRegistry(critic=None)
        reg.register_executor(EchoExecutor())
        return reg

    @pytest.mark.asyncio
    async def test_execute_pending_success(self, registry):
        """승인 후 대기 액션 실행 성공"""
        # 수동으로 pending action 추가
        registry.pending_store.add(PendingAction(
            confirmation_id="test-confirm",
            tool_name="echo_tool",
            parameters={"message": "approved!"},
        ))

        result = await registry.execute_pending("test-confirm")
        assert result.success
        assert "approved!" in result.output

    @pytest.mark.asyncio
    async def test_execute_pending_removes_action(self, registry):
        """실행 후 pending store에서 삭제됨"""
        registry.pending_store.add(PendingAction(
            confirmation_id="once",
            tool_name="echo_tool",
            parameters={"message": "x"},
        ))

        await registry.execute_pending("once")
        assert registry.pending_store.get("once") is None

    @pytest.mark.asyncio
    async def test_execute_pending_not_found(self, registry):
        """존재하지 않는 confirmation_id → ToolNotFoundError"""
        with pytest.raises(ToolNotFoundError):
            await registry.execute_pending("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_pending_expired(self, registry):
        """만료된 액션 → ToolNotFoundError"""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=3600)
        registry.pending_store.add(PendingAction(
            confirmation_id="expired",
            tool_name="echo_tool",
            parameters={"message": "x"},
            created_at=old_time,
            ttl_seconds=1800,
        ))

        with pytest.raises(ToolNotFoundError):
            await registry.execute_pending("expired")

    @pytest.mark.asyncio
    async def test_execute_pending_tool_removed(self, registry):
        """도구가 제거된 상태에서 실행 → ToolNotFoundError"""
        registry.pending_store.add(PendingAction(
            confirmation_id="orphan",
            tool_name="nonexistent_tool",
            parameters={},
        ))

        with pytest.raises(ToolNotFoundError):
            await registry.execute_pending("orphan")

    @pytest.mark.asyncio
    async def test_execute_pending_executor_crash(self):
        """실행 중 예외 → 실패 ToolResult 반환"""
        reg = ToolRegistry(critic=None)
        reg.register_executor(FailExecutor())
        reg.pending_store.add(PendingAction(
            confirmation_id="crash",
            tool_name="fail_tool",
            parameters={},
        ))

        result = await reg.execute_pending("crash")
        assert not result.success
        assert "의도적 실패" in result.error

    def test_deny_pending_exists(self, registry):
        """거부 시 삭제"""
        registry.pending_store.add(PendingAction(
            confirmation_id="deny-me",
            tool_name="echo_tool",
            parameters={},
        ))

        assert registry.deny_pending("deny-me")
        assert registry.pending_store.get("deny-me") is None

    def test_deny_pending_nonexistent(self, registry):
        """존재하지 않는 것 거부 → False"""
        assert not registry.deny_pending("nonexistent")

    def test_pending_store_property(self, registry):
        """pending_store 프로퍼티 접근"""
        assert registry.pending_store is not None
        assert isinstance(registry.pending_store, PendingStore)


# === ARIAClient HITL 메서드 테스트 ===


class TestARIAClientHITL:

    def test_client_has_execute_pending(self):
        client = ARIAClient(base_url="http://localhost:8100")
        assert hasattr(client, "execute_pending")

    def test_client_has_deny_pending(self):
        client = ARIAClient(base_url="http://localhost:8100")
        assert hasattr(client, "deny_pending")

    def test_client_has_delete_method(self):
        client = ARIAClient(base_url="http://localhost:8100")
        assert hasattr(client, "_delete")
