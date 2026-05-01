"""ARIA Engine - Pending Action Store

HITL(Human-in-the-Loop) 대기 도구 액션 저장소
Critic이 NEEDS_CONFIRMATION 판정 시 도구 정보를 보관하여
사용자 승인 후 실제 실행할 수 있게 함

설계:
- 인메모리 dict (1인 사용 + 단일 서버 → DB 불필요)
- TTL 기반 자동 만료 (기본 30분)
- asyncio Lock으로 thread-safe
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()

# 기본 TTL: 30분
DEFAULT_TTL_SECONDS = 1800


@dataclass
class PendingAction:
    """대기 중인 도구 실행 액션"""
    confirmation_id: str
    tool_name: str
    parameters: dict[str, Any]
    context: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = DEFAULT_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        """TTL 초과 여부"""
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


class PendingStore:
    """HITL 대기 액션 저장소

    confirmation_id를 키로 PendingAction을 저장/조회/삭제
    만료된 액션은 조회 시 자동 정리

    사용법:
        store = PendingStore()
        store.add(PendingAction(confirmation_id="abc", tool_name="notion_create_page", ...))
        action = store.get("abc")  # None if expired
        store.remove("abc")
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._store: dict[str, PendingAction] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def add(self, action: PendingAction) -> None:
        """대기 액션 추가"""
        self._store[action.confirmation_id] = action
        logger.info(
            "pending_action_added",
            confirmation_id=action.confirmation_id,
            tool_name=action.tool_name,
            ttl=action.ttl_seconds,
        )

    def get(self, confirmation_id: str) -> PendingAction | None:
        """대기 액션 조회 (만료 시 자동 삭제 후 None 반환)"""
        action = self._store.get(confirmation_id)
        if action is None:
            return None
        if action.is_expired:
            del self._store[confirmation_id]
            logger.info("pending_action_expired", confirmation_id=confirmation_id)
            return None
        return action

    def remove(self, confirmation_id: str) -> bool:
        """대기 액션 삭제 (존재하면 True)"""
        if confirmation_id in self._store:
            del self._store[confirmation_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """만료된 모든 액션 정리 → 정리된 개수 반환"""
        expired = [
            cid for cid, action in self._store.items()
            if action.is_expired
        ]
        for cid in expired:
            del self._store[cid]
        if expired:
            logger.info("pending_actions_cleaned", count=len(expired))
        return len(expired)

    @property
    def count(self) -> int:
        """현재 저장된 액션 수 (만료 포함)"""
        return len(self._store)

    @property
    def active_count(self) -> int:
        """만료되지 않은 액션 수"""
        return sum(1 for a in self._store.values() if not a.is_expired)
