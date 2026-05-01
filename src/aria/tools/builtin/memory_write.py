"""ARIA Engine - Built-in Tool: Memory Write

ARIA 메모리 시스템에 토픽을 생성/업데이트하는 내장 도구
에이전트가 대화에서 학습한 정보를 자율적으로 메모리에 저장할 때 사용

사용 시나리오:
- "이 정보를 기억해둬" → memory_write(scope="global", domain="...", content="...")
- 에이전트가 추론 중 새 인사이트 발견 → 자동 메모리 업데이트

안전성: SafetyLevelHint.WRITE → Critic이 NEEDS_CONFIRMATION 판단할 수 있음
read-before-write 규율: 기존 토픽 업데이트 시 expected_version 자동 처리
"""

from __future__ import annotations

from typing import Any

import structlog

from aria.core.exceptions import MemoryNotFoundError, VersionConflictError
from aria.memory.index_manager import IndexManager
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)

logger = structlog.get_logger()


class MemoryWriteTool(ToolExecutor):
    """메모리 토픽 생성/업데이트 도구

    read-before-write 자동 처리:
    - 기존 토픽이 있으면 → 현재 버전을 읽어서 expected_version으로 전달
    - 기존 토픽이 없으면 → expected_version=None으로 신규 생성

    Args:
        index_manager: ARIA IndexManager 인스턴스

    사용법:
        tool = MemoryWriteTool(index_manager)
        registry.register_executor(tool)
    """

    def __init__(self, index_manager: IndexManager) -> None:
        self._manager = index_manager

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_write",
            description=(
                "ARIA 메모리에 토픽을 생성하거나 업데이트합니다. "
                "기존 토픽이 있으면 자동으로 버전을 확인하고 업데이트하며, "
                "없으면 새로 생성합니다. "
                "데이터를 수정하는 작업이므로 사용자 확인이 필요할 수 있습니다."
            ),
            parameters=[
                ToolParameter(
                    name="scope",
                    type="string",
                    description="메모리 스코프 (global / testorum / talksim / autotube)",
                    required=True,
                ),
                ToolParameter(
                    name="domain",
                    type="string",
                    description="토픽 도메인 식별자 (예: user-profile / meeting-notes)",
                    required=True,
                ),
                ToolParameter(
                    name="summary",
                    type="string",
                    description="토픽 요약 (120자 이내 — 인덱스 스캔용)",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="토픽 본문 (마크다운 형식)",
                    required=True,
                ),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.WRITE,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """메모리 토픽 upsert 실행

        read-before-write 자동 처리:
        1. 기존 토픽 존재 여부 확인
        2. 존재하면 → 현재 version을 expected_version으로 전달
        3. 미존재 → expected_version=None (신규 생성)
        """
        scope = parameters["scope"]
        domain = parameters["domain"]
        summary = parameters["summary"]
        content = parameters["content"]

        try:
            # read-before-write: 기존 토픽 확인
            expected_version: int | None = None
            try:
                existing = self._manager.get_topic(scope, domain)
                expected_version = existing.version
                is_update = True
            except MemoryNotFoundError:
                is_update = False

            # upsert 실행
            topic = self._manager.upsert_topic(
                scope=scope,
                domain=domain,
                summary=summary,
                content=content,
                expected_version=expected_version,
            )

            action = "업데이트" if is_update else "생성"
            logger.info(
                "memory_write_success",
                scope=scope,
                domain=domain,
                action=action,
                version=topic.version,
                content_length=len(content),
            )

            return ToolResult(
                tool_name="memory_write",
                success=True,
                output={
                    "action": action,
                    "scope": topic.scope,
                    "domain": topic.domain,
                    "version": topic.version,
                    "updated_at": topic.updated_at.isoformat(),
                    "content_length": len(content),
                },
            )

        except VersionConflictError as e:
            # 동시 수정 충돌 — 에이전트에게 재시도 안내
            logger.warning(
                "memory_write_conflict",
                scope=scope,
                domain=domain,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="memory_write",
                success=False,
                error=(
                    f"버전 충돌: '{domain}' 토픽이 다른 곳에서 수정되었습니다. "
                    "최신 버전을 다시 읽고 재시도하세요."
                ),
            )

        except Exception as e:
            logger.error(
                "memory_write_failed",
                scope=scope,
                domain=domain,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="memory_write",
                success=False,
                error=f"메모리 쓰기 실패: {str(e)[:300]}",
            )
