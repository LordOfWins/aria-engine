"""ARIA Engine - Built-in Tool: Memory Read

ARIA 메모리 시스템에서 토픽을 조회하는 내장 도구
에이전트가 자율적으로 메모리를 읽어 컨텍스트를 보강할 때 사용

사용 시나리오:
- "사용자 프로필은?" → memory_read(scope="global", domain="user-profile")
- "Testorum 기능 목록" → memory_read(scope="testorum", domain="features")
- "어떤 토픽이 있는지 보여줘" → memory_read(scope="global", domain="") → 인덱스 반환
"""

from __future__ import annotations

from typing import Any

import structlog

from aria.core.exceptions import MemoryNotFoundError
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


class MemoryReadTool(ToolExecutor):
    """메모리 토픽 조회 도구

    Args:
        index_manager: ARIA IndexManager 인스턴스

    사용법:
        tool = MemoryReadTool(index_manager)
        registry.register_executor(tool)
    """

    def __init__(self, index_manager: IndexManager) -> None:
        self._manager = index_manager

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_read",
            description=(
                "ARIA 메모리에서 토픽을 조회합니다. "
                "domain을 지정하면 해당 토픽의 본문을 반환하고, "
                "domain을 비우면 해당 스코프의 인덱스(토픽 목록)를 반환합니다."
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
                    description="토픽 도메인 식별자 (예: user-profile / features). 비우면 인덱스 조회",
                    required=False,
                ),
            ],
            category=ToolCategory.BUILTIN,
            safety_hint=SafetyLevelHint.READ_ONLY,
            version="1.0.0",
        )

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """메모리 조회 실행

        domain이 있으면: 토픽 본문 반환
        domain이 없거나 빈 문자열이면: 인덱스(토픽 목록) 반환
        """
        scope = parameters.get("scope", "global")
        domain = parameters.get("domain", "")

        try:
            if domain:
                return await self._read_topic(scope, domain)
            return await self._read_index(scope)
        except MemoryNotFoundError as e:
            return ToolResult(
                tool_name="memory_read",
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "memory_read_failed",
                scope=scope,
                domain=domain,
                error=str(e)[:200],
            )
            return ToolResult(
                tool_name="memory_read",
                success=False,
                error=f"메모리 조회 실패: {str(e)[:300]}",
            )

    async def _read_topic(self, scope: str, domain: str) -> ToolResult:
        """토픽 본문 조회"""
        topic = self._manager.get_topic(scope, domain)
        entry = self._manager.get_entry(scope, domain)

        output = {
            "domain": topic.domain,
            "scope": topic.scope,
            "content": topic.content,
            "version": topic.version,
            "updated_at": topic.updated_at.isoformat(),
            "summary": entry.summary if entry else "",
        }

        logger.info(
            "memory_read_topic",
            scope=scope,
            domain=domain,
            content_length=len(topic.content),
        )

        return ToolResult(
            tool_name="memory_read",
            success=True,
            output=output,
        )

    async def _read_index(self, scope: str) -> ToolResult:
        """인덱스(토픽 목록) 조회"""
        index = self._manager.get_index(scope)
        entries = [
            {
                "domain": entry.domain,
                "summary": entry.summary,
                "updated_at": entry.updated_at.isoformat(),
                "token_estimate": entry.token_estimate,
            }
            for entry in index.entries
        ]

        logger.info(
            "memory_read_index",
            scope=scope,
            entry_count=len(entries),
        )

        return ToolResult(
            tool_name="memory_read",
            success=True,
            output={
                "scope": scope,
                "entries": entries,
                "total_topics": len(entries),
            },
        )
