"""ARIA Engine - Tools Module

도구 연결 및 안전성 제어
- Tool Registry: 도구 등록/발견/실행 중앙 관리
- Tool Types: 도구 정의 스키마 + 실행 인터페이스
- Critic Pattern: 도구 실행 전 cheap 모델 기반 안전성 판단
- MCP Integration: Phase 3에서 구현 예정
"""

from aria.tools.critic_types import (
    CriticConfig,
    CriticJudgment,
    SafetyLevel,
    ToolAction,
)
from aria.tools.critic import CriticEvaluator
from aria.tools.tool_types import (
    SafetyLevelHint,
    ToolCategory,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolResult,
)
from aria.tools.tool_registry import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolParameterError,
    ToolRegistry,
)
from aria.core.exceptions import ToolExecutionBlockedError

__all__ = [
    # Critic
    "CriticConfig",
    "CriticEvaluator",
    "CriticJudgment",
    "SafetyLevel",
    "ToolAction",
    "ToolExecutionBlockedError",
    # Tool Types
    "SafetyLevelHint",
    "ToolCategory",
    "ToolDefinition",
    "ToolExecutor",
    "ToolParameter",
    "ToolResult",
    # Tool Registry
    "ToolAlreadyRegisteredError",
    "ToolNotFoundError",
    "ToolParameterError",
    "ToolRegistry",
]
