"""ARIA Engine - Tool Types

도구 등록/발견/실행을 위한 타입 정의
- ToolDefinition: 도구 스펙 (이름 / 설명 / 파라미터 JSON Schema / 안전성 힌트)
- ToolResult: 실행 결과 (성공 여부 / 출력 / 에러 / 지연시간)
- ToolExecutor: 도구 실행 추상 인터페이스 (ABC)
- ToolCategory: 도구 분류 (builtin / mcp / custom)

LiteLLM function calling 포맷 호환:
    ToolDefinition.to_llm_tool() → {"type": "function", "function": {...}}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ToolCategory(str, Enum):
    """도구 분류"""
    BUILTIN = "builtin"       # ARIA 내장 도구 (memory_write / knowledge_search 등)
    MCP = "mcp"               # 외부 MCP 서비스 연결 (Supabase / Notion 등)
    CUSTOM = "custom"         # 사용자 정의 도구


class SafetyLevelHint(str, Enum):
    """도구의 기본 안전성 힌트 (Critic 판단 참고용)

    실제 판단은 Critic이 컨텍스트 기반으로 수행하지만
    도구 등록 시 기본 성향을 힌트로 제공하여 판단 정확도를 높임
    """
    READ_ONLY = "read_only"         # 읽기 전용 → 대부분 SAFE
    WRITE = "write"                 # 쓰기 → NEEDS_CONFIRMATION 가능성
    DESTRUCTIVE = "destructive"     # 삭제/비가역 → NEEDS_CONFIRMATION~UNSAFE
    EXTERNAL = "external"           # 외부 전송 (이메일/메시지) → NEEDS_CONFIRMATION


class ToolParameter(BaseModel):
    """도구 파라미터 개별 정의"""
    name: str = Field(..., min_length=1, max_length=100, description="파라미터 이름")
    type: str = Field(
        default="string",
        description="JSON Schema 타입 (string / number / integer / boolean / array / object)",
    )
    description: str = Field(default="", max_length=500, description="파라미터 설명")
    required: bool = Field(default=False, description="필수 여부")
    enum: list[str] | None = Field(default=None, description="허용 값 목록")
    default: Any = Field(default=None, description="기본값")


class ToolDefinition(BaseModel):
    """도구 스펙 정의

    LiteLLM function calling과 Critic 평가에 모두 사용
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="도구 이름 (snake_case / 영문소문자+숫자+언더스코어)",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="도구 설명 (LLM이 도구 선택 시 참고)",
    )
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="도구 파라미터 목록",
    )
    category: ToolCategory = Field(
        default=ToolCategory.BUILTIN,
        description="도구 분류",
    )
    safety_hint: SafetyLevelHint = Field(
        default=SafetyLevelHint.READ_ONLY,
        description="기본 안전성 힌트 (Critic 판단 참고용)",
    )
    version: str = Field(default="1.0.0", description="도구 버전")
    enabled: bool = Field(default=True, description="활성화 여부")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """도구 이름 검증 — 예약어 충돌 방지"""
        reserved = {"__init__", "__del__", "self", "cls", "none", "true", "false"}
        if v.lower() in reserved:
            msg = f"예약어는 도구 이름으로 사용할 수 없습니다: {v}"
            raise ValueError(msg)
        return v

    def to_llm_tool(self) -> dict[str, Any]:
        """LiteLLM function calling 포맷으로 변환

        Returns:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        # JSON Schema properties 구성
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required_params.append(param.name)

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required_params:
            parameters_schema["required"] = required_params

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }


class ToolResult(BaseModel):
    """도구 실행 결과"""
    tool_name: str = Field(..., description="실행된 도구 이름")
    success: bool = Field(..., description="성공 여부")
    output: Any = Field(default=None, description="실행 결과 (성공 시)")
    error: str = Field(default="", description="에러 메시지 (실패 시)")
    pending_confirmation: bool = Field(
        default=False,
        description="사용자 확인 대기 중 (NEEDS_CONFIRMATION)",
    )
    confirmation_id: str = Field(
        default="",
        description="확인 대기 ID (HITL 추적용)",
    )
    latency_ms: float = Field(default=0.0, description="실행 소요 시간 (ms)")
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_observation(self) -> str:
        """ReAct 에이전트의 Observe 단계에 전달할 텍스트 변환

        에이전트가 도구 결과를 이해하고 다음 행동을 결정하는 데 사용
        """
        if self.pending_confirmation:
            return (
                f"[도구: {self.tool_name}] 사용자 확인 대기 중 — "
                f"확인 ID: {self.confirmation_id}"
            )
        if self.success:
            output_str = str(self.output) if self.output is not None else "(빈 결과)"
            # 너무 긴 출력은 잘라서 전달 (토큰 절약)
            if len(output_str) > 3000:
                output_str = output_str[:3000] + "\n... (결과 잘림)"
            return f"[도구: {self.tool_name}] 성공\n{output_str}"
        return f"[도구: {self.tool_name}] 실패 — {self.error}"


class ToolExecutor(ABC):
    """도구 실행 추상 인터페이스

    모든 도구(내장/MCP/커스텀)는 이 인터페이스를 구현해야 함

    사용법:
        class MyTool(ToolExecutor):
            async def execute(self, parameters: dict[str, Any]) -> ToolResult:
                result = do_something(parameters)
                return ToolResult(tool_name="my_tool", success=True, output=result)
    """

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """도구 실행

        Args:
            parameters: 도구 파라미터 (ToolDefinition.parameters 스키마에 맞는 값)

        Returns:
            ToolResult: 실행 결과

        Note:
            - 구현체는 내부에서 예외를 직접 처리하고 ToolResult로 반환해야 함
            - 예외가 밖으로 전파되면 ToolRegistry가 catch하여 실패 ToolResult 생성
            - parameters 검증은 ToolRegistry가 실행 전에 수행
        """
        ...

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """이 executor가 실행하는 도구의 정의 반환

        ToolRegistry.register_executor()로 등록 시
        definition을 별도로 전달하지 않아도 되도록
        executor 자체가 정의를 포함
        """
        ...
