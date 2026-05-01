"""ARIA Engine - Critic Pattern Types

도구 실행 전 안전성 판단을 위한 타입 정의
- ToolAction: 실행 대상 도구 액션 표현
- CriticJudgment: Critic 판단 결과 (SAFE / NEEDS_CONFIRMATION / UNSAFE)
- CriticConfig: Critic 설정 (활성화 여부 / 허용 목록 등)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SafetyLevel(str, Enum):
    """Critic 안전성 판단 등급

    SAFE: 즉시 실행 가능
    NEEDS_CONFIRMATION: 사용자 확인 필요 (부작용 가능성)
    UNSAFE: 실행 차단 (위험 행위)
    """
    SAFE = "safe"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNSAFE = "unsafe"


class ToolAction(BaseModel):
    """실행 대상 도구 액션

    Critic에게 전달되어 안전성 평가를 받는 단위
    """
    tool_name: str = Field(..., min_length=1, max_length=200, description="도구 이름")
    action: str = Field(..., min_length=1, max_length=500, description="실행할 동작 설명")
    parameters: dict = Field(default_factory=dict, description="도구 파라미터")
    context: str = Field(default="", max_length=5000, description="실행 컨텍스트 (대화 이력 요약)")


class CriticJudgment(BaseModel):
    """Critic 판단 결과

    cheap 모델(Haiku)이 도구 액션의 안전성을 평가한 결과
    """
    safety_level: SafetyLevel = Field(..., description="안전성 등급")
    reason: str = Field(..., min_length=1, max_length=1000, description="판단 근거")
    risk_factors: list[str] = Field(default_factory=list, description="식별된 리스크 요소")
    suggested_mitigation: str = Field(default="", description="위험 완화 제안 (NEEDS_CONFIRMATION/UNSAFE 시)")
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str = Field(default="", description="판단에 사용된 모델")
    latency_ms: float = Field(default=0.0, description="Critic 평가 소요 시간")


class CriticConfig(BaseModel):
    """Critic 설정"""
    enabled: bool = Field(default=True, description="Critic 활성화 여부")
    bypass_tools: list[str] = Field(
        default_factory=list,
        description="Critic 건너뛰기 허용 도구 목록 (예: health_check / read_only 도구)",
    )
    block_on_unsafe: bool = Field(
        default=True,
        description="UNSAFE 판단 시 실행 차단 (False면 경고만)",
    )
    require_confirmation_on_needs_confirm: bool = Field(
        default=True,
        description="NEEDS_CONFIRMATION 시 사용자 확인 요구",
    )
