"""ARIA Engine - Custom Exceptions

모든 ARIA 에러를 구조화하여 일관된 에러 응답 제공
- AriaError: 베이스 예외
- KillSwitchError: 비용 상한 초과
- LLMProviderError: LLM 호출 실패 (모든 fallback 소진)
- VectorStoreError: 벡터DB 관련 에러
- AgentError: 에이전트 실행 에러
"""

from __future__ import annotations


class AriaError(Exception):
    """ARIA 엔진 베이스 예외"""

    def __init__(self, message: str, *, code: str = "ARIA_ERROR", details: dict | None = None) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class KillSwitchError(AriaError):
    """비용 상한 초과 시 발생"""

    def __init__(self, message: str, *, daily_cost: float = 0.0, monthly_cost: float = 0.0) -> None:
        super().__init__(
            message,
            code="KILLSWITCH_TRIGGERED",
            details={"daily_cost_usd": daily_cost, "monthly_cost_usd": monthly_cost},
        )


class LLMProviderError(AriaError):
    """LLM 호출 실패 (모든 fallback 소진 포함)"""

    def __init__(self, message: str, *, model: str = "", attempts: list[dict] | None = None) -> None:
        super().__init__(
            message,
            code="LLM_PROVIDER_ERROR",
            details={"model": model, "attempts": attempts or []},
        )


class LLMAllProvidersExhaustedError(LLMProviderError):
    """모든 LLM 프로바이더/모델이 실패"""

    def __init__(self, attempts: list[dict]) -> None:
        models_tried = [a.get("model", "unknown") for a in attempts]
        super().__init__(
            f"모든 LLM 프로바이더 실패: {', '.join(models_tried)}",
            attempts=attempts,
        )
        self.code = "ALL_PROVIDERS_EXHAUSTED"


class VectorStoreError(AriaError):
    """벡터DB 관련 에러"""

    def __init__(self, message: str, *, collection: str = "") -> None:
        super().__init__(
            message,
            code="VECTOR_STORE_ERROR",
            details={"collection": collection},
        )


class CollectionNotFoundError(VectorStoreError):
    """컬렉션 미존재"""

    def __init__(self, collection: str) -> None:
        super().__init__(f"컬렉션을 찾을 수 없습니다: {collection}", collection=collection)
        self.code = "COLLECTION_NOT_FOUND"


class AgentError(AriaError):
    """에이전트 실행 에러"""

    def __init__(self, message: str, *, query: str = "", iteration: int = 0) -> None:
        super().__init__(
            message,
            code="AGENT_ERROR",
            details={"query": query[:200], "iteration": iteration},
        )
