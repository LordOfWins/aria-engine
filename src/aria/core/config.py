"""ARIA Engine - Configuration Management

pydantic-settings 기반 환경변수 관리
모든 설정은 .env 파일 또는 환경변수에서 로드
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMConfig(BaseSettings):
    """LLM Provider 설정"""

    model_config = SettingsConfigDict(env_prefix="ARIA_", env_file=".env", extra="ignore")

    default_model: str = Field(default="claude-sonnet-4-20250514", description="기본 LLM 모델")
    fallback_model: str = Field(default="gpt-4o", description="장애 시 대체 모델")
    cheap_model: str = Field(default="deepseek/deepseek-chat", description="저비용 작업용 모델")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="로컬 임베딩 모델")
    max_tokens_per_request: int = Field(default=4096, description="요청당 최대 토큰")


class QdrantConfig(BaseSettings):
    """Qdrant Vector DB 설정"""

    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    api_key: str = Field(default="")
    url: str = Field(default="", description="Qdrant Cloud URL (설정 시 host/port 무시)")


class CostControlConfig(BaseSettings):
    """비용 제어 (KillSwitch) 설정"""

    model_config = SettingsConfigDict(env_prefix="ARIA_", env_file=".env", extra="ignore")

    daily_cost_limit_usd: float = Field(default=10.0, description="일일 API 비용 상한 (USD)")
    monthly_cost_limit_usd: float = Field(default=300.0, description="월간 API 비용 상한 (USD)")


class APIConfig(BaseSettings):
    """FastAPI 서버 설정"""

    model_config = SettingsConfigDict(env_prefix="ARIA_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8100)
    env: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: str = Field(default="INFO")
    api_key: str = Field(default="aria-dev-key-change-me", description="API 인증 키")
    auth_disabled: bool = Field(
        default=False,
        description="True면 API 인증 스킵 (development 로컬 테스트용 / production에서는 무시됨)",
    )
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10, description="버스트 허용 횟수 (짧은 시간 집중 요청)")

    @model_validator(mode="after")
    def _enforce_production_auth(self) -> "APIConfig":
        """production/staging에서는 인증 스킵 불가 + 기본 키 사용 금지"""
        if self.env in (Environment.PRODUCTION, Environment.STAGING):
            if self.auth_disabled:
                raise ValueError(
                    f"ARIA_AUTH_DISABLED=true는 {self.env.value} 환경에서 사용할 수 없습니다"
                )
            if self.api_key == "aria-dev-key-change-me":
                raise ValueError(
                    f"기본 API 키를 {self.env.value} 환경에서 사용할 수 없습니다. "
                    "ARIA_API_KEY를 변경하세요"
                )
        return self


class AriaConfig(BaseSettings):
    """ARIA 통합 설정"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Provider API Keys (LiteLLM이 자동 참조)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")


@lru_cache()
def get_config() -> AriaConfig:
    """싱글톤 설정 인스턴스 반환"""
    return AriaConfig()
