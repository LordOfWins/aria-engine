#!/bin/bash
# ============================================================
# ARIA Engine - Agentic Reasoning and Information Access
# Project Setup Script
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "🚀 ARIA Engine 프로젝트 세팅 시작..."
echo "📁 프로젝트 루트: $PROJECT_ROOT"

# ============================================================
# 1. 디렉토리 구조 생성
# ============================================================
echo ""
echo "📂 디렉토리 구조 생성..."

mkdir -p src/aria/{core,providers,rag,agents,tools,memory,api}
mkdir -p src/aria/core
mkdir -p src/aria/providers
mkdir -p src/aria/rag
mkdir -p src/aria/agents
mkdir -p src/aria/tools
mkdir -p src/aria/memory
mkdir -p src/aria/api
mkdir -p tests/{unit,integration,e2e}
mkdir -p configs
mkdir -p scripts
mkdir -p docs
mkdir -p data/knowledge_base

# __init__.py 파일 생성
touch src/__init__.py
touch src/aria/__init__.py
touch src/aria/core/__init__.py
touch src/aria/providers/__init__.py
touch src/aria/rag/__init__.py
touch src/aria/agents/__init__.py
touch src/aria/tools/__init__.py
touch src/aria/memory/__init__.py
touch src/aria/api/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/e2e/__init__.py

echo "✅ 디렉토리 구조 완료"

# ============================================================
# 2. pyproject.toml 생성
# ============================================================
echo ""
echo "📦 pyproject.toml 생성..."

cat > pyproject.toml << 'PYPROJECT'
[project]
name = "aria-engine"
version = "0.1.0"
description = "ARIA - Agentic Reasoning and Information Access Engine"
requires-python = ">=3.11"
license = {text = "PROPRIETARY"}

dependencies = [
    # === LLM Provider Abstraction ===
    "litellm>=1.55.0",

    # === Agent Orchestration ===
    "langgraph>=0.4.0",
    "langchain-core>=0.3.0",

    # === Vector Database ===
    "qdrant-client>=1.12.0",

    # === Local Embeddings (no API cost) ===
    "fastembed>=0.4.0",

    # === API Framework ===
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",

    # === Configuration & Environment ===
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "python-dotenv>=1.0.0",

    # === HTTP Client ===
    "httpx>=0.28.0",

    # === Logging & Observability ===
    "structlog>=24.4.0",

    # === Utilities ===
    "tiktoken>=0.8.0",
    "tenacity>=9.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "httpx>=0.28.0",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.11"
strict = true
PYPROJECT

echo "✅ pyproject.toml 완료"

# ============================================================
# 3. 환경변수 템플릿 생성
# ============================================================
echo ""
echo "🔐 환경변수 파일 생성..."

cat > .env.example << 'ENVFILE'
# ============================================================
# ARIA Engine - Environment Variables
# ============================================================
# 이 파일을 .env로 복사한 후 실제 키 값을 입력하세요
# cp .env.example .env

# === LLM Provider API Keys ===
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
GOOGLE_API_KEY=xxxxx
DEEPSEEK_API_KEY=xxxxx

# === Default LLM Configuration ===
ARIA_DEFAULT_MODEL=claude-sonnet-4-20250514
ARIA_FALLBACK_MODEL=gpt-4o
ARIA_CHEAP_MODEL=deepseek/deepseek-chat
ARIA_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# === Qdrant Vector Database ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
# For Qdrant Cloud: QDRANT_URL=https://xxx.qdrant.io

# === API Server ===
ARIA_HOST=0.0.0.0
ARIA_PORT=8100
ARIA_ENV=development
ARIA_LOG_LEVEL=INFO
ARIA_API_KEY=aria-dev-key-change-me

# === Rate Limiting ===
ARIA_RATE_LIMIT_PER_MINUTE=60
ARIA_MAX_TOKENS_PER_REQUEST=4096

# === Cost Control (KillSwitch) ===
ARIA_DAILY_COST_LIMIT_USD=10.0
ARIA_MONTHLY_COST_LIMIT_USD=300.0
ENVFILE

# 실제 .env 파일 생성 (gitignore 대상)
cp .env.example .env

echo "✅ 환경변수 파일 완료"

# ============================================================
# 4. Docker Compose (Qdrant)
# ============================================================
echo ""
echo "🐳 Docker Compose 생성..."

cat > docker-compose.yml << 'DOCKERCOMPOSE'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: aria-qdrant
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

volumes:
  qdrant_data:
    driver: local
DOCKERCOMPOSE

echo "✅ Docker Compose 완료"

# ============================================================
# 5. .gitignore
# ============================================================
echo ""
echo "📝 .gitignore 생성..."

cat > .gitignore << 'GITIGNORE'
# Environment
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual Environment
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
htmlcov/
.pytest_cache/

# OS
.DS_Store
Thumbs.db

# Data (large files)
data/knowledge_base/*.bin
data/knowledge_base/*.npy

# Qdrant local storage
qdrant_data/

# Logs
logs/
*.log
GITIGNORE

echo "✅ .gitignore 완료"

# ============================================================
# 6. Core Configuration Module
# ============================================================
echo ""
echo "⚙️ Core 설정 모듈 생성..."

cat > src/aria/core/config.py << 'CONFIGPY'
"""ARIA Engine - Configuration Management

pydantic-settings 기반 환경변수 관리
모든 설정은 .env 파일 또는 환경변수에서 로드
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
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
    rate_limit_per_minute: int = Field(default=60)


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
CONFIGPY

echo "✅ config.py 완료"

# ============================================================
# 7. LLM Provider Abstraction (LiteLLM Wrapper)
# ============================================================
echo ""
echo "🤖 LLM Provider 추상화 레이어 생성..."

cat > src/aria/providers/llm_provider.py << 'LLMPROVIDER'
"""ARIA Engine - LLM Provider Abstraction Layer

LiteLLM 기반 멀티 프로바이더 LLM 호출
- Provider Agnostic: Claude/GPT/Gemini/DeepSeek 한 줄 교체
- Cost Tracking: 요청별 비용 추적
- Fallback: 장애 시 자동 대체 모델 전환
- KillSwitch: 비용 상한 초과 시 자동 차단
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import litellm
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from aria.core.config import AriaConfig, get_config

logger = structlog.get_logger()

# LiteLLM 전역 설정
litellm.drop_params = True  # 지원 안 하는 파라미터 자동 무시
litellm.set_verbose = False


@dataclass
class UsageRecord:
    """단일 API 호출 사용량 기록"""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostTracker:
    """일별/월별 비용 추적기 (KillSwitch 연동)"""

    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    current_date: str = ""
    current_month: str = ""
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        today = date.today()
        today_str = today.isoformat()
        month_str = today.strftime("%Y-%m")

        # 날짜 변경 시 리셋
        if self.current_date != today_str:
            self.daily_cost = 0.0
            self.current_date = today_str

        if self.current_month != month_str:
            self.monthly_cost = 0.0
            self.current_month = month_str

        self.daily_cost += record.cost_usd
        self.monthly_cost += record.cost_usd
        self.records.append(record)

    def check_limits(self, config: AriaConfig) -> tuple[bool, str]:
        """비용 상한 체크 → (허용여부, 사유)"""
        if self.daily_cost >= config.cost_control.daily_cost_limit_usd:
            return False, f"일일 비용 상한 도달: ${self.daily_cost:.2f} / ${config.cost_control.daily_cost_limit_usd}"
        if self.monthly_cost >= config.cost_control.monthly_cost_limit_usd:
            return False, f"월간 비용 상한 도달: ${self.monthly_cost:.2f} / ${config.cost_control.monthly_cost_limit_usd}"
        return True, "OK"


class LLMProvider:
    """LiteLLM 기반 멀티 프로바이더 LLM 클라이언트

    사용법:
        provider = LLMProvider()
        response = await provider.complete("오늘 날씨 어때?")
        response = await provider.complete("복잡한 분석", model_tier="heavy")
        response = await provider.complete("간단한 분류", model_tier="cheap")
    """

    def __init__(self, config: AriaConfig | None = None) -> None:
        self.config = config or get_config()
        self.cost_tracker = CostTracker()

        # 모델 티어 매핑
        self._model_tiers: dict[str, str] = {
            "default": self.config.llm.default_model,
            "heavy": self.config.llm.default_model,  # 복잡한 추론
            "cheap": self.config.llm.cheap_model,     # 분류/태깅/짧은 응답
            "fallback": self.config.llm.fallback_model,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        *,
        model_tier: str = "default",
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """LLM 호출 (자동 비용 추적 + KillSwitch)

        Args:
            prompt: 사용자 프롬프트
            model_tier: "default" | "heavy" | "cheap" | "fallback"
            model: 직접 모델명 지정 (tier보다 우선)
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 출력 토큰
            temperature: 생성 온도
            response_format: 응답 포맷 (JSON mode 등)

        Returns:
            {"content": str, "model": str, "usage": UsageRecord}

        Raises:
            RuntimeError: KillSwitch 발동 시
        """
        # KillSwitch 체크
        allowed, reason = self.cost_tracker.check_limits(self.config)
        if not allowed:
            logger.error("killswitch_triggered", reason=reason)
            raise RuntimeError(f"🛑 ARIA KillSwitch: {reason}")

        # 모델 결정
        target_model = model or self._model_tiers.get(model_tier, self.config.llm.default_model)

        # 메시지 구성
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # LiteLLM 호출
        start_time = time.time()

        try:
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.config.llm.max_tokens_per_request,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await litellm.acompletion(**kwargs)

        except Exception as e:
            logger.warning("llm_call_failed", model=target_model, error=str(e))

            # Fallback 시도
            if target_model != self.config.llm.fallback_model:
                logger.info("fallback_triggered", from_model=target_model, to_model=self.config.llm.fallback_model)
                kwargs["model"] = self.config.llm.fallback_model
                response = await litellm.acompletion(**kwargs)
            else:
                raise

        latency_ms = (time.time() - start_time) * 1000

        # 사용량 기록
        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        record = UsageRecord(
            model=response.model or target_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=latency_ms,
        )
        self.cost_tracker.add(record)

        content = response.choices[0].message.content or ""

        logger.info(
            "llm_call_success",
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost_usd=f"${record.cost_usd:.4f}",
            latency_ms=f"{record.latency_ms:.0f}ms",
        )

        return {
            "content": content,
            "model": record.model,
            "usage": record,
        }

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model_tier: str = "default",
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """멀티턴 대화용 LLM 호출 (messages 직접 전달)"""
        allowed, reason = self.cost_tracker.check_limits(self.config)
        if not allowed:
            raise RuntimeError(f"🛑 ARIA KillSwitch: {reason}")

        target_model = model or self._model_tiers.get(model_tier, self.config.llm.default_model)

        start_time = time.time()
        response = await litellm.acompletion(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.config.llm.max_tokens_per_request,
        )
        latency_ms = (time.time() - start_time) * 1000

        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        record = UsageRecord(
            model=response.model or target_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=latency_ms,
        )
        self.cost_tracker.add(record)

        return {
            "content": response.choices[0].message.content or "",
            "model": record.model,
            "usage": record,
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """현재 비용 요약"""
        return {
            "daily_cost_usd": round(self.cost_tracker.daily_cost, 4),
            "monthly_cost_usd": round(self.cost_tracker.monthly_cost, 4),
            "daily_limit_usd": self.config.cost_control.daily_cost_limit_usd,
            "monthly_limit_usd": self.config.cost_control.monthly_cost_limit_usd,
            "total_requests": len(self.cost_tracker.records),
        }
LLMPROVIDER

echo "✅ llm_provider.py 완료"

# ============================================================
# 8. RAG Pipeline (Qdrant + FastEmbed)
# ============================================================
echo ""
echo "🔍 RAG 파이프라인 생성..."

cat > src/aria/rag/vector_store.py << 'VECTORSTORE'
"""ARIA Engine - Vector Store (Qdrant + FastEmbed)

플랫폼 독립형 벡터 저장소
- 로컬 임베딩 (FastEmbed) → API 비용 0원
- Qdrant 셀프호스팅 → 클라우드 종속 없음
- 컬렉션 기반 지식 분리 → 제품별/도메인별 격리
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import structlog
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from aria.core.config import AriaConfig, get_config

logger = structlog.get_logger()


class VectorStore:
    """Qdrant 기반 벡터 저장소

    사용법:
        store = VectorStore()
        await store.ensure_collection("psychology_kb")
        await store.add_documents("psychology_kb", [
            {"text": "회피형 애착은...", "metadata": {"source": "dsm5", "topic": "attachment"}}
        ])
        results = await store.search("psychology_kb", "회피형 애착 패턴", top_k=5)
    """

    def __init__(self, config: AriaConfig | None = None) -> None:
        self.config = config or get_config()

        # Qdrant 클라이언트 초기화
        if self.config.qdrant.url:
            self._client = QdrantClient(
                url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key or None,
            )
        else:
            self._client = QdrantClient(
                host=self.config.qdrant.host,
                port=self.config.qdrant.port,
            )

        # FastEmbed 로컬 임베딩 모델
        self._embedder = TextEmbedding(model_name=self.config.llm.embedding_model)

        # 임베딩 차원 크기 캐시
        self._vector_size: int | None = None

    def _get_vector_size(self) -> int:
        """임베딩 모델의 벡터 차원 크기 확인"""
        if self._vector_size is None:
            test_embedding = list(self._embedder.embed(["test"]))[0]
            self._vector_size = len(test_embedding)
        return self._vector_size

    def _generate_deterministic_id(self, text: str) -> str:
        """텍스트 기반 결정적 UUID 생성 (중복 방지)"""
        hash_bytes = hashlib.md5(text.encode()).hexdigest()
        return str(uuid.UUID(hash_bytes))

    def ensure_collection(self, collection_name: str) -> None:
        """컬렉션 존재 확인 → 없으면 생성"""
        collections = self._client.get_collections().collections
        existing_names = [c.name for c in collections]

        if collection_name not in existing_names:
            vector_size = self._get_vector_size()
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("collection_created", name=collection_name, vector_size=vector_size)
        else:
            logger.debug("collection_exists", name=collection_name)

    def add_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        """문서를 벡터화하여 저장

        Args:
            collection_name: 컬렉션 이름
            documents: [{"text": str, "metadata": dict}, ...]
            batch_size: 배치 크기

        Returns:
            저장된 문서 수
        """
        self.ensure_collection(collection_name)

        texts = [doc["text"] for doc in documents]
        total_added = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]

            # 로컬 임베딩 생성
            embeddings = list(self._embedder.embed(batch_texts))

            points = [
                models.PointStruct(
                    id=self._generate_deterministic_id(doc["text"]),
                    vector=embedding.tolist(),
                    payload={
                        "text": doc["text"],
                        **(doc.get("metadata", {})),
                    },
                )
                for doc, embedding in zip(batch_docs, embeddings)
            ]

            self._client.upsert(collection_name=collection_name, points=points)
            total_added += len(points)

        logger.info("documents_added", collection=collection_name, count=total_added)
        return total_added

    def search(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """시맨틱 검색

        Args:
            collection_name: 컬렉션 이름
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            score_threshold: 최소 유사도 점수
            filter_conditions: Qdrant 필터 조건

        Returns:
            [{"text": str, "score": float, "metadata": dict}, ...]
        """
        query_embedding = list(self._embedder.embed([query]))[0]

        search_params: dict[str, Any] = {
            "collection_name": collection_name,
            "query_vector": query_embedding.tolist(),
            "limit": top_k,
            "score_threshold": score_threshold,
        }

        if filter_conditions:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        ).points

        output = []
        for point in results:
            payload = point.payload or {}
            text = payload.pop("text", "")
            output.append({
                "text": text,
                "score": point.score,
                "metadata": payload,
            })

        logger.info("search_completed", collection=collection_name, query=query[:50], results=len(output))
        return output

    def delete_collection(self, collection_name: str) -> None:
        """컬렉션 삭제"""
        self._client.delete_collection(collection_name=collection_name)
        logger.info("collection_deleted", name=collection_name)

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """컬렉션 정보 조회"""
        info = self._client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
        }
VECTORSTORE

echo "✅ vector_store.py 완료"

# ============================================================
# 9. ReAct Agent (LangGraph)
# ============================================================
echo ""
echo "🧠 ReAct 에이전트 생성..."

cat > src/aria/agents/react_agent.py << 'REACTAGENT'
"""ARIA Engine - ReAct Agent (LangGraph)

Think → Act → Observe 루프 기반 자율 추론 에이전트
- 질문 분석 → 검색 전략 결정 → 실행 → 결과 검증 → 답변
- Self-Reflection 노드로 답변 품질 자체 평가
- 최대 반복 횟수 제한으로 무한루프 방지
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated

import structlog
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from aria.providers.llm_provider import LLMProvider
from aria.rag.vector_store import VectorStore

logger = structlog.get_logger()

MAX_ITERATIONS = 5  # 최대 ReAct 루프 반복 횟수


class AgentAction(str, Enum):
    """에이전트가 취할 수 있는 행동"""
    SEARCH_KNOWLEDGE = "search_knowledge"      # 벡터DB 검색
    SEARCH_WEB = "search_web"                  # 웹 검색 (추후 구현)
    REASON = "reason"                          # 추가 추론
    RESPOND = "respond"                        # 최종 답변
    CLARIFY = "clarify"                        # 사용자에게 명확화 요청


@dataclass
class AgentState:
    """에이전트 상태 (LangGraph State)"""
    messages: Annotated[list[dict[str, str]], add_messages] = field(default_factory=list)
    query: str = ""
    intent: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    current_answer: str = ""
    confidence: float = 0.0
    iteration: int = 0
    should_stop: bool = False


INTENT_ANALYSIS_PROMPT = """당신은 사용자 의도 분석 전문가입니다.

사용자 질문을 분석하여 다음 JSON 형태로 응답하세요:
{{
    "surface_intent": "표면적 질문의 핵심",
    "deeper_intent": "질문 뒤에 숨겨진 실제 의도/욕구",
    "required_knowledge": ["필요한 지식 영역 목록"],
    "search_queries": ["벡터DB 검색에 사용할 쿼리 목록 (최대 3개)"],
    "complexity": "simple|moderate|complex",
    "recommended_action": "search_knowledge|reason|respond|clarify"
}}

사용자 질문: {query}"""


REASONING_PROMPT = """당신은 논리적 추론 전문가입니다.

주어진 정보를 바탕으로 단계적으로 사고하여 답변을 도출하세요.

## 사용자 질문
{query}

## 의도 분석
{intent}

## 검색된 관련 정보
{search_results}

## 이전 추론 단계
{previous_reasoning}

다음 형식으로 응답하세요:
1. 현재까지 알고 있는 것을 정리
2. 부족한 정보가 있다면 무엇인지
3. 논리적 추론 과정
4. 결론 및 답변
5. 확신도 (0.0 ~ 1.0)"""


SELF_REFLECTION_PROMPT = """당신은 답변 품질 평가 전문가입니다.

다음 답변의 품질을 평가하세요:

## 원래 질문
{query}

## 사용자의 실제 의도
{intent}

## 현재 답변
{answer}

평가 기준:
1. 질문에 대한 직접적 답변인가?
2. 근거가 충분한가?
3. 논리적 오류가 없는가?
4. 사용자의 숨겨진 의도도 충족하는가?

JSON으로 응답하세요:
{{
    "quality_score": 0.0~1.0,
    "issues": ["발견된 문제점"],
    "should_retry": true/false,
    "improvement_suggestion": "개선 방향"
}}"""


class ReActAgent:
    """LangGraph 기반 ReAct 에이전트

    사용법:
        agent = ReActAgent(llm_provider, vector_store)
        result = await agent.run("회피형 애착 패턴에 대해 설명해줘", collection="psychology_kb")
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStore,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)

        # 노드 등록
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("reason", self._reason)
        workflow.add_node("self_reflect", self._self_reflect)
        workflow.add_node("respond", self._respond)

        # 엣지 연결
        workflow.set_entry_point("analyze_intent")
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_after_intent,
            {
                "search_knowledge": "search_knowledge",
                "reason": "reason",
                "respond": "respond",
            },
        )
        workflow.add_edge("search_knowledge", "reason")
        workflow.add_edge("reason", "self_reflect")
        workflow.add_conditional_edges(
            "self_reflect",
            self._route_after_reflection,
            {
                "retry": "search_knowledge",
                "respond": "respond",
            },
        )
        workflow.add_edge("respond", END)

        return workflow.compile()

    async def _analyze_intent(self, state: AgentState) -> dict[str, Any]:
        """Step 1: 사용자 의도 분석"""
        prompt = INTENT_ANALYSIS_PROMPT.format(query=state.query)

        result = await self.llm.complete(
            prompt,
            model_tier="cheap",  # 의도 분석은 저비용 모델로 충분
            temperature=0.3,
        )

        # JSON 파싱 시도
        import json
        try:
            content = result["content"]
            # JSON 블록 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            intent = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            intent = {
                "surface_intent": state.query,
                "deeper_intent": "",
                "required_knowledge": [],
                "search_queries": [state.query],
                "complexity": "moderate",
                "recommended_action": "search_knowledge",
            }

        logger.info("intent_analyzed", intent=intent)
        return {"intent": intent}

    def _route_after_intent(self, state: AgentState) -> str:
        """의도 분석 후 라우팅 결정"""
        action = state.intent.get("recommended_action", "search_knowledge")
        complexity = state.intent.get("complexity", "moderate")

        if action == "respond" and complexity == "simple":
            return "respond"
        if action == "clarify":
            return "respond"  # 명확화 요청도 응답으로 처리
        return "search_knowledge"

    async def _search_knowledge(self, state: AgentState) -> dict[str, Any]:
        """Step 2: 지식 검색 (RAG)"""
        queries = state.intent.get("search_queries", [state.query])
        all_results: list[dict[str, Any]] = []

        for query in queries[:3]:  # 최대 3개 쿼리
            try:
                results = self.vector_store.search(
                    collection_name=self._collection,
                    query=query,
                    top_k=3,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning("search_failed", query=query, error=str(e))

        # 중복 제거 + 점수순 정렬
        seen_texts: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            text_hash = r["text"][:100]
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(r)

        return {"search_results": unique_results[:10]}

    async def _reason(self, state: AgentState) -> dict[str, Any]:
        """Step 3: 논리적 추론"""
        search_context = "\n\n".join(
            f"[관련도: {r['score']:.2f}] {r['text']}"
            for r in state.search_results
        ) if state.search_results else "검색 결과 없음"

        previous = "\n".join(state.reasoning_steps) if state.reasoning_steps else "첫 번째 추론 단계"

        prompt = REASONING_PROMPT.format(
            query=state.query,
            intent=str(state.intent),
            search_results=search_context,
            previous_reasoning=previous,
        )

        result = await self.llm.complete(
            prompt,
            model_tier="default",
            temperature=0.5,
        )

        new_steps = state.reasoning_steps + [f"[Iteration {state.iteration + 1}] {result['content'][:500]}"]

        return {
            "current_answer": result["content"],
            "reasoning_steps": new_steps,
            "iteration": state.iteration + 1,
        }

    async def _self_reflect(self, state: AgentState) -> dict[str, Any]:
        """Step 4: 자기 성찰 - 답변 품질 평가"""
        prompt = SELF_REFLECTION_PROMPT.format(
            query=state.query,
            intent=str(state.intent),
            answer=state.current_answer,
        )

        result = await self.llm.complete(
            prompt,
            model_tier="cheap",
            temperature=0.2,
        )

        import json
        try:
            content = result["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            reflection = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            reflection = {"quality_score": 0.7, "should_retry": False}

        confidence = reflection.get("quality_score", 0.7)
        should_retry = reflection.get("should_retry", False)

        # 최대 반복 횟수 도달 시 강제 종료
        if state.iteration >= MAX_ITERATIONS:
            should_retry = False

        return {
            "confidence": confidence,
            "should_stop": not should_retry,
        }

    def _route_after_reflection(self, state: AgentState) -> str:
        """자기 성찰 후 라우팅"""
        if state.should_stop or state.iteration >= MAX_ITERATIONS:
            return "respond"
        return "retry"

    async def _respond(self, state: AgentState) -> dict[str, Any]:
        """Step 5: 최종 답변 생성"""
        return {
            "messages": [{"role": "assistant", "content": state.current_answer}],
        }

    async def run(
        self,
        query: str,
        collection: str = "default",
        context: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """에이전트 실행

        Args:
            query: 사용자 질문
            collection: 검색 대상 벡터DB 컬렉션
            context: 이전 대화 이력

        Returns:
            {"answer": str, "confidence": float, "reasoning_steps": list, "usage_summary": dict}
        """
        self._collection = collection

        initial_state = AgentState(
            query=query,
            messages=context or [],
        )

        # LangGraph 실행
        final_state = await self._graph.ainvoke(initial_state)

        return {
            "answer": final_state.get("current_answer", ""),
            "confidence": final_state.get("confidence", 0.0),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "iterations": final_state.get("iteration", 0),
            "search_results_count": len(final_state.get("search_results", [])),
            "cost_summary": self.llm.get_cost_summary(),
        }
REACTAGENT

echo "✅ react_agent.py 완료"

# ============================================================
# 10. FastAPI Application
# ============================================================
echo ""
echo "🌐 FastAPI 애플리케이션 생성..."

cat > src/aria/api/app.py << 'FASTAPIAPP'
"""ARIA Engine - FastAPI Application

REST API 엔드포인트
- POST /v1/query → 에이전트에게 질문
- POST /v1/knowledge → 지식 추가
- GET /v1/knowledge/{collection}/search → 벡터 검색
- GET /v1/health → 헬스 체크
- GET /v1/cost → 비용 현황
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from aria.core.config import get_config, AriaConfig
from aria.providers.llm_provider import LLMProvider
from aria.rag.vector_store import VectorStore
from aria.agents.react_agent import ReActAgent

logger = structlog.get_logger()

# === Global Instances ===
llm_provider: LLMProvider | None = None
vector_store: VectorStore | None = None
react_agent: ReActAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 시작/종료 시 초기화"""
    global llm_provider, vector_store, react_agent

    config = get_config()
    llm_provider = LLMProvider(config)
    vector_store = VectorStore(config)
    react_agent = ReActAgent(llm_provider, vector_store)

    logger.info("aria_engine_started", env=config.api.env.value)
    yield
    logger.info("aria_engine_stopped")


app = FastAPI(
    title="ARIA Engine",
    description="Agentic Reasoning and Information Access Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API Key Authentication ===
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    config = get_config()
    if config.api.env.value == "development":
        return "dev"  # 개발 환경에서는 인증 스킵
    if not api_key or api_key != config.api.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# === Request/Response Models ===
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="질문")
    collection: str = Field(default="default", description="검색 대상 컬렉션")
    model_tier: str = Field(default="default", description="모델 티어: default|heavy|cheap")
    context: list[dict[str, str]] = Field(default_factory=list, description="이전 대화 이력")


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    iterations: int
    reasoning_steps: list[str]
    cost_summary: dict[str, Any]
    latency_ms: float


class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, description="문서 텍스트")
    metadata: dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class KnowledgeAddRequest(BaseModel):
    collection: str = Field(..., description="컬렉션 이름")
    documents: list[DocumentInput] = Field(..., min_length=1, description="추가할 문서 목록")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# === Endpoints ===
@app.get("/v1/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "engine": "ARIA", "version": "0.1.0"}


@app.post("/v1/query", response_model=QueryResponse)
async def query_agent(
    request: QueryRequest,
    _api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """ARIA 에이전트에게 질문"""
    if react_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start_time = time.time()

    try:
        result = await react_agent.run(
            query=request.query,
            collection=request.collection,
            context=request.context if request.context else None,
        )
    except RuntimeError as e:
        if "KillSwitch" in str(e):
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        iterations=result["iterations"],
        reasoning_steps=result["reasoning_steps"],
        cost_summary=result["cost_summary"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/v1/knowledge")
async def add_knowledge(
    request: KnowledgeAddRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """지식 베이스에 문서 추가"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    documents = [{"text": doc.text, "metadata": doc.metadata} for doc in request.documents]
    count = vector_store.add_documents(request.collection, documents)

    return {"status": "ok", "collection": request.collection, "documents_added": count}


@app.post("/v1/knowledge/{collection}/search")
async def search_knowledge(
    collection: str,
    request: SearchRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """벡터 검색"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        results = vector_store.search(
            collection_name=collection,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {collection}")

    return {"collection": collection, "query": request.query, "results": results}


@app.get("/v1/cost")
async def get_cost(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """현재 비용 현황"""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    return llm_provider.get_cost_summary()


@app.get("/v1/collections")
async def list_collections(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """벡터DB 컬렉션 목록"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    collections = vector_store._client.get_collections().collections
    return {
        "collections": [
            {"name": c.name}
            for c in collections
        ]
    }
FASTAPIAPP

echo "✅ FastAPI app.py 완료"

# ============================================================
# 11. 서버 실행 스크립트
# ============================================================
echo ""
echo "🚀 실행 스크립트 생성..."

cat > run.py << 'RUNPY'
"""ARIA Engine - Server Runner"""

import uvicorn
from aria.core.config import get_config


def main() -> None:
    config = get_config()
    uvicorn.run(
        "aria.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.env.value == "development",
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()
RUNPY

echo "✅ run.py 완료"

# ============================================================
# 12. 테스트 파일
# ============================================================
echo ""
echo "🧪 테스트 파일 생성..."

cat > tests/unit/test_llm_provider.py << 'TESTLLM'
"""LLM Provider 단위 테스트"""

import pytest
from aria.core.config import AriaConfig
from aria.providers.llm_provider import LLMProvider, CostTracker, UsageRecord


class TestCostTracker:
    def test_add_record(self):
        tracker = CostTracker()
        record = UsageRecord(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=200.0,
        )
        tracker.add(record)
        assert tracker.daily_cost == 0.001
        assert tracker.monthly_cost == 0.001
        assert len(tracker.records) == 1

    def test_killswitch_daily_limit(self):
        config = AriaConfig()
        config.cost_control.daily_cost_limit_usd = 0.01

        tracker = CostTracker()
        record = UsageRecord(
            model="test", input_tokens=100, output_tokens=50,
            cost_usd=0.02, latency_ms=100.0,
        )
        tracker.add(record)

        allowed, reason = tracker.check_limits(config)
        assert not allowed
        assert "일일 비용 상한" in reason

    def test_killswitch_allows_within_limit(self):
        config = AriaConfig()
        config.cost_control.daily_cost_limit_usd = 10.0

        tracker = CostTracker()
        record = UsageRecord(
            model="test", input_tokens=100, output_tokens=50,
            cost_usd=0.001, latency_ms=100.0,
        )
        tracker.add(record)

        allowed, reason = tracker.check_limits(config)
        assert allowed
        assert reason == "OK"


class TestLLMProvider:
    def test_model_tiers(self):
        provider = LLMProvider()
        assert "default" in provider._model_tiers
        assert "cheap" in provider._model_tiers
        assert "heavy" in provider._model_tiers
        assert "fallback" in provider._model_tiers

    def test_cost_summary(self):
        provider = LLMProvider()
        summary = provider.get_cost_summary()
        assert "daily_cost_usd" in summary
        assert "monthly_cost_usd" in summary
        assert summary["total_requests"] == 0
TESTLLM

cat > tests/unit/test_config.py << 'TESTCONFIG'
"""Configuration 단위 테스트"""

from aria.core.config import get_config, AriaConfig, Environment


class TestConfig:
    def test_default_config(self):
        config = AriaConfig()
        assert config.llm.default_model == "claude-sonnet-4-20250514"
        assert config.api.port == 8100
        assert config.api.env == Environment.DEVELOPMENT

    def test_cost_control_defaults(self):
        config = AriaConfig()
        assert config.cost_control.daily_cost_limit_usd == 10.0
        assert config.cost_control.monthly_cost_limit_usd == 300.0

    def test_qdrant_defaults(self):
        config = AriaConfig()
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
TESTCONFIG

echo "✅ 테스트 파일 완료"

# ============================================================
# 13. README
# ============================================================
echo ""
echo "📖 README 생성..."

cat > README.md << 'README'
# ARIA Engine

> **A**gentic **R**easoning and **I**nformation **A**ccess

범용 AI 추론 엔진. LLM 종속 없이 자율적으로 사고하고 검색하고 판단하는 시스템.

## Architecture

```
ARIA Engine
├── Intent Analyzer     → 다층 의도 분석 (Multi-Level Intent Parsing)
├── Knowledge Router    → 검색 소스 자동 선택 (Vector DB / Web / Tools)
├── ReAct Agent Loop    → Think → Act → Observe → Reflect 반복
├── Provider Abstraction → LiteLLM (Claude/GPT/Gemini/DeepSeek/Local)
└── Vector Store        → Qdrant + FastEmbed (로컬 임베딩)
```

## Quick Start

```bash
# 1. Qdrant 시작
docker compose up -d

# 2. Python 가상환경 세팅
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키 입력

# 4. 서버 시작
python run.py

# 5. 테스트
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | 헬스 체크 |
| POST | `/v1/query` | 에이전트에게 질문 |
| POST | `/v1/knowledge` | 지식 추가 |
| POST | `/v1/knowledge/{collection}/search` | 벡터 검색 |
| GET | `/v1/cost` | 비용 현황 |
| GET | `/v1/collections` | 컬렉션 목록 |

## Design Principles

1. **Provider Agnostic** - LLM/벡터DB 언제든 교체 가능
2. **Product Agnostic** - 어떤 제품이든 REST API로 연결
3. **Cost Aware** - 모든 호출의 비용 추적 + KillSwitch
4. **Solo Operator** - 1인 운영 가능한 복잡도
README

echo "✅ README 완료"

# ============================================================
# 완료
# ============================================================
echo ""
echo "============================================"
echo "🎉 ARIA Engine 프로젝트 세팅 완료!"
echo "============================================"
echo ""
echo "📁 프로젝트 구조:"
find . -type f -not -path './.git/*' -not -path './.venv/*' | sort
echo ""
echo "다음 단계:"
echo "1. docker compose up -d    → Qdrant 시작"
echo "2. python -m venv .venv && source .venv/bin/activate"
echo "3. pip install -e '.[dev]'"
echo "4. .env 파일에 API 키 입력"
echo "5. python run.py           → ARIA 서버 시작"
echo "6. pytest tests/ -v        → 테스트 실행"
