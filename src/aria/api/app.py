"""ARIA Engine - FastAPI Application

REST API 엔드포인트
- POST /v1/query → 에이전트에게 질문
- POST /v1/knowledge → 지식 추가
- POST /v1/knowledge/{collection}/search → 벡터 검색
- GET /v1/health → 헬스 체크
- GET /v1/cost → 비용 현황
- 글로벌 에러 핸들러 → 구조화된 에러 응답
- Rate limiting → 요청 제한
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from aria.core.config import get_config, AriaConfig
from aria.core.exceptions import (
    AriaError,
    KillSwitchError,
    LLMAllProvidersExhaustedError,
    LLMProviderError,
    CollectionNotFoundError,
    VectorStoreError,
    AgentError,
)
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
    version="0.2.0",
    lifespan=lifespan,
)


# === Rate Limiter (in-memory / 단일 인스턴스용) ===
class RateLimiter:
    """간단한 인메모리 rate limiter (sliding window)"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # 만료된 요청 제거
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# === Global Exception Handlers ===
@app.exception_handler(KillSwitchError)
async def killswitch_handler(request: Request, exc: KillSwitchError) -> JSONResponse:
    logger.error("killswitch_triggered_http", path=request.url.path, details=exc.details)
    return JSONResponse(
        status_code=429,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(LLMAllProvidersExhaustedError)
async def all_providers_exhausted_handler(request: Request, exc: LLMAllProvidersExhaustedError) -> JSONResponse:
    logger.error("all_providers_exhausted_http", path=request.url.path, attempts=exc.details.get("attempts"))
    return JSONResponse(
        status_code=502,
        content={
            "error": exc.code,
            "message": "모든 AI 모델에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.",
            "details": {"attempts": exc.details.get("attempts", [])},
        },
    )


@app.exception_handler(CollectionNotFoundError)
async def collection_not_found_handler(request: Request, exc: CollectionNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(VectorStoreError)
async def vector_store_handler(request: Request, exc: VectorStoreError) -> JSONResponse:
    logger.error("vector_store_error_http", path=request.url.path, error=exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.code,
            "message": exc.message,
        },
    )


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    logger.error("agent_error_http", path=request.url.path, error=exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.code,
            "message": "에이전트 처리 중 오류가 발생했습니다.",
            "details": exc.details,
        },
    )


@app.exception_handler(AriaError)
async def aria_error_handler(request: Request, exc: AriaError) -> JSONResponse:
    logger.error("aria_error_http", path=request.url.path, code=exc.code, error=exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.code,
            "message": exc.message,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """예상 못한 에러 → 500 + 내부 정보 노출 방지"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc)[:500],
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "내부 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        },
    )


# === API Key Authentication ===
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    config = get_config()
    if config.api.env.value == "development":
        client_id = "dev"
    else:
        if not api_key or api_key != config.api.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        client_id = api_key[:8]  # API 키 앞 8자를 client_id로 사용

    # Rate limit 체크
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="요청 제한을 초과했습니다. 잠시 후 다시 시도해주세요.",
        )

    return client_id


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
    text: str = Field(..., min_length=1, max_length=50000, description="문서 텍스트")
    metadata: dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class KnowledgeAddRequest(BaseModel):
    collection: str = Field(..., min_length=1, max_length=100, description="컬렉션 이름")
    documents: list[DocumentInput] = Field(..., min_length=1, max_length=100, description="추가할 문서 목록")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# === Endpoints ===
@app.get("/v1/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "engine": "ARIA", "version": "0.2.0"}


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "KillSwitch 발동 또는 Rate limit"},
        502: {"model": ErrorResponse, "description": "모든 LLM 프로바이더 실패"},
    },
)
async def query_agent(
    request: QueryRequest,
    _client_id: str = Depends(verify_api_key),
) -> QueryResponse:
    """ARIA 에이전트에게 질문"""
    if react_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start_time = time.time()

    # KillSwitch / LLMAllProvidersExhausted / AgentError는
    # 글로벌 핸들러가 처리 → 여기서 별도 catch 불필요
    result = await react_agent.run(
        query=request.query,
        collection=request.collection,
        context=request.context if request.context else None,
    )

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
    _client_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """지식 베이스에 문서 추가"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    documents = [{"text": doc.text, "metadata": doc.metadata} for doc in request.documents]
    count = vector_store.add_documents(request.collection, documents)

    return {"status": "ok", "collection": request.collection, "documents_added": count}


@app.post(
    "/v1/knowledge/{collection}/search",
    responses={404: {"model": ErrorResponse, "description": "컬렉션 미존재"}},
)
async def search_knowledge(
    collection: str,
    request: SearchRequest,
    _client_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """벡터 검색"""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    # CollectionNotFoundError / VectorStoreError는 글로벌 핸들러가 처리
    results = vector_store.search(
        collection_name=collection,
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    return {"collection": collection, "query": request.query, "results": results}


@app.get("/v1/cost")
async def get_cost(
    _client_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """현재 비용 현황"""
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    return llm_provider.get_cost_summary()


@app.get("/v1/collections")
async def list_collections(
    _client_id: str = Depends(verify_api_key),
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
