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
