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
