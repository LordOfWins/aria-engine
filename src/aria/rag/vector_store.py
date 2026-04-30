"""ARIA Engine - Vector Store (Qdrant + FastEmbed)

플랫폼 독립형 벡터 저장소
- 로컬 임베딩 (FastEmbed) → API 비용 0원
- Qdrant 셀프호스팅 → 클라우드 종속 없음
- 컬렉션 기반 지식 분리 → 제품별/도메인별 격리
- 구조화된 예외 처리 → CollectionNotFoundError 등
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import structlog
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from aria.core.config import AriaConfig, get_config
from aria.core.exceptions import CollectionNotFoundError, VectorStoreError

logger = structlog.get_logger()


class VectorStore:
    """Qdrant 기반 벡터 저장소

    사용법:
        store = VectorStore()
        store.ensure_collection("psychology_kb")
        store.add_documents("psychology_kb", [
            {"text": "회피형 애착은...", "metadata": {"source": "dsm5", "topic": "attachment"}}
        ])
        results = store.search("psychology_kb", "회피형 애착 패턴", top_k=5)
    """

    def __init__(self, config: AriaConfig | None = None) -> None:
        self.config = config or get_config()

        # Qdrant 클라이언트 초기화
        try:
            if self.config.qdrant.url:
                self._client = QdrantClient(
                    url=self.config.qdrant.url,
                    api_key=self.config.qdrant.api_key or None,
                    timeout=10,
                )
            else:
                self._client = QdrantClient(
                    host=self.config.qdrant.host,
                    port=self.config.qdrant.port,
                    timeout=10,
                )
        except Exception as e:
            raise VectorStoreError(f"Qdrant 연결 실패: {e}") from e

        # FastEmbed 로컬 임베딩 모델
        try:
            self._embedder = TextEmbedding(model_name=self.config.llm.embedding_model)
            logger.info("embedding_model_loaded", model=self.config.llm.embedding_model)
        except ValueError as e:
            raise VectorStoreError(
                f"임베딩 모델 '{self.config.llm.embedding_model}' 로드 실패: {e}. "
                f"지원 모델 확인: TextEmbedding.list_supported_models()"
            ) from e

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

    def _collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        try:
            collections = self._client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            raise VectorStoreError(f"컬렉션 목록 조회 실패: {e}", collection=collection_name) from e

    def ensure_collection(self, collection_name: str) -> None:
        """컬렉션 존재 확인 → 없으면 생성 / 있으면 차원 호환성 검증"""
        if not self._collection_exists(collection_name):
            vector_size = self._get_vector_size()
            try:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("collection_created", name=collection_name, vector_size=vector_size)
            except Exception as e:
                raise VectorStoreError(f"컬렉션 생성 실패: {e}", collection=collection_name) from e
        else:
            # 기존 컬렉션의 벡터 차원이 현재 모델과 일치하는지 검증
            try:
                info = self._client.get_collection(collection_name)
                existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
                current_size = self._get_vector_size()
                if existing_size != current_size:
                    logger.error(
                        "vector_dimension_mismatch",
                        collection=collection_name,
                        existing_dim=existing_size,
                        model_dim=current_size,
                        model=self.config.llm.embedding_model,
                        action="기존 컬렉션 삭제 후 재색인 필요: "
                               f"DELETE /collections/{collection_name} → 문서 재적재",
                    )
                    raise VectorStoreError(
                        f"컬렉션 '{collection_name}'의 벡터 차원({existing_size})이 "
                        f"현재 임베딩 모델({self.config.llm.embedding_model})의 차원({current_size})과 "
                        f"일치하지 않습니다. 컬렉션을 삭제하고 재색인하세요.",
                        collection=collection_name,
                    )
            except VectorStoreError:
                raise
            except Exception as e:
                logger.warning("dimension_check_skipped", collection=collection_name, error=str(e))
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

        Raises:
            VectorStoreError: 문서 저장 실패
        """
        if not documents:
            return 0

        self.ensure_collection(collection_name)

        texts = [doc["text"] for doc in documents]
        total_added = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]

            try:
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
            except Exception as e:
                logger.error(
                    "batch_upsert_failed",
                    collection=collection_name,
                    batch_start=i,
                    batch_size=len(batch_texts),
                    error=str(e),
                )
                raise VectorStoreError(
                    f"문서 저장 실패 (batch {i}~{i + len(batch_texts)}): {e}",
                    collection=collection_name,
                ) from e

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

        Raises:
            CollectionNotFoundError: 컬렉션 미존재
            VectorStoreError: 검색 실패
        """
        # 컬렉션 존재 확인
        if not self._collection_exists(collection_name):
            raise CollectionNotFoundError(collection_name)

        try:
            query_embedding = list(self._embedder.embed([query]))[0]

            search_kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "query": query_embedding.tolist(),
                "limit": top_k,
                "score_threshold": score_threshold,
            }

            if filter_conditions:
                search_kwargs["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter_conditions.items()
                    ]
                )

            results = self._client.query_points(**search_kwargs).points

        except CollectionNotFoundError:
            raise
        except UnexpectedResponse as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                raise CollectionNotFoundError(collection_name) from e
            raise VectorStoreError(f"벡터 검색 실패: {e}", collection=collection_name) from e
        except Exception as e:
            raise VectorStoreError(f"벡터 검색 실패: {e}", collection=collection_name) from e

        output = []
        for point in results:
            payload = dict(point.payload) if point.payload else {}
            text = payload.pop("text", "")
            output.append({
                "text": text,
                "score": point.score,
                "metadata": payload,
            })

        logger.info("search_completed", collection=collection_name, query=query[:50], results=len(output))
        return output

    def delete_collection(self, collection_name: str) -> None:
        """컬렉션 삭제

        Raises:
            CollectionNotFoundError: 컬렉션 미존재
        """
        if not self._collection_exists(collection_name):
            raise CollectionNotFoundError(collection_name)

        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.info("collection_deleted", name=collection_name)
        except Exception as e:
            raise VectorStoreError(f"컬렉션 삭제 실패: {e}", collection=collection_name) from e

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """컬렉션 정보 조회

        Raises:
            CollectionNotFoundError: 컬렉션 미존재
        """
        if not self._collection_exists(collection_name):
            raise CollectionNotFoundError(collection_name)

        try:
            info = self._client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            raise VectorStoreError(f"컬렉션 정보 조회 실패: {e}", collection=collection_name) from e
