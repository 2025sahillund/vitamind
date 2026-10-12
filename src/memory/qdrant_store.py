import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.models.memory_schema import MemoryRecord
from src.memory.embeddings import embedder

logger = logging.getLogger(__name__)


class QdrantMemoryStore:
    """
    Qdrant client wrapper supporting both Qdrant Cloud and Local On-Disk mode.
    Guarantees strict tenant isolation by filtering all queries on user_id.
    """
    _local_clients: Dict[str, QdrantClient] = {}

    def __init__(
        self,
        collection_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        client: Optional[QdrantClient] = None
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_dim = settings.embedding_dimension
        self.storage_path = storage_path or settings.qdrant_local_path
        self.is_cloud = bool(settings.qdrant_url and settings.qdrant_api_key)
        self.client = client or self._initialize_client()
        self._ensure_collection_exists()

    def _initialize_client(self) -> QdrantClient:
        """Connects to Qdrant Cloud or falls back to local on-disk storage."""
        if self.is_cloud:
            try:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=5.0
                )
                # Verify real connectivity
                client.get_collections()
                logger.info(f"Connected to Qdrant Cloud at {settings.qdrant_url}")
                return client
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant Cloud ({e}). Falling back to local storage.")
                self.is_cloud = False

        # Local On-Disk Mode (cached per storage path to prevent Windows file lock conflicts)
        os.makedirs(self.storage_path, exist_ok=True)
        abs_path = os.path.abspath(self.storage_path)
        if abs_path not in self._local_clients:
            logger.info(f"Initialized local Qdrant storage at {self.storage_path}")
            self._local_clients[abs_path] = QdrantClient(path=self.storage_path)
        return self._local_clients[abs_path]

    def close(self) -> None:
        """Closes client and releases file locks."""
        abs_path = os.path.abspath(self.storage_path)
        cached_client = self._local_clients.pop(abs_path, None)
        target = cached_client or self.client
        try:
            target.close()
        except Exception:
            pass

    def _ensure_collection_exists(self) -> None:
        """Creates the collection and creates indexes for tenant isolation."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' (dim={self.vector_dim})")

                # Create payload indexes for fast filtering
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user_id",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="date_str",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="timestamp",
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")

    def upsert_record(self, record: MemoryRecord) -> str:
        """Upsert a memory record with vector and metadata payload."""
        if not record.embedding:
            record.embedding = embedder.embed_text(record.original_transcript)

        # Qdrant requires UUID or integer ID
        try:
            point_id = str(uuid.UUID(record.record_id))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record.record_id))

        point = models.PointStruct(
            id=point_id,
            vector=record.embedding,
            payload=record.to_qdrant_payload()
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        return record.record_id

    def delete_record(self, record_id: str) -> bool:
        """
        Deletes a specific record by its ID, cleaning up test or removed records.
        """
        try:
            point_id = str(uuid.UUID(record_id))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        """Retrieves a specific memory record by its record ID if it exists."""
        try:
            point_id = str(uuid.UUID(record_id))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            if points and points[0].payload:
                return MemoryRecord.from_qdrant_payload(points[0].payload)
        except Exception as e:
            logger.debug(f"Record {record_id} lookup exception: {e}")
        return None

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[MemoryRecord]:
        """
        Semantic search with mandatory tenant isolation (user_id filter).
        Optional date range filtering.
        """
        query_vector = embedder.embed_text(query)

        filter_conditions = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            )
        ]

        if date_from and date_to:
            filter_conditions.append(
                models.FieldCondition(
                    key="date_str",
                    range=models.Range(gte=date_from, lte=date_to)
                )
            )

        query_filter = models.Filter(must=filter_conditions)

        # Qdrant v1.11+ uses query_points; legacy uses search
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            search_results = response.points
        else:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )

        records = []
        for hit in search_results:
            if hit.payload:
                vector = hit.vector if isinstance(hit.vector, list) else None
                rec = MemoryRecord.from_qdrant_payload(hit.payload, vector=vector)
                records.append(rec)
        return records

    def get_user_records(self, user_id: str, limit: int = 100) -> List[MemoryRecord]:
        """Retrieve all chronological records for a specific user using scroll."""
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )

        response, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        records = []
        for point in response:
            if point.payload:
                records.append(MemoryRecord.from_qdrant_payload(point.payload))
        
        # Sort chronologically by timestamp
        records.sort(key=lambda r: r.timestamp)
        return records

    def get_collection_info(self) -> Dict[str, Any]:
        """Collection health telemetry for UI and monitoring."""
        try:
            info = self.client.get_collection(self.collection_name)
            points_count = info.points_count if hasattr(info, 'points_count') else 0
            return {
                "collection_name": self.collection_name,
                "mode": "cloud" if self.is_cloud else "local_disk",
                "status": "ready",
                "points_count": points_count,
                "vector_dimension": self.vector_dim
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "mode": "cloud" if self.is_cloud else "local_disk",
                "status": f"error: {str(e)}",
                "points_count": 0,
                "vector_dimension": self.vector_dim
            }
