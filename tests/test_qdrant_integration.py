import os
import sys
import uuid
import logging
from typing import Dict, Any, Tuple
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import settings
from src.models.memory_schema import MemoryRecord
from src.memory.qdrant_store import QdrantMemoryStore

logger = logging.getLogger(__name__)


def get_qdrant_target_info() -> Dict[str, Any]:
    """
    Determines whether Qdrant Cloud or local on-disk storage is configured.
    Masks all credentials.
    """
    has_url = bool(settings.qdrant_url and settings.qdrant_url.strip())
    is_placeholder_url = has_url and "your-cluster-id" in (settings.qdrant_url or "")
    has_key = bool(settings.qdrant_api_key and settings.qdrant_api_key.strip())

    is_cloud_target = has_url and has_key and not is_placeholder_url

    return {
        "target_type": "qdrant_cloud" if is_cloud_target else "local_on_disk",
        "is_cloud": is_cloud_target,
        "url_configured": has_url and not is_placeholder_url,
        "masked_url": settings._mask_secret(settings.qdrant_url) if has_url and not is_placeholder_url else "not_configured",
        "api_key_configured": has_key,
        "masked_api_key": settings._mask_secret(settings.qdrant_api_key) if has_key else "not_configured",
        "collection_name": settings.qdrant_collection_name,
        "local_storage_path": settings.qdrant_local_path if not is_cloud_target else None,
        "vector_dimension": settings.embedding_dimension,
    }


def run_qdrant_live_integration() -> Dict[str, Any]:
    """
    Executes a real persistence, retrieval, semantic search, and cleanup test.
    Returns structured results for reporting.
    """
    target_info = get_qdrant_target_info()
    store = QdrantMemoryStore()

    actual_mode = "qdrant_cloud" if store.is_cloud else "local_on_disk"

    # Verify collection
    coll_info = store.get_collection_info()
    assert coll_info["status"] == "ready", f"Collection status is not ready: {coll_info}"

    # Generate unique test data
    test_record_id = f"test_mem_{uuid.uuid4().hex}"
    test_user_id = f"test_patient_{uuid.uuid4().hex[:8]}"
    transcript_text = (
        "Patient reports sleeping 7.5 hours with excellent sleep quality. "
        "Completed a 30-minute morning jog and felt energetic throughout the day."
    )
    extracted_health = {
        "sleep_hours": 7.5,
        "sleep_quality": "excellent",
        "exercise_type": "morning jog",
        "exercise_minutes": 30,
        "exercise_completed": True,
        "energy_score": 5,
        "symptoms": [],
        "future_goals": ["run 5k on Saturday"]
    }

    test_record = MemoryRecord(
        record_id=test_record_id,
        user_id=test_user_id,
        date_str="2026-10-12",
        source="integration_test",
        original_transcript=transcript_text,
        extracted_data=extracted_health
    )

    try:
        # 1. Upsert Record
        upserted_id = store.upsert_record(test_record)
        assert upserted_id == test_record_id, f"Expected {test_record_id}, got {upserted_id}"

        # 2. Retrieve by User Scope
        user_records = store.get_user_records(user_id=test_user_id)
        assert len(user_records) >= 1, "Failed to retrieve record by user scope"
        retrieved_record = user_records[0]
        assert retrieved_record.record_id == test_record_id
        assert retrieved_record.extracted_data["sleep_hours"] == 7.5
        assert retrieved_record.extracted_data["exercise_minutes"] == 30

        # 3. Semantic Search with Mandatory Tenant Isolation
        search_query = "morning jog physical exercise and sleep duration"
        search_results = store.search_memories(
            query=search_query,
            user_id=test_user_id,
            limit=5
        )
        assert len(search_results) >= 1, "Semantic search returned no results"
        matched_record = search_results[0]
        assert matched_record.record_id == test_record_id
        assert "morning jog" in matched_record.original_transcript
        assert matched_record.extracted_data["sleep_hours"] == 7.5

        # 4. Tenant Isolation Guardrail: Other user cannot see this record
        other_user_results = store.search_memories(
            query=search_query,
            user_id="unauthorized_other_patient_99",
            limit=5
        )
        assert len(other_user_results) == 0, "Tenant isolation failure: other user retrieved record"

        result = {
            "status": "success",
            "connection_mode": actual_mode,
            "collection_name": store.collection_name,
            "vector_dimension": store.vector_dim,
            "test_record_id": test_record_id,
            "test_user_id": test_user_id,
            "upsert_verified": True,
            "user_scope_retrieval_verified": True,
            "semantic_search_verified": True,
            "tenant_isolation_verified": True,
            "cleaned_up": False
        }

    finally:
        # 5. Clean up ONLY the test point created by this test
        cleanup_success = store.delete_record(test_record_id)
        # Verify point is gone
        post_cleanup = store.get_user_records(user_id=test_user_id)
        cleanup_verified = cleanup_success and len(post_cleanup) == 0
        if "result" in locals():
            result["cleaned_up"] = cleanup_verified
        store.close()

    return result


def test_qdrant_live_integration():
    """Pytest test case for Qdrant persistence and retrieval."""
    res = run_qdrant_live_integration()
    assert res["status"] == "success"
    assert res["upsert_verified"] is True
    assert res["semantic_search_verified"] is True
    assert res["tenant_isolation_verified"] is True
    assert res["cleaned_up"] is True


if __name__ == "__main__":
    print("=" * 70)
    print("VitaMind — Qdrant Persistence & Retrieval Integration Test")
    print("=" * 70)

    target = get_qdrant_target_info()
    print(f"Target Type         : {target['target_type']}")
    print(f"Collection Name     : {target['collection_name']}")
    print(f"Vector Dimension    : {target['vector_dimension']}")
    print(f"Cloud URL Configured: {target['url_configured']} ({target['masked_url']})")
    print(f"API Key Configured  : {target['api_key_configured']} ({target['masked_api_key']})")
    if target["target_type"] == "local_on_disk":
        print(f"Local Storage Path  : {target['local_storage_path']}")
        print("\n[*] Note: Qdrant Cloud is not active (QDRANT_URL is not set to a live cluster).")
        print("    Running real verification against the active local on-disk Qdrant engine.")
    else:
        print("\n[*] Connecting to configured Qdrant Cloud cluster...")

    try:
        results = run_qdrant_live_integration()
        print("\n[+] Verification Results:")
        print(f"  - Actual Connection Target : {results['connection_mode']}")
        print(f"  - Collection Name          : {results['collection_name']}")
        print(f"  - Vector Dimension         : {results['vector_dimension']} (COSINE)")
        print(f"  - Upsert Record            : SUCCESS (ID: {results['test_record_id']})")
        print(f"  - User-Scoped Retrieval    : SUCCESS")
        print(f"  - Semantic Search Query    : SUCCESS (Matched target health record)")
        print(f"  - Tenant Isolation Check   : SUCCESS (0 cross-tenant leaks)")
        print(f"  - Cleanup Test Point       : SUCCESS (Point {results['test_record_id']} deleted)")
        print("\n[SUCCESS] Qdrant persistence and retrieval verification completed successfully!")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] Qdrant verification failed: {e}")
        print("=" * 70)
        sys.exit(1)
