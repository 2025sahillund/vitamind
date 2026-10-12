import shutil
import os
import uuid
import pytest
from src.models.memory_schema import MemoryRecord
from src.memory.qdrant_store import QdrantMemoryStore


@pytest.fixture
def clean_test_store(tmp_path):
    """Creates a temporary isolated Qdrant store for testing."""
    test_collection = f"test_memories_{uuid.uuid4().hex[:8]}"
    store_dir = str(tmp_path / f"qdrant_{uuid.uuid4().hex[:8]}")
    store = QdrantMemoryStore(collection_name=test_collection, storage_path=store_dir)
    yield store
    # Cleanup collection and close client to release file locks
    try:
        store.close()
    except Exception:
        pass


def test_qdrant_collection_initialization(clean_test_store):
    info = clean_test_store.get_collection_info()
    assert info["collection_name"] == clean_test_store.collection_name
    assert info["vector_dimension"] == 384
    assert info["status"] == "ready"


def test_qdrant_upsert_and_retrieval(clean_test_store):
    record = MemoryRecord(
        user_id="patient_test_01",
        date_str="2026-10-11",
        source="simulated_omi",
        original_transcript="I slept for seven hours and went for a 20-minute jog.",
        extracted_data={
            "sleep_hours": 7.0,
            "exercise_type": "jog",
            "exercise_minutes": 20,
            "exercise_completed": True,
            "energy_score": 4
        }
    )

    # 1. Upsert Record
    rec_id = clean_test_store.upsert_record(record)
    assert rec_id == record.record_id

    # 2. Retrieve by user scope
    user_records = clean_test_store.get_user_records(user_id="patient_test_01")
    assert len(user_records) == 1
    retrieved = user_records[0]
    assert retrieved.record_id == record.record_id
    assert retrieved.extracted_data["sleep_hours"] == 7.0
    assert retrieved.extracted_data["exercise_minutes"] == 20

    # 3. Tenant Isolation verification: querying another user should return empty
    other_records = clean_test_store.get_user_records(user_id="different_patient_99")
    assert len(other_records) == 0


def test_qdrant_semantic_search(clean_test_store):
    rec1 = MemoryRecord(
        user_id="patient_test_02",
        date_str="2026-10-10",
        original_transcript="Felt extreme knee pain and stayed in bed all afternoon."
    )
    rec2 = MemoryRecord(
        user_id="patient_test_02",
        date_str="2026-10-11",
        original_transcript="Drank 2 liters of water and ate a high-protein breakfast."
    )
    clean_test_store.upsert_record(rec1)
    clean_test_store.upsert_record(rec2)

    # Semantic search for joint/pain
    results = clean_test_store.search_memories(
        query="joint discomfort and physical pain",
        user_id="patient_test_02",
        limit=1
    )
    assert len(results) >= 1
    assert "knee pain" in results[0].original_transcript


def test_qdrant_persistence_across_reconnections(tmp_path):
    """
    Verifies that records written to on-disk Qdrant persist across
    client destruction and reconnection.
    """
    custom_path = str(tmp_path / "persistent_qdrant")
    collection_name = "reconnect_test_coll"

    # Step 1: Create store and write record
    store1 = QdrantMemoryStore(collection_name=collection_name, storage_path=custom_path)
    record = MemoryRecord(
        user_id="persist_user",
        original_transcript="Persistent memory across client restart."
    )
    rec_id = store1.upsert_record(record)
    # Explicitly close to release lock file on Windows
    store1.close()

    # Step 2: Reconnect with a new client pointing to the same storage
    store2 = QdrantMemoryStore(collection_name=collection_name, storage_path=custom_path)
    try:
        reconnected_records = store2.get_user_records(user_id="persist_user")
        assert len(reconnected_records) == 1
        assert reconnected_records[0].record_id == rec_id
        assert reconnected_records[0].original_transcript == "Persistent memory across client restart."
    finally:
        store2.close()


def test_qdrant_delete_record(clean_test_store):
    record = MemoryRecord(
        user_id="delete_test_user",
        original_transcript="Temporary record for deletion test."
    )
    rec_id = clean_test_store.upsert_record(record)
    assert len(clean_test_store.get_user_records(user_id="delete_test_user")) == 1

    success = clean_test_store.delete_record(rec_id)
    assert success is True
    assert len(clean_test_store.get_user_records(user_id="delete_test_user")) == 0
