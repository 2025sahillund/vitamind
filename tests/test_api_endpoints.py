import uuid
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "vitamind"
    assert data["version"] == "0.1.0"


def test_config_status_endpoint():
    response = client.get("/api/v1/config/status")
    assert response.status_code == 200
    data = response.json()
    assert "qdrant" in data
    assert "lyzr" in data
    assert "omi" in data
    # Ensure no secrets leak
    assert data["qdrant"]["cloud_configured"] is False or "***" in str(data)


def test_simulate_voice_log_and_trends():
    # 1. Post a simulated voice log
    resp = client.post(
        "/api/v1/simulate/log",
        params={
            "transcript": "I slept for 7.5 hours last night and walked for 30 minutes. Felt great!",
            "user_id": "test_api_user",
            "date_str": "2026-10-11"
        }
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "success"
    assert result["source"] == "simulated_omi"
    assert result["extracted_summary"]["sleep_hours"] == 7.5
    assert result["extracted_summary"]["exercise_minutes"] == 30

    # 2. Query memories endpoint
    mem_resp = client.get("/api/v1/memories", params={"user_id": "test_api_user"})
    assert mem_resp.status_code == 200
    mem_data = mem_resp.json()
    assert mem_data["count"] >= 1

    # 3. Query trends endpoint
    trend_resp = client.get("/api/v1/analytics/trends", params={"user_id": "test_api_user", "days": 7})
    assert trend_resp.status_code == 200
    trend_data = trend_resp.json()
    assert trend_data["sleep"]["recorded_days"] >= 1
    assert trend_data["exercise"]["total_minutes"] >= 30


def test_omi_webhook_real_payload_format():
    unique_mem_id = f"real_{uuid.uuid4().hex[:8]}"
    real_payload = {
        "id": unique_mem_id,
        "created_at": "2026-10-12T07:30:00.000000+00:00",
        "started_at": "2026-10-12T07:00:00.000000+00:00",
        "finished_at": "2026-10-12T07:29:00.000000+00:00",
        "transcript_segments": [
            {
                "text": "I slept for six hours last night and skipped my morning walk.",
                "speaker": "SPEAKER_00",
                "speakerId": 0,
                "speaker_name": "User",
                "is_user": True,
                "start": 0.0,
                "end": 5.2
            },
            {
                "text": "Tomorrow I plan to walk for 20 minutes.",
                "speaker": "SPEAKER_00",
                "speakerId": 0,
                "speaker_name": "User",
                "is_user": True,
                "start": 5.5,
                "end": 9.1
            }
        ],
        "structured": {
            "title": "Morning Routine Log",
            "overview": "User slept 6 hours, skipped walk, plans 20 min walk tomorrow.",
            "category": "health",
            "emoji": "🏃",
            "action_items": [
                {"description": "Walk 20 minutes tomorrow", "completed": False}
            ]
        },
        "discarded": False
    }

    try:
        resp = client.post("/api/v1/omi/webhook?uid=patient_omi_101", json=real_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["is_duplicate"] is False
        assert data["source"] == "omi_webhook"
        assert data["record_id"] == f"omi_{unique_mem_id}"

        # Query memories to confirm Qdrant persistence under user_id
        mem_resp = client.get("/api/v1/memories?user_id=patient_omi_101")
        assert mem_resp.status_code == 200
        mem_data = mem_resp.json()
        assert mem_data["count"] >= 1
        record_ids = [r["record_id"] for r in mem_data["records"]]
        assert f"omi_{unique_mem_id}" in record_ids
    finally:
        from src.api.routes import memory_store
        memory_store.delete_record(f"omi_{unique_mem_id}")


def test_omi_webhook_duplicate_handling():
    unique_dup_id = f"dup_{uuid.uuid4().hex[:8]}"
    payload = {
        "id": unique_dup_id,
        "created_at": "2026-10-12T08:00:00.000Z",
        "transcript_segments": [
            {"text": "Slept 8 hours and walked 40 minutes.", "speaker": "SPEAKER_00"}
        ],
        "discarded": False
    }

    try:
        # First delivery
        resp1 = client.post("/api/v1/omi/webhook?uid=dup_user", json=payload)
        assert resp1.status_code == 200
        assert resp1.json()["is_duplicate"] is False

        # Second delivery with exact same Omi ID
        resp2 = client.post("/api/v1/omi/webhook?uid=dup_user", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["is_duplicate"] is True
        assert data2["record_id"] == resp1.json()["record_id"]
    finally:
        from src.api.routes import memory_store
        memory_store.delete_record(f"omi_{unique_dup_id}")


def test_omi_webhook_discarded_memory():
    discarded_payload = {
        "id": "omi_mem_discarded_01",
        "created_at": "2026-10-12T08:00:00.000Z",
        "transcript_segments": [],
        "discarded": True
    }
    resp = client.post("/api/v1/omi/webhook", json=discarded_payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert resp.json()["reason"] == "discarded_memory"


def test_omi_webhook_malformed_and_empty_payloads():
    # 1. Missing required 'id' field -> 422
    resp_missing_id = client.post("/api/v1/omi/webhook", json={"transcript_segments": []})
    assert resp_missing_id.status_code == 422

    # 2. Empty transcript (no segments, no overview) -> 400
    resp_empty = client.post("/api/v1/omi/webhook", json={"id": "empty_mem", "transcript_segments": []})
    assert resp_empty.status_code == 400
    assert "contained no transcript text" in resp_empty.json()["detail"]
