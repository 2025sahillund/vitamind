from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, Header

from src.config import settings
from src.models.omi_schema import OmiMemoryPayload
from src.models.health_schema import DailyHealthEvent
from src.models.memory_schema import MemoryRecord
from src.memory.qdrant_store import QdrantMemoryStore
from src.analytics.temporal_engine import TemporalTrendEngine
from src.agents.lyzr_client import LyzrAgentClient

router = APIRouter()
memory_store = QdrantMemoryStore()
lyzr_client = LyzrAgentClient()


@router.get("/health")
def health_check():
    """FastAPI service health check."""
    return {
        "status": "healthy",
        "service": "vitamind",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/api/v1/config/status")
def config_status():
    """
    Validates configuration and reports integration readiness
    without exposing confidential API keys.
    """
    status = settings.get_integration_status()
    # Add live Qdrant collection status
    status["qdrant_live"] = memory_store.get_collection_info()
    return status


@router.post("/api/v1/omi/webhook")
def omi_webhook(
    payload: OmiMemoryPayload,
    x_omi_secret: Optional[str] = Header(default=None),
    uid: Optional[str] = Query(default=None, description="Omi user ID passed via query param")
):
    """
    Live Omi Memory Trigger Webhook receiver.
    Documented at https://docs.omi.me/
    """
    # Validate secret header if configured with an actual secret
    active_secret = settings.omi_webhook_secret
    if active_secret and active_secret not in ("", "your_optional_webhook_secret_here"):
        if x_omi_secret != active_secret:
            raise HTTPException(status_code=401, detail="Invalid Omi webhook secret header")

    # Handle discarded memories (e.g. background noise or discarded sessions)
    if payload.discarded:
        return {
            "status": "ignored",
            "reason": "discarded_memory",
            "memory_id": payload.id
        }

    transcript = payload.get_full_transcript()
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Omi payload contained no transcript text")

    # Resolve user ID: from query parameter ?uid=..., or payload metadata, or default
    user_id = uid or (payload.metadata.get("user_id") if payload.metadata else None) or "default_user"

    # Deterministic record ID based on Omi memory ID
    record_id = f"omi_{payload.id}"

    # Duplicate-event handling: return existing record if already processed
    existing_record = memory_store.get_record(record_id)
    if existing_record:
        return {
            "status": "success",
            "is_duplicate": True,
            "record_id": existing_record.record_id,
            "source": "omi_webhook",
            "extracted_summary": existing_record.extracted_data
        }

    # Extract date from Omi created_at if available
    date_str = None
    if payload.created_at:
        try:
            date_candidate = payload.created_at[:10]
            datetime.strptime(date_candidate, "%Y-%m-%d")
            date_str = date_candidate
        except Exception:
            date_str = None

    # Structured extraction via Lyzr (or deterministic fallback if unconfigured)
    extracted_data = _extract_or_fallback(transcript, user_id=user_id)
    if date_str:
        extracted_data.date_str = date_str

    # Persist to Qdrant memory
    record = MemoryRecord(
        record_id=record_id,
        user_id=user_id,
        date_str=extracted_data.date_str,
        source="omi_webhook",
        original_transcript=transcript,
        extracted_data=extracted_data.model_dump()
    )
    upserted_id = memory_store.upsert_record(record)

    return {
        "status": "success",
        "is_duplicate": False,
        "record_id": upserted_id,
        "source": "omi_webhook",
        "extracted_summary": extracted_data.model_dump()
    }


@router.post("/api/v1/simulate/log")
def simulate_voice_log(
    transcript: str = Query(..., description="Simulated raw voice transcript"),
    user_id: str = Query(default="default_user", description="Scoped user ID"),
    date_str: Optional[str] = Query(default=None, description="Optional override date (YYYY-MM-DD)")
):
    """
    Test harness for simulating Omi voice transcripts.
    Explicitly labelled with source='simulated_omi'.
    """
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    extracted_data = _extract_or_fallback(transcript, user_id=user_id)
    if date_str:
        extracted_data.date_str = date_str

    record = MemoryRecord(
        user_id=user_id,
        source="simulated_omi",
        date_str=extracted_data.date_str,
        original_transcript=transcript,
        extracted_data=extracted_data.model_dump()
    )
    record_id = memory_store.upsert_record(record)

    return {
        "status": "success",
        "record_id": record_id,
        "source": "simulated_omi",
        "extracted_summary": extracted_data.model_dump()
    }


@router.get("/api/v1/memories")
def get_memories(
    user_id: str = Query(default="default_user"),
    query: Optional[str] = Query(default=None, description="Semantic search query"),
    limit: int = Query(default=10, ge=1, le=100)
):
    """Search or retrieve memory records for a user."""
    if query:
        records = memory_store.search_memories(query=query, user_id=user_id, limit=limit)
    else:
        records = memory_store.get_user_records(user_id=user_id, limit=limit)

    return {
        "user_id": user_id,
        "count": len(records),
        "records": [r.model_dump(exclude={"embedding"}) for r in records]
    }


@router.get("/api/v1/analytics/trends")
def get_temporal_trends(
    user_id: str = Query(default="default_user"),
    days: int = Query(default=7, ge=1, le=90)
):
    """
    Runs deterministic analytics over stored Qdrant records
    and provides verified source evidence quotes.
    """
    records = memory_store.get_user_records(user_id=user_id, limit=days * 5)
    
    sleep_analytics = TemporalTrendEngine.analyze_sleep_trends(records, target_days=days)
    exercise_analytics = TemporalTrendEngine.analyze_exercise_frequency(records)
    energy_analytics = TemporalTrendEngine.analyze_energy_and_stress(records)

    return {
        "user_id": user_id,
        "target_days": days,
        "sleep": sleep_analytics,
        "exercise": exercise_analytics,
        "energy_and_stress": energy_analytics,
    }


def _extract_or_fallback(transcript: str, user_id: str) -> DailyHealthEvent:
    """
    Uses Lyzr if credentials exist; otherwise falls back to a deterministic regex parser
    so the pipeline never crashes during offline development.
    """
    if lyzr_client.is_configured():
        try:
            return lyzr_client.extract_health_event(transcript, user_id=user_id)
        except Exception as e:
            # Fallback if Lyzr call errors
            pass

    # Deterministic parser fallback for local testing
    return _deterministic_rule_based_parser(transcript, user_id=user_id)


def _deterministic_rule_based_parser(transcript: str, user_id: str) -> DailyHealthEvent:
    """
    Rule-based extractor for reliable local testing without an active LLM API key.
    """
    import re
    lower = transcript.lower()

    # Sleep hours
    sleep_hours = None
    sleep_match = re.search(r"(\d+(\.\d+)?)\s*(hours?|hrs?)\s*(of\s*sleep|sleep)", lower)
    if not sleep_match:
        sleep_match = re.search(r"slept\s*(for\s*)?(\d+(\.\d+)?)", lower)
        if sleep_match:
            sleep_hours = float(sleep_match.group(2))
    else:
        sleep_hours = float(sleep_match.group(1))

    # Exercise
    exercise_completed = None
    exercise_minutes = None
    exercise_type = None

    if "walk" in lower or "walking" in lower:
        exercise_type = "walking"
    elif "run" in lower or "jog" in lower:
        exercise_type = "running"
    elif "gym" in lower or "workout" in lower:
        exercise_type = "workout"

    if "skipped" in lower or "didn't" in lower or "missed" in lower:
        if exercise_type:
            exercise_completed = False
            exercise_minutes = 0
    elif exercise_type:
        mins_match = re.search(r"(\d+)\s*(mins?|minutes?)", lower)
        if mins_match:
            exercise_minutes = int(mins_match.group(1))
            exercise_completed = True
        else:
            exercise_completed = True

    # Energy score (1-5)
    energy_score = None
    if "exhausted" in lower or "very tired" in lower:
        energy_score = 1
    elif "tired" in lower or "sluggish" in lower or "fatigued" in lower:
        energy_score = 2
    elif "okay" in lower or "fine" in lower:
        energy_score = 3
    elif "good energy" in lower or "energetic" in lower:
        energy_score = 4
    elif "great" in lower or "full of energy" in lower:
        energy_score = 5

    # Symptoms
    symptoms = []
    for s in ["headache", "tired", "sluggish", "fatigue", "nausea", "dizziness", "soreness"]:
        if s in lower:
            symptoms.append(s)

    # Future goals
    future_goals = []
    goal_match = re.search(r"(want to|plan to|goal is to)\s*([^.]+)", lower)
    if goal_match:
        future_goals.append(goal_match.group(0).strip())

    return DailyHealthEvent(
        user_id=user_id,
        sleep_hours=sleep_hours,
        exercise_type=exercise_type,
        exercise_minutes=exercise_minutes,
        exercise_completed=exercise_completed,
        energy_score=energy_score,
        symptoms=symptoms,
        future_goals=future_goals,
        raw_transcript=transcript
    )
