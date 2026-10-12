import pytest
from pydantic import ValidationError
from src.models.health_schema import DailyHealthEvent
from src.models.omi_schema import OmiMemoryPayload, OmiTranscriptSegment, OmiStructuredData
from src.models.memory_schema import MemoryRecord


def test_daily_health_event_valid():
    event = DailyHealthEvent(
        user_id="user_123",
        date_str="2026-10-11",
        sleep_hours=7.5,
        sleep_quality="good",
        exercise_type="walking",
        exercise_minutes=30,
        exercise_completed=True,
        energy_score=4,
        stress_score=2,
        symptoms=["mild fatigue"],
        future_goals=["walk 30 mins tomorrow"],
        raw_transcript="Slept 7.5 hours and went for a 30 minute walk."
    )
    assert event.sleep_hours == 7.5
    assert event.exercise_completed is True
    assert event.energy_score == 4


def test_daily_health_event_validation_errors():
    # Negative sleep hours should fail
    with pytest.raises(ValidationError):
        DailyHealthEvent(
            sleep_hours=-2.0,
            raw_transcript="invalid sleep"
        )

    # Invalid energy score (> 5) should fail
    with pytest.raises(ValidationError):
        DailyHealthEvent(
            energy_score=10,
            raw_transcript="super energy"
        )

    # Invalid date format should fail
    with pytest.raises(ValidationError):
        DailyHealthEvent(
            date_str="11-10-2026",
            raw_transcript="bad date format"
        )


def test_omi_payload_concatenation():
    payload = OmiMemoryPayload(
        id="omi_123",
        transcript_segments=[
            OmiTranscriptSegment(text="I slept for six hours last night.", speaker="SPEAKER_00"),
            OmiTranscriptSegment(text="Skipped my morning walk.", speaker="SPEAKER_00")
        ],
        structured=OmiStructuredData(title="Morning update", summary="Short summary")
    )
    assert payload.get_full_transcript() == "I slept for six hours last night. Skipped my morning walk."


def test_memory_record_qdrant_serialization():
    record = MemoryRecord(
        user_id="alice",
        date_str="2026-10-11",
        source="omi_webhook",
        original_transcript="Had lunch and felt tired.",
        extracted_data={"energy_score": 2, "symptoms": ["tired"]}
    )
    payload = record.to_qdrant_payload()
    assert payload["user_id"] == "alice"
    assert payload["energy_score"] == 2
    assert payload["symptoms"] == ["tired"]

    # Reconstitute from payload
    reconstructed = MemoryRecord.from_qdrant_payload(payload)
    assert reconstructed.user_id == "alice"
    assert reconstructed.extracted_data["energy_score"] == 2


def test_real_omi_payload_schema_parsing():
    raw_omi_json = {
        "id": "6708b8b9-50c1-4ef3-a9d0-141a12345678",
        "created_at": "2026-10-12T07:15:30.123456+00:00",
        "started_at": "2026-10-12T07:10:00.000000+00:00",
        "finished_at": "2026-10-12T07:15:00.000000+00:00",
        "transcript_segments": [
            {
                "text": "Slept seven hours, felt good.",
                "speaker": "SPEAKER_00",
                "speakerId": 0,
                "speaker_name": "User",
                "is_user": True,
                "start": 0.0,
                "end": 3.5
            }
        ],
        "structured": {
            "title": "Morning Check-In",
            "overview": "User reports good sleep duration.",
            "emoji": "☀️",
            "category": "health",
            "action_items": [
                {"description": "Drink 2L water", "completed": False}
            ]
        },
        "discarded": False,
        "folder_id": "none",
        "apps_response": []
    }

    parsed = OmiMemoryPayload(**raw_omi_json)
    assert parsed.id == "6708b8b9-50c1-4ef3-a9d0-141a12345678"
    assert parsed.discarded is False
    assert len(parsed.transcript_segments) == 1
    assert parsed.transcript_segments[0].speaker_id == 0
    assert parsed.structured.overview == "User reports good sleep duration."
    assert parsed.get_full_transcript() == "Slept seven hours, felt good."
