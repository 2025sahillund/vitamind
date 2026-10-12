from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """
    Unified memory representation stored inside Qdrant and retrieved by VitaMind.
    Enforces tenant isolation and timestamp traceability.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="default_user", description="Tenant user identifier")
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp()),
        description="Unix timestamp (seconds) for temporal sorting & filtering"
    )
    date_str: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        description="Date string (YYYY-MM-DD)"
    )
    source: str = Field(
        default="omi_webhook",
        description="'omi_webhook', 'simulated_omi', or 'manual_input'"
    )
    original_transcript: str = Field(
        description="Raw speech transcript for exact verification and citations"
    )
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized key-value health entities"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector representation of the original transcript"
    )

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert model attributes to Qdrant point payload dictionary."""
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "date_str": self.date_str,
            "source": self.source,
            "original_transcript": self.original_transcript,
            "sleep_hours": self.extracted_data.get("sleep_hours"),
            "exercise_minutes": self.extracted_data.get("exercise_minutes"),
            "exercise_completed": self.extracted_data.get("exercise_completed"),
            "energy_score": self.extracted_data.get("energy_score"),
            "stress_score": self.extracted_data.get("stress_score"),
            "symptoms": self.extracted_data.get("symptoms", []),
            "future_goals": self.extracted_data.get("future_goals", []),
            "extracted_data": self.extracted_data
        }

    @classmethod
    def from_qdrant_payload(cls, payload: Dict[str, Any], vector: Optional[List[float]] = None) -> "MemoryRecord":
        return cls(
            record_id=payload.get("record_id", str(uuid.uuid4())),
            user_id=payload.get("user_id", "default_user"),
            timestamp=payload.get("timestamp", 0),
            date_str=payload.get("date_str", ""),
            source=payload.get("source", "unknown"),
            original_transcript=payload.get("original_transcript", ""),
            extracted_data=payload.get("extracted_data", {}),
            embedding=vector
        )
