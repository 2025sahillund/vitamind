from datetime import datetime, timezone
from typing import List, Optional, Literal
import uuid
from pydantic import BaseModel, Field, field_validator


class DailyHealthEvent(BaseModel):
    """
    Strongly-typed clinical habit schema extracted from free-form speech.
    Supports deterministic calculations without LLM hallucinations.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="default_user", description="Scoped user identifier")
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    date_str: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        description="ISO date string (YYYY-MM-DD) for time-series grouping"
    )

    # Sleep Metrics
    sleep_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=24.0,
        description="Reported sleep duration in hours"
    )
    sleep_quality: Optional[Literal["poor", "fair", "good", "excellent"]] = Field(
        default=None,
        description="Self-reported subjective sleep quality"
    )

    # Physical Activity
    exercise_type: Optional[str] = Field(
        default=None,
        description="e.g., walking, jogging, cycling, yoga"
    )
    exercise_minutes: Optional[int] = Field(
        default=None,
        ge=0,
        le=1440,
        description="Exercise duration in minutes"
    )
    exercise_completed: Optional[bool] = Field(
        default=None,
        description="True if planned exercise was completed, False if explicitly skipped"
    )

    # Subjective State (1 to 5 Likert Scale)
    energy_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Self-reported energy level: 1 (exhausted) to 5 (full energy)"
    )
    stress_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Self-reported stress level: 1 (very calm) to 5 (extreme stress)"
    )

    # Habits & Symptoms
    meals_logged: List[str] = Field(
        default_factory=list,
        description="Items or meals mentioned in voice log"
    )
    medications_logged: List[str] = Field(
        default_factory=list,
        description="Medications or supplements mentioned"
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Reported physical symptoms (e.g., headache, fatigue, nausea)"
    )
    future_goals: List[str] = Field(
        default_factory=list,
        description="Commitments or goals stated for tomorrow or future"
    )

    # Grounding & Traceability
    raw_transcript: str = Field(
        description="Exact raw speech transcript segment grounding this extraction"
    )

    @field_validator("date_str")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date_str must follow YYYY-MM-DD format")
