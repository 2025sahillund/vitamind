import json
import logging
import re
from typing import Dict, Any, Optional
import httpx

from src.config import settings
from src.models.health_schema import DailyHealthEvent

logger = logging.getLogger(__name__)


_DEFAULT = object()


class LyzrAgentClient:
    """
    Client adapter for Lyzr Agent Studio REST API.
    Documented at https://docs.lyzr.ai/
    Endpoint: POST https://agent-prod.studio.lyzr.ai/v3/inference/chat/
    Required headers: x-api-key, Content-Type: application/json
    Required payload: user_id, agent_id, session_id, message
    """

    def __init__(self, api_key: Any = _DEFAULT, agent_id: Any = _DEFAULT):
        self._api_key = api_key
        self._agent_id = agent_id
        self.base_url = settings.lyzr_studio_url.rstrip("/")
        self.endpoint = f"{self.base_url}/inference/chat/"

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key is _DEFAULT:
            return settings.lyzr_api_key
        return self._api_key

    @property
    def agent_id(self) -> Optional[str]:
        if self._agent_id is _DEFAULT:
            return settings.lyzr_agent_id
        return self._agent_id

    def is_configured(self) -> bool:
        return bool(self.api_key and self.agent_id)

    def call_inference_raw(
        self,
        message: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        timeout: float = 30.0
    ) -> httpx.Response:
        """
        Executes raw HTTP POST to Lyzr inference endpoint without schema parsing.
        Useful for live diagnostics, health checks, and HTTP status verification.
        """
        if not self.is_configured():
            raise ValueError(
                "Lyzr is not configured. Set LYZR_API_KEY and LYZR_AGENT_ID in your .env file."
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "user_id": user_id,
            "agent_id": self.agent_id,
            "session_id": session_id or f"{user_id}_session",
            "message": message
        }

        with httpx.Client(timeout=timeout) as client:
            return client.post(self.endpoint, headers=headers, json=payload)

    def extract_health_event(
        self,
        transcript: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None
    ) -> DailyHealthEvent:
        """
        Sends voice transcript to Lyzr Agent and parses structured DailyHealthEvent.
        """
        if not self.is_configured():
            raise ValueError(
                "Lyzr is not configured. Set LYZR_API_KEY and LYZR_AGENT_ID in your .env file."
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        extraction_prompt = f"""You are VitaMind Health Log Extractor, an AI agent that converts user-reported voice transcripts into structured health and habit observations.

Extract only information explicitly stated in the transcript, including:
- Sleep duration and sleep quality
- Physical activity and exercise duration
- Self-reported energy and mood
- Habits completed or skipped
- Future goals and intentions
- Relevant event dates and time references

Return valid JSON only, following the exact schema requested below. Use null for unknown values. Never interpret an unmentioned habit as skipped or incomplete. Separate completed actions from future intentions. Preserve the original meaning and do not invent dates, quantities, or medical facts.

Do not diagnose illnesses, recommend treatments, or present correlations as medical conclusions. If information is ambiguous, preserve that uncertainty.

JSON Schema:
{{
  "sleep_hours": <float or null>,
  "sleep_quality": <"poor" | "fair" | "good" | "excellent" | null>,
  "exercise_type": <string or null>,
  "exercise_minutes": <int or null>,
  "exercise_completed": <boolean or null, true if completed, false if skipped/missed>,
  "energy_score": <int 1-5 or null>,
  "stress_score": <int 1-5 or null>,
  "meals_logged": [<list of strings>],
  "medications_logged": [<list of strings>],
  "symptoms": [<list of strings>],
  "future_goals": [<list of strings>]
}}

Voice Transcript:
"{transcript}"
"""

        payload = {
            "user_id": user_id,
            "agent_id": self.agent_id,
            "session_id": session_id or f"{user_id}_session",
            "message": extraction_prompt
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            logger.error("Lyzr API request timed out after 30s")
            raise TimeoutError("Lyzr Agent inference timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"Lyzr API HTTP Error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Lyzr API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Lyzr API Connection error: {str(e)}")
            raise ConnectionError(f"Failed to communicate with Lyzr Studio: {str(e)}")

        # Parse response content
        raw_output = data.get("response", "")
        parsed_dict: Dict[str, Any] = {}

        if isinstance(raw_output, dict):
            parsed_dict = raw_output
        elif isinstance(raw_output, str):
            clean_json = raw_output.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            try:
                parsed_dict = json.loads(clean_json)
            except json.JSONDecodeError:
                json_match = re.search(r"(\{.*\})", clean_json, re.DOTALL)
                if json_match:
                    try:
                        parsed_dict = json.loads(json_match.group(1))
                    except json.JSONDecodeError as err:
                        logger.error(f"Failed to parse Lyzr response as JSON: {raw_output}")
                        raise ValueError(f"Lyzr returned invalid JSON: {str(err)}")
                else:
                    logger.error(f"Failed to parse Lyzr response as JSON: {raw_output}")
                    raise ValueError("Lyzr returned invalid JSON: no JSON object found")

        if not isinstance(parsed_dict, dict):
            raise ValueError(f"Lyzr response is not a valid JSON object: {type(parsed_dict)}")

        # Sanitize / normalize field types for robust DailyHealthEvent validation
        if parsed_dict.get("sleep_hours") is not None:
            try:
                parsed_dict["sleep_hours"] = float(parsed_dict["sleep_hours"])
            except (ValueError, TypeError):
                parsed_dict["sleep_hours"] = None

        if parsed_dict.get("sleep_quality") not in ("poor", "fair", "good", "excellent", None):
            parsed_dict["sleep_quality"] = None

        if parsed_dict.get("exercise_minutes") is not None:
            try:
                parsed_dict["exercise_minutes"] = int(parsed_dict["exercise_minutes"])
            except (ValueError, TypeError):
                parsed_dict["exercise_minutes"] = None

        for score_key in ("energy_score", "stress_score"):
            score_val = parsed_dict.get(score_key)
            if score_val is not None:
                try:
                    score_int = int(score_val)
                    parsed_dict[score_key] = score_int if 1 <= score_int <= 5 else None
                except (ValueError, TypeError):
                    parsed_dict[score_key] = None

        for list_key in ("meals_logged", "medications_logged", "symptoms", "future_goals"):
            val = parsed_dict.get(list_key)
            if val is None:
                parsed_dict[list_key] = []
            elif isinstance(val, str):
                parsed_dict[list_key] = [val]
            elif not isinstance(val, list):
                parsed_dict[list_key] = list(val)

        # Attach grounding raw transcript and user_id
        parsed_dict["raw_transcript"] = transcript
        parsed_dict["user_id"] = user_id

        # Validate with Pydantic
        return DailyHealthEvent(**parsed_dict)
