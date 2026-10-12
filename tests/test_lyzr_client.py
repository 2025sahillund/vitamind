import json
import pytest
from unittest.mock import patch, MagicMock
import httpx
from src.agents.lyzr_client import LyzrAgentClient
from src.models.health_schema import DailyHealthEvent


def test_lyzr_unconfigured_raises_value_error():
    client = LyzrAgentClient(api_key=None, agent_id=None)
    assert client.is_configured() is False
    with pytest.raises(ValueError, match="Lyzr is not configured"):
        client.extract_health_event("I slept for 7 hours.")


def test_lyzr_successful_structured_extraction():
    client = LyzrAgentClient(api_key="mock_key", agent_id="mock_agent")
    assert client.is_configured() is True

    mock_llm_json = json.dumps({
        "sleep_hours": 6.5,
        "sleep_quality": "fair",
        "exercise_type": "walking",
        "exercise_minutes": 25,
        "exercise_completed": True,
        "energy_score": 3,
        "stress_score": 2,
        "meals_logged": ["oatmeal"],
        "medications_logged": ["vitamin D"],
        "symptoms": ["mild fatigue"],
        "future_goals": ["walk 30 minutes tomorrow"]
    })

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": f"```json\n{mock_llm_json}\n```"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        event = client.extract_health_event("Slept 6.5 hours, had oatmeal, took vitamin D, walked 25 min.")

        assert isinstance(event, DailyHealthEvent)
        assert event.sleep_hours == 6.5
        assert event.exercise_minutes == 25
        assert event.energy_score == 3
        assert "vitamin D" in event.medications_logged


def test_lyzr_invalid_json_handling():
    client = LyzrAgentClient(api_key="mock_key", agent_id="mock_agent")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Sorry, I am just a language model and didn't output JSON"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(ValueError, match="Lyzr returned invalid JSON"):
            client.extract_health_event("Test transcript")


def test_lyzr_timeout_handling():
    client = LyzrAgentClient(api_key="mock_key", agent_id="mock_agent")

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(TimeoutError, match="Lyzr Agent inference timed out"):
            client.extract_health_event("Test transcript")


def test_lyzr_http_error_handling():
    client = LyzrAgentClient(api_key="mock_key", agent_id="mock_agent")

    mock_req = httpx.Request("POST", "https://agent-prod.studio.lyzr.ai/v3/inference/chat/")
    mock_resp = httpx.Response(401, request=mock_req, text="Unauthorized API Key")

    with patch("httpx.Client.post", side_effect=httpx.HTTPStatusError("401", request=mock_req, response=mock_resp)):
        with pytest.raises(RuntimeError, match="Lyzr API error: 401"):
            client.extract_health_event("Test transcript")


def test_lyzr_endpoint_payload_and_headers():
    client = LyzrAgentClient(api_key="lyzr_test_secret_123", agent_id="agent_abc_789")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": json.dumps({
            "sleep_hours": 6.0,
            "sleep_quality": "fair",
            "exercise_type": "walking",
            "exercise_minutes": 20,
            "exercise_completed": True,
            "energy_score": 3,
            "stress_score": 2,
            "meals_logged": [],
            "medications_logged": [],
            "symptoms": [],
            "future_goals": ["walk tomorrow"]
        })
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        event = client.extract_health_event(
            transcript="I slept 6 hours.",
            user_id="user_custom",
            session_id="session_custom_1"
        )

        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        assert call_kwargs["headers"]["x-api-key"] == "lyzr_test_secret_123"
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

        payload = call_kwargs["json"]
        assert payload["user_id"] == "user_custom"
        assert payload["agent_id"] == "agent_abc_789"
        assert payload["session_id"] == "session_custom_1"
        assert "message" in payload
        assert "I slept 6 hours." in payload["message"]
        assert isinstance(event, DailyHealthEvent)
        assert event.sleep_hours == 6.0


def test_lyzr_call_inference_raw():
    client = LyzrAgentClient(api_key="test_key", agent_id="test_agent")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        resp = client.call_inference_raw(
            message="Ping",
            user_id="u1",
            session_id="s1"
        )
        assert resp.status_code == 200
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        assert call_kwargs["headers"]["x-api-key"] == "test_key"
        assert call_kwargs["json"]["user_id"] == "u1"
        assert call_kwargs["json"]["agent_id"] == "test_agent"
        assert call_kwargs["json"]["session_id"] == "s1"
        assert call_kwargs["json"]["message"] == "Ping"
