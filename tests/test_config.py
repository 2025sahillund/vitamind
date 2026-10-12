from src.config import Settings


def test_settings_initialization():
    settings = Settings()
    assert settings.environment == "development"
    assert settings.embedding_dimension == 384
    assert settings.qdrant_collection_name == "vitamind_memories"


def test_secret_masking():
    settings = Settings()
    assert settings._mask_secret(None) == "not_configured"
    assert settings._mask_secret("") == "not_configured"
    assert settings._mask_secret("12345") == "***"
    assert settings._mask_secret("sk-live-secret-key-12345") == "sk-...345"


def test_integration_status_does_not_leak_secrets():
    settings = Settings(
        LYZR_API_KEY="super_secret_lyzr_key_xyz",
        QDRANT_API_KEY="super_secret_qdrant_key_abc",
        QDRANT_URL="https://cluster.qdrant.tech:6333"
    )
    status = settings.get_integration_status()

    # Verify secrets are masked
    assert "super_secret" not in str(status)
    assert status["lyzr"]["configured"] is False  # agent_id is missing
    assert status["lyzr"]["api_key_set"] is True
    assert status["qdrant"]["cloud_configured"] is True
