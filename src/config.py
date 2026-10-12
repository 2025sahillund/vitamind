import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is discovered across different execution contexts
load_dotenv(find_dotenv(), override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **values):
        if values:
            if "lyzr_agent_id" not in values and "LYZR_AGENT_ID" not in values:
                values.setdefault("lyzr_agent_id", None)
        super().__init__(**values)

    # Server Settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Qdrant Vector Database
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="vitamind_memories", alias="QDRANT_COLLECTION_NAME")
    qdrant_local_path: str = Field(default="./data/qdrant_local", alias="QDRANT_LOCAL_PATH")

    # Embedding Model Configuration
    embedding_provider: str = Field(default="fastembed", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL_NAME")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Lyzr Agent Studio
    lyzr_api_key: Optional[str] = Field(default=None, alias="LYZR_API_KEY")
    lyzr_agent_id: Optional[str] = Field(default=None, alias="LYZR_AGENT_ID")
    lyzr_studio_url: str = Field(default="https://agent-prod.studio.lyzr.ai/v3", alias="LYZR_STUDIO_URL")

    # Omi Webhook Integration
    omi_webhook_secret: Optional[str] = Field(default=None, alias="OMI_WEBHOOK_SECRET")

    # Optional direct LLM fallback
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    def _mask_secret(self, secret: Optional[str]) -> str:
        if not secret:
            return "not_configured"
        if len(secret) <= 6:
            return "***"
        return f"{secret[:3]}...{secret[-3:]}"

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Reports which credentials and services are configured without
        exposing secret values.
        """
        is_qdrant_cloud = bool(
            self.qdrant_url and self.qdrant_api_key and "your-cluster-id" not in (self.qdrant_url or "")
        )
        is_lyzr_ready = bool(self.lyzr_api_key and self.lyzr_agent_id)
        is_openai_ready = bool(self.openai_api_key)

        return {
            "qdrant": {
                "mode": "cloud" if is_qdrant_cloud else "local_disk",
                "cloud_configured": is_qdrant_cloud,
                "url": self._mask_secret(self.qdrant_url) if is_qdrant_cloud else None,
                "local_storage_path": self.qdrant_local_path if not is_qdrant_cloud else None,
                "collection_name": self.qdrant_collection_name,
            },
            "lyzr": {
                "configured": is_lyzr_ready,
                "api_key_set": bool(self.lyzr_api_key),
                "agent_id_set": bool(self.lyzr_agent_id),
                "agent_id": self._mask_secret(self.lyzr_agent_id) if self.lyzr_agent_id else "not_configured",
                "studio_url": self.lyzr_studio_url,
            },
            "embedding": {
                "provider": self.embedding_provider,
                "model_name": self.embedding_model_name,
                "dimension": self.embedding_dimension,
                "zero_cost_local": self.embedding_provider == "fastembed",
            },
            "omi": {
                "webhook_ready": True,
                "secret_configured": bool(self.omi_webhook_secret),
                "endpoint_path": "/api/v1/omi/webhook",
            },
            "external_llm": {
                "openai_configured": is_openai_ready,
            }
        }


settings = Settings()
