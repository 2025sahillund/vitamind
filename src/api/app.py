from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="VitaMind Cognitive Agent Service",
    description="Voice-first continuous health and habit tracking API for Lyzr × Qdrant × Omi Hackathon",
    version="0.1.0"
)

# Enable CORS for local dashboards and webhooks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
