"""Local dashboard API and static file hosting."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .storage import JSONLStore


def create_app(checkpoint_dir: str | Path = "checkpoints") -> FastAPI:
    app = FastAPI(title="Conversation Pattern Studio")
    store = JSONLStore(checkpoint_dir)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/analysis")
    def analysis() -> dict:
        return {stage: store.read(stage) or [] for stage in (
            "conversations", "summaries", "clusters", "meta_clusters", "dimensionality"
        )}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="dashboard")
    return app
