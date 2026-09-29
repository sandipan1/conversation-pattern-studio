"""Public data records and input adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: datetime | None = None


class Conversation(BaseModel):
    chat_id: str
    messages: list[Message]
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> list[Conversation]:
        """Read a JSON array or JSONL file of conversation objects."""
        path = Path(path)
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        records = json.loads(raw) if raw.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
        return [cls.model_validate(record) for record in records]

    @classmethod
    def from_claude_export(cls, path: str | Path) -> list[Conversation]:
        """Convert the conversations.json export from Claude to this schema."""
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        conversations: list[Conversation] = []
        for record in records:
            messages = []
            for item in sorted(record.get("chat_messages", []), key=lambda m: m.get("created_at", "")):
                content = "\n".join(part.get("text", "") for part in item.get("content", []) if part.get("type") == "text")
                if content:
                    messages.append(Message(role="user" if item.get("sender") == "human" else "assistant", content=content, created_at=item.get("created_at")))
            conversations.append(cls(chat_id=record.get("uuid") or uuid4().hex, created_at=record.get("created_at"), messages=messages))
        return conversations

    @classmethod
    def from_huggingface(
        cls,
        dataset: str,
        *,
        split: str = "train",
        limit: int | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> list[Conversation]:
        """Load a Hugging Face dataset; transform maps a row to this schema."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the huggingface extra to load datasets") from exc
        rows = load_dataset(dataset, split=split, streaming=True)
        if limit is not None:
            rows = rows.take(limit)
        return [cls.model_validate(transform(row) if transform else row) for row in rows]


class Summary(BaseModel):
    chat_id: str
    summary: str
    request: str | None = None
    topic: str | None = None
    languages: list[str] = Field(default_factory=list)
    task: str | None = None
    concerning_score: int | None = Field(default=None, ge=1, le=5)
    user_frustration: int | None = Field(default=None, ge=1, le=5)
    assistant_errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Cluster(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str
    slug: str
    chat_ids: list[str]
    parent_id: str | None = None
    level: int = 0
    x_coord: float | None = None
    y_coord: float | None = None

    @property
    def count(self) -> int:
        return len(self.chat_ids)


class SummaryPayload(BaseModel):
    summary: str
    request: str | None = None
    topic: str | None = None
    languages: list[str] = Field(default_factory=list)
    task: str | None = None
    concerning_score: int | None = Field(default=None, ge=1, le=5)
    user_frustration: int | None = Field(default=None, ge=1, le=5)
    assistant_errors: list[str] = Field(default_factory=list)


class LabelPayload(BaseModel):
    name: str
    description: str
