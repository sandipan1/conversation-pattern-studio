"""Analyze a small real-data sample from allenai/WildChat."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pattern_studio import Conversation, Message, PipelineConfig, analyze
from pattern_studio.providers import FileCache, OpenAIEmbedder, OpenAITextModel, load_environment
from pattern_studio.storage import JSONLStore


def load_wildchat_sample(limit: int) -> list[Conversation]:
    """Stream WildChat and convert its conversation rows to the app's schema."""
    dataset = load_dataset("allenai/WildChat", split="train", streaming=True).shuffle(
        seed=42, buffer_size=1_000,
    )
    chats: list[Conversation] = []
    role_map = {"human": "user", "gpt": "assistant"}

    for row in dataset:
        messages = []
        for turn in row.get("conversation") or []:
            role = role_map.get(str(turn.get("role", "")).lower(), str(turn.get("role", "")).lower())
            content = turn.get("content")
            if role in {"user", "assistant", "system", "tool"} and isinstance(content, str) and content.strip():
                messages.append(Message(role=role, content=content.strip()))
        chat_id = row.get("conversation_id")
        if not chat_id or not messages:
            continue

        metadata = {
            key: row[key]
            for key in ("model", "language", "turn", "toxic", "redacted")
            if row.get(key) is not None
        }
        chats.append(Conversation(
            chat_id=str(chat_id),
            created_at=row.get("timestamp"),
            messages=messages,
            metadata=metadata,
        ))
        if len(chats) >= limit:
            break

    if not chats:
        raise RuntimeError("No usable WildChat conversations were found in the train split")
    return chats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120, help="Number of valid conversations to analyze (default: 120)")
    parser.add_argument("--run-name", help="Checkpoint folder name (defaults to wildchat-<limit>)")
    parser.add_argument("--model", default="gpt-6-luna", help="OpenAI-compatible chat model")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel model requests")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    load_environment()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in .env or the environment before running this real-data example")

    conversations = load_wildchat_sample(args.limit)
    sample_file = Path(__file__).with_name(f"conversations-{len(conversations)}.jsonl")
    sample_file.write_text(
        "".join(json.dumps(chat.model_dump(mode="json"), ensure_ascii=False) + "\n" for chat in conversations),
        encoding="utf-8",
    )
    requested_name = args.run_name or f"wildchat-{len(conversations)}"
    run_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", requested_name).strip("-_").lower()
    if not run_name:
        parser.error("--run-name must contain at least one letter or number")
    checkpoints = Path("checkpoints") / run_name
    text_model = OpenAITextModel(args.model, cache=FileCache(".cache/pattern-studio"))
    embedder = OpenAIEmbedder("text-embedding-3-small")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(), console=Console(stderr=True),
    ) as meter:
        task_ids: dict[str, int] = {}

        def show_progress(stage: str, completed: int, total: int) -> None:
            if stage not in task_ids:
                task_ids[stage] = meter.add_task(stage, total=max(total, 1))
            meter.update(task_ids[stage], completed=completed if total else 1, total=max(total, 1))

        result = asyncio.run(analyze(
            conversations,
            text_model=text_model,
            embedder=embedder,
            config=PipelineConfig(max_depth=2, concurrency=args.concurrency),
            checkpoints=JSONLStore(checkpoints),
            on_progress=show_progress,
        ))

    print(
        f"Analyzed {len(result.conversations)} WildChat conversations into "
        f"{len(result.clusters)} base groups and {len(result.meta_clusters)} higher-level groups."
    )
    print(f"Sample conversations: {sample_file}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Dashboard: uv run pattern-studio serve --dir {checkpoints}")


if __name__ == "__main__":
    main()
