"""Interchangeable checkpoint stores for pipeline stages."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

STAGES = ("manifest", "conversations", "summaries", "clusters", "meta_clusters", "dimensionality")


class CheckpointStore(Protocol):
    def write(self, stage: str, records: list[dict[str, Any]]) -> None: ...
    def read(self, stage: str) -> list[dict[str, Any]] | None: ...


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown checkpoint stage: {stage}")


class JSONLStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        _check_stage(stage)
        return self.directory / f"{stage}.jsonl"

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        path = self.path(stage)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        temporary.replace(path)

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        path = self.path(stage)
        if not path.exists():
            return None
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.execute("CREATE TABLE IF NOT EXISTS checkpoints (stage TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        _check_stage(stage)
        with sqlite3.connect(self.path) as database:
            database.execute("INSERT OR REPLACE INTO checkpoints(stage, payload) VALUES (?, ?)", (stage, json.dumps(records, default=str)))

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        _check_stage(stage)
        with sqlite3.connect(self.path) as database:
            row = database.execute("SELECT payload FROM checkpoints WHERE stage = ?", (stage,)).fetchone()
        return json.loads(row[0]) if row else None


class ParquetStore:
    def __init__(self, directory: str | Path):
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install the parquet extra to use Parquet checkpoints") from exc
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        _check_stage(stage)
        # Store each record as JSON to preserve optional and nested fields losslessly.
        table = pa.table({"record": [json.dumps(record, default=str) for record in records]})
        path = self.directory / f"{stage}.parquet"
        temporary = path.with_suffix(".tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        import pyarrow.parquet as pq

        _check_stage(stage)
        path = self.directory / f"{stage}.parquet"
        return [json.loads(record) for record in pq.read_table(path)["record"].to_pylist()] if path.exists() else None


class HuggingFaceStore:
    def __init__(self, directory: str | Path):
        try:
            import datasets  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install the huggingface extra to use dataset checkpoints") from exc
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        from datasets import Dataset

        _check_stage(stage)
        path = self.directory / stage
        if path.exists():
            import shutil

            shutil.rmtree(path)
        Dataset.from_dict({"record": [json.dumps(record, default=str) for record in records]}).save_to_disk(str(path))

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        from datasets import load_from_disk

        _check_stage(stage)
        path = self.directory / stage
        return [json.loads(row["record"]) for row in load_from_disk(str(path))] if path.exists() else None


class MultiStore:
    def __init__(self, *stores: CheckpointStore):
        self.stores = stores

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        for store in self.stores:
            store.write(stage, records)

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        for store in self.stores:
            result = store.read(stage)
            if result is not None:
                return result
        return None


class SQLStore:
    """SQL checkpoints with independent named runs on SQLAlchemy database URLs."""

    def __init__(self, database_url: str, *, run_id: str = "default"):
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                LargeBinary,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
                insert,
                select,
            )
        except ImportError as exc:
            raise RuntimeError("Install the sql extra to use SQL checkpoints") from exc
        from datetime import datetime, timezone

        self.engine = create_engine(database_url)
        metadata = MetaData()
        self.runs = Table(
            "pattern_runs", metadata,
            Column("id", String(64), primary_key=True),
            Column("name", String(255), nullable=False),
            Column("description", Text),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        self.checkpoints = Table(
            "pattern_checkpoints", metadata,
            Column("run_id", String(64), primary_key=True),
            Column("stage", String(40), primary_key=True),
            Column("payload", LargeBinary, nullable=False),
        )
        metadata.create_all(self.engine)
        self.run_id = run_id
        with self.engine.begin() as connection:
            if connection.execute(select(self.runs.c.id).where(self.runs.c.id == run_id)).first() is None:
                connection.execute(insert(self.runs).values(id=run_id, name=run_id, created_at=datetime.now(timezone.utc)))

    def create_run(self, name: str, description: str | None = None) -> str:
        from datetime import datetime, timezone
        from uuid import uuid4

        from sqlalchemy import insert

        run_id = uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(insert(self.runs).values(id=run_id, name=name, description=description, created_at=datetime.now(timezone.utc)))
        self.run_id = run_id
        return run_id

    def list_runs(self) -> list[dict[str, Any]]:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            rows = connection.execute(select(self.runs).order_by(self.runs.c.created_at.desc())).mappings().all()
        return [dict(row) for row in rows]

    def use_run(self, run_id: str) -> None:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            exists = connection.execute(select(self.runs.c.id).where(self.runs.c.id == run_id)).first()
        if exists is None:
            raise ValueError(f"Unknown run: {run_id}")
        self.run_id = run_id

    def write(self, stage: str, records: list[dict[str, Any]]) -> None:
        from sqlalchemy import delete, insert

        _check_stage(stage)
        payload = json.dumps(records, ensure_ascii=False, default=str).encode("utf-8")
        with self.engine.begin() as connection:
            connection.execute(delete(self.checkpoints).where(self.checkpoints.c.run_id == self.run_id, self.checkpoints.c.stage == stage))
            connection.execute(insert(self.checkpoints).values(run_id=self.run_id, stage=stage, payload=payload))

    def read(self, stage: str) -> list[dict[str, Any]] | None:
        from sqlalchemy import select

        _check_stage(stage)
        with self.engine.connect() as connection:
            payload = connection.execute(select(self.checkpoints.c.payload).where(self.checkpoints.c.run_id == self.run_id, self.checkpoints.c.stage == stage)).scalar_one_or_none()
        return json.loads(payload) if payload is not None else None
