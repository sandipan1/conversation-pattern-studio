"""Command line entry points for analysis and the dashboard."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .models import Conversation
from .pipeline import PipelineConfig, analyze
from .providers import (
    CohereEmbedder,
    FileCache,
    HashingEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    OpenAITextModel,
    RuleBasedTextModel,
    load_environment,
)
from .storage import HuggingFaceStore, JSONLStore, ParquetStore, SQLiteStore, SQLStore
from .webapp import create_app

app = typer.Typer(help="Find and explore patterns in conversation data.", no_args_is_help=True)


@app.command()
def run(
    input_file: Annotated[Path, typer.Argument(help="JSON or JSONL conversation file")],
    output: Annotated[Path, typer.Option(help="Root directory for run-specific checkpoints")] = Path("checkpoints"),
    run_name: Annotated[str | None, typer.Option(help="Checkpoint folder name; defaults to the input filename")]=None,
    input_format: Annotated[str, typer.Option(help="standard or claude")] = "standard",
    checkpoint_format: Annotated[str, typer.Option(help="jsonl, sqlite, sql, parquet, or huggingface")] = "jsonl",
    database_url: Annotated[str | None, typer.Option(help="SQLAlchemy URL when using --checkpoint-format sql")] = None,
    model: Annotated[str, typer.Option(help="OpenAI-compatible chat model")] = "gpt-6-luna",
    embedding_model: Annotated[str, typer.Option(help="Embedding model for API or local option")] = "text-embedding-3-small",
    embeddings: Annotated[str, typer.Option(help="openai, cohere, local, or hashing")] = "openai",
    offline: Annotated[bool, typer.Option(help="Use simple local labels for a no-key preview")] = False,
    clustering: Annotated[str, typer.Option(help="kmeans or hdbscan")] = "kmeans",
    clusters: Annotated[int | None, typer.Option(help="Number of base groups; automatic by default")] = None,
    max_depth: Annotated[int, typer.Option(help="Maximum hierarchy depth")] = 2,
    concurrency: Annotated[int, typer.Option(help="Parallel model requests; increase if your provider allows it")] = 10,
    resume: Annotated[bool, typer.Option(help="Resume compatible checkpoints")] = True,
) -> None:
    """Analyze conversations and save every stage for the dashboard."""
    load_environment()
    if input_format not in {"standard", "claude"}:
        raise typer.BadParameter("input-format must be standard or claude")
    if concurrency < 1:
        raise typer.BadParameter("concurrency must be at least 1")
    if not offline and not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("Set OPENAI_API_KEY in .env or the environment, or use --offline for a preview")
    conversations = (
        Conversation.from_claude_export(input_file)
        if input_format == "claude"
        else Conversation.from_file(input_file)
    )
    default_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", input_file.stem).strip("-_").lower() or "run"
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name or default_name).strip("-_").lower()
    if not name:
        raise typer.BadParameter("--run-name must contain at least one letter or number")
    run_directory = output / name
    stores = {
        "jsonl": lambda: JSONLStore(run_directory),
        "sqlite": lambda: SQLiteStore(run_directory.with_suffix(".sqlite3")),
        "sql": lambda: SQLStore(database_url or "", run_id=name),
        "parquet": lambda: ParquetStore(run_directory),
        "huggingface": lambda: HuggingFaceStore(run_directory),
    }
    if checkpoint_format not in stores:
        raise typer.BadParameter("Unsupported checkpoint format")
    if checkpoint_format == "sql" and not database_url:
        raise typer.BadParameter("--database-url is required for SQL checkpoints")
    if embeddings not in {"openai", "cohere", "local", "hashing"}:
        raise typer.BadParameter("embeddings must be openai, cohere, local, or hashing")
    text_model = RuleBasedTextModel() if offline else OpenAITextModel(model, cache=FileCache(".cache/pattern-studio"))
    if offline or embeddings == "hashing":
        embedder = HashingEmbedder()
    elif embeddings == "local":
        embedder = LocalEmbedder(embedding_model)
    elif embeddings == "cohere":
        embedder = CohereEmbedder(embedding_model if embedding_model != "text-embedding-3-small" else "embed-v4.0")
    else:
        embedder = OpenAIEmbedder(embedding_model)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
        refresh_per_second=5,
    ) as meter:
        tasks: dict[str, int] = {}

        def show_progress(stage: str, completed: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = meter.add_task(stage, total=max(total, 1))
            meter.update(tasks[stage], completed=completed if total else 1, total=max(total, 1))

        result = asyncio.run(analyze(
            conversations,
            text_model=text_model,
            embedder=embedder,
            config=PipelineConfig(
                cluster_count=clusters, clustering=clustering,
                max_depth=max_depth, concurrency=concurrency,
            ),
            checkpoints=stores[checkpoint_format](),
            resume=resume,
            on_progress=show_progress,
        ))
    typer.echo(f"Analyzed {len(result.conversations)} conversations into {len(result.clusters)} base groups and {len(result.meta_clusters)} higher-level groups.")
    if checkpoint_format == "jsonl":
        typer.echo(f"Checkpoints: {run_directory}")
        typer.echo(f"Open the dashboard with: pattern-studio serve --dir {run_directory}")


@app.command()
def serve(
    directory: Annotated[Path, typer.Option("--dir", help="JSONL checkpoint directory")] = Path("checkpoints"),
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port")] = 8000,
) -> None:
    """Open the local dashboard; files can also be imported in the browser."""
    import uvicorn

    typer.echo(f"Dashboard: http://{host}:{port}")
    uvicorn.run(create_app(directory), host=host, port=port)


@app.command()
def inspect(
    directory: Annotated[Path, typer.Option("--dir", help="JSONL checkpoint directory")] = Path("checkpoints"),
) -> None:
    """Print the theme hierarchy from saved JSONL checkpoints."""
    from .models import Cluster

    store = JSONLStore(directory)
    rows = store.read("dimensionality") or store.read("clusters") or []
    clusters = [Cluster.model_validate(row) for row in rows]
    if not clusters:
        typer.echo("No clusters found in this directory.")
        return
    children: dict[str | None, list[Cluster]] = {}
    for cluster in clusters:
        children.setdefault(cluster.parent_id, []).append(cluster)
    known_ids = {cluster.id for cluster in clusters}
    roots = [cluster for cluster in clusters if cluster.parent_id not in known_ids]

    def show(cluster: Cluster, indent: int = 0) -> None:
        typer.echo(f"{'  ' * indent}{cluster.name} ({cluster.count})")
        for child in sorted(children.get(cluster.id, []), key=lambda item: -item.count):
            show(child, indent + 1)

    for root in sorted(roots, key=lambda item: -item.count):
        show(root)


if __name__ == "__main__":
    app()
