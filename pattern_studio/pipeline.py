"""Conversation summarization, semantic grouping, hierarchy, and projection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .models import Cluster, Conversation, Summary, SummaryPayload
from .providers import Embedder, TextModel
from .storage import CheckpointStore

ProgressCallback = Callable[[str, int, int], None]


def _report(callback: ProgressCallback | None, stage: str, completed: int, total: int) -> None:
    if callback is not None:
        callback(stage, completed, total)


class PipelineConfig(BaseModel):
    cluster_count: int | None = Field(default=None, ge=1)
    clustering: Literal["kmeans", "hdbscan"] = "kmeans"
    max_depth: int = Field(default=2, ge=0)
    children_per_parent: int = Field(default=8, ge=2)
    concurrency: int = Field(default=10, ge=1)
    projection: Literal["auto", "pca", "umap"] = "auto"


@dataclass
class AnalysisResult:
    conversations: list[Conversation]
    summaries: list[Summary]
    clusters: list[Cluster]
    meta_clusters: list[Cluster]
    projected_clusters: list[Cluster]

    def as_dict(self) -> dict:
        return {
            "conversations": [item.model_dump(mode="json") for item in self.conversations],
            "summaries": [item.model_dump(mode="json") for item in self.summaries],
            "clusters": [item.model_dump(mode="json") for item in self.clusters],
            "meta_clusters": [item.model_dump(mode="json") for item in self.meta_clusters],
            "dimensionality": [item.model_dump(mode="json") for item in self.projected_clusters],
        }


def _slug(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return "_".join(words[:5]) or "other_requests"


def _group(vectors: list[list[float]], count: int, method: str) -> list[int]:
    if not vectors:
        return []
    if len(vectors) == 1 or count == 1:
        return [0] * len(vectors)
    matrix = np.asarray(vectors, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values")
    if method == "hdbscan":
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise RuntimeError("Install the hdbscan extra to use density clustering") from exc
        labels = HDBSCAN(min_cluster_size=max(2, min(5, len(vectors) // 3))).fit_predict(matrix).tolist()
        # Give noise points one shared group so none are lost.
        return [label if label >= 0 else max(labels) + 1 for label in labels]
    return KMeans(n_clusters=min(count, len(vectors)), random_state=42, n_init=10).fit_predict(matrix).tolist()


def _automatic_cluster_count(size: int) -> int:
    # Aim for several parent themes without making tiny datasets mostly singletons.
    return min(size, max(1, min(size // 3, round(2 * math.sqrt(size)))))


async def summarize_conversations(
    conversations: list[Conversation], model: TextModel, *, concurrency: int = 10,
    on_progress: ProgressCallback | None = None,
) -> list[Summary]:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    stage = "Summarizing conversations"
    _report(on_progress, stage, 0, len(conversations))

    async def one(conversation: Conversation) -> Summary:
        nonlocal completed
        async with semaphore:
            generated = await model.summarize(conversation)
            fields = generated.model_dump()
            known = set(SummaryPayload.model_fields)
            custom = {key: fields.pop(key) for key in list(fields) if key not in known}
            summary = Summary(chat_id=conversation.chat_id, metadata={**conversation.metadata, **custom}, **fields)
            completed += 1
            _report(on_progress, stage, completed, len(conversations))
            return summary

    return list(await asyncio.gather(*(one(item) for item in conversations)))


async def embed_summaries(
    summaries: list[Summary], embedder: Embedder, *, batch_size: int = 100,
    concurrency: int = 4, on_progress: ProgressCallback | None = None,
) -> list[Summary]:
    missing = [item for item in summaries if item.embedding is None]
    completed = 0
    stage = "Embedding requests"
    _report(on_progress, stage, 0, len(missing))
    semaphore = asyncio.Semaphore(concurrency)

    async def one_batch(offset: int) -> None:
        nonlocal completed
        batch = missing[offset : offset + batch_size]
        async with semaphore:
            vectors = await embedder.embed([item.request or item.summary for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding model returned a different number of vectors")
        for item, vector in zip(batch, vectors):
            item.embedding = vector
        completed += len(batch)
        _report(on_progress, stage, completed, len(missing))

    await asyncio.gather(*(one_batch(offset) for offset in range(0, len(missing), batch_size)))
    return summaries


async def _label_groups(
    groups: list[list[Summary]], model: TextModel, concurrency: int, level: int,
    on_progress: ProgressCallback | None = None,
) -> list[Cluster]:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    stage = "Naming base patterns"
    _report(on_progress, stage, 0, len(groups))

    async def one(index: int, members: list[Summary]) -> Cluster:
        nonlocal completed
        other = [item.summary for j, group in enumerate(groups) if j != index for item in group[:2]][:10]
        async with semaphore:
            label = await model.label([item.summary for item in members[:20]], other)
        cluster = Cluster(
            name=label.name,
            description=label.description,
            slug=_slug(label.name),
            chat_ids=[item.chat_id for item in members],
            level=level,
        )
        completed += 1
        _report(on_progress, stage, completed, len(groups))
        return cluster

    return list(await asyncio.gather(*(one(i, group) for i, group in enumerate(groups))))


async def cluster_summaries(
    summaries: list[Summary], model: TextModel, config: PipelineConfig,
    on_progress: ProgressCallback | None = None,
) -> list[Cluster]:
    if not summaries:
        return []
    if any(item.embedding is None for item in summaries):
        raise ValueError("Summaries must have embeddings before clustering")
    _report(on_progress, "Grouping conversations", 0, 1)
    labels = _group(
        [item.embedding for item in summaries if item.embedding is not None],
        config.cluster_count or _automatic_cluster_count(len(summaries)),
        config.clustering,
    )
    _report(on_progress, "Grouping conversations", 1, 1)
    groups = [[item for item, label in zip(summaries, labels) if label == group] for group in sorted(set(labels))]
    return await _label_groups(groups, model, config.concurrency, level=0, on_progress=on_progress)


async def build_hierarchy(
    base_clusters: list[Cluster], model: TextModel, embedder: Embedder, config: PipelineConfig,
    on_progress: ProgressCallback | None = None,
) -> list[Cluster]:
    if config.max_depth == 0 or len(base_clusters) <= 1:
        return []
    current = base_clusters
    parents: list[Cluster] = []
    for level in range(1, config.max_depth + 1):
        if len(current) <= 1:
            break
        stage = f"Embedding level {level + 1} themes"
        _report(on_progress, stage, 0, 1)
        vectors = await embedder.embed([f"{item.name}. {item.description}" for item in current])
        _report(on_progress, stage, 1, 1)
        group_count = math.ceil(len(current) / config.children_per_parent)
        labels = _group(vectors, group_count, "kmeans")
        groups = [[item for item, label in zip(current, labels) if label == group] for group in sorted(set(labels))]
        semaphore = asyncio.Semaphore(config.concurrency)
        completed = 0
        label_stage = f"Naming level {level + 1} themes"
        group_total = len(groups)
        _report(on_progress, label_stage, 0, group_total)

        async def make_parent(
            index: int, members: list[Cluster], group_collection=groups,
            limiter=semaphore, depth=level, stage_name=label_stage,
            total_groups=group_total,
        ) -> Cluster:
            nonlocal completed
            contrast = [item.name for j, group in enumerate(group_collection) if j != index for item in group][:10]
            async with limiter:
                label = await model.label([f"{item.name}: {item.description}" for item in members], contrast)
            chat_ids = list(dict.fromkeys(chat_id for item in members for chat_id in item.chat_ids))
            parent = Cluster(name=label.name, description=label.description, slug=_slug(label.name), chat_ids=chat_ids, level=depth)
            for child in members:
                child.parent_id = parent.id
            completed += 1
            _report(on_progress, stage_name, completed, total_groups)
            return parent

        current = list(await asyncio.gather(*(make_parent(i, group) for i, group in enumerate(groups))))
        parents.extend(current)
    return parents


def project_clusters(clusters: list[Cluster], embedder_vectors: list[list[float]], method: str = "auto") -> list[Cluster]:
    if len(clusters) != len(embedder_vectors):
        raise ValueError("Each cluster needs one embedding")
    if not clusters:
        return []
    coordinates = np.zeros((len(clusters), 2), dtype=float)
    for level in sorted({item.level for item in clusters}):
        indices = [index for index, item in enumerate(clusters) if item.level == level]
        if len(indices) < 2:
            continue
        matrix = np.asarray([embedder_vectors[index] for index in indices], dtype=float)
        result = None
        if method in {"auto", "umap"} and len(indices) >= 4:
            try:
                from umap import UMAP
            except ImportError:
                if method == "umap":
                    raise RuntimeError("Install the projection extra to use UMAP") from None
            else:
                result = UMAP(n_components=2, n_neighbors=min(10, len(indices) - 1), random_state=42).fit_transform(matrix)
        if result is None:
            components = min(2, *matrix.shape)
            result = PCA(n_components=components).fit_transform(matrix)
        coordinates[indices, : result.shape[1]] = result
    projected = [item.model_copy(deep=True) for item in clusters]
    for item, point in zip(projected, coordinates):
        item.x_coord = float(point[0])
        item.y_coord = float(point[1])
    return projected


async def analyze(
    conversations: list[Conversation],
    *,
    text_model: TextModel,
    embedder: Embedder,
    config: PipelineConfig | None = None,
    checkpoints: CheckpointStore | None = None,
    resume: bool = True,
    on_progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """Run the complete analysis. Checkpoints resume only when inputs and config match."""
    config = config or PipelineConfig()
    ids = [item.chat_id for item in conversations]
    if len(ids) != len(set(ids)):
        raise ValueError("Conversation chat_id values must be unique")
    signature = {
        "input": [item.model_dump(mode="json") for item in conversations],
        "text_model": getattr(text_model, "identity", getattr(text_model, "model", type(text_model).__name__)),
        "embedder": getattr(embedder, "model", type(embedder).__name__),
    }

    def fingerprint_for(settings: dict) -> str:
        payload = {**signature, "config": settings}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    settings = config.model_dump()
    fingerprint = fingerprint_for({key: value for key, value in settings.items() if key != "concurrency"})
    legacy_fingerprints = {fingerprint_for(settings)}
    if config.concurrency != 10:
        legacy_fingerprints.add(fingerprint_for({**settings, "concurrency": 10}))
    manifest = checkpoints.read("manifest") if resume and checkpoints else None
    saved_fingerprint = (
        manifest[0].get("fingerprint")
        if isinstance(manifest, list) and len(manifest) == 1 and isinstance(manifest[0], dict)
        else None
    )
    valid = bool(resume and checkpoints and saved_fingerprint in {fingerprint, *legacy_fingerprints})

    def load(stage: str, record_type: type):
        records = checkpoints.read(stage) if valid and checkpoints else None
        return [record_type.model_validate(row) for row in records] if records is not None else None

    def save(stage: str, records: list[BaseModel]):
        if checkpoints:
            checkpoints.write(stage, [record.model_dump(mode="json") for record in records])

    if checkpoints and not valid:
        save("conversations", conversations)

    summaries = load("summaries", Summary)
    if summaries is None:
        summaries = await summarize_conversations(
            conversations, text_model, concurrency=config.concurrency, on_progress=on_progress,
        )
        await embed_summaries(summaries, embedder, concurrency=config.concurrency, on_progress=on_progress)
        save("summaries", summaries)
    else:
        _report(on_progress, "Reused saved summaries", 1, 1)
        if any(item.embedding is None for item in summaries):
            await embed_summaries(summaries, embedder, concurrency=config.concurrency, on_progress=on_progress)
            save("summaries", summaries)
        else:
            _report(on_progress, "Reused saved embeddings", 1, 1)

    base_clusters = load("clusters", Cluster)
    if base_clusters is None:
        base_clusters = await cluster_summaries(summaries, text_model, config, on_progress=on_progress)
        save("clusters", base_clusters)
    else:
        _report(on_progress, "Reused saved base patterns", 1, 1)

    meta_clusters = load("meta_clusters", Cluster)
    if meta_clusters is None:
        meta_clusters = await build_hierarchy(base_clusters, text_model, embedder, config, on_progress=on_progress)
        save("clusters", base_clusters)  # parent links are assigned during hierarchy construction
        save("meta_clusters", meta_clusters)
    else:
        _report(on_progress, "Reused saved higher themes", 1, 1)

    projected = load("dimensionality", Cluster)
    if projected is None:
        _report(on_progress, "Placing patterns on map", 0, 1)
        all_clusters = base_clusters + meta_clusters
        vectors = await embedder.embed([f"{item.name}. {item.description}" for item in all_clusters])
        projected = project_clusters(all_clusters, vectors, config.projection)
        _report(on_progress, "Placing patterns on map", 1, 1)
        save("dimensionality", projected)
    else:
        _report(on_progress, "Reused saved map", 1, 1)
    if checkpoints and (not valid or saved_fingerprint != fingerprint):
        checkpoints.write("manifest", [{"fingerprint": fingerprint}])
    return AnalysisResult(conversations, summaries, base_clusters, meta_clusters, projected)
