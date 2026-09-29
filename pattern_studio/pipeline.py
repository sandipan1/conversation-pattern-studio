"""Conversation summarization, semantic grouping, hierarchy, and projection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .models import Cluster, Conversation, Summary, SummaryPayload
from .providers import Embedder, TextModel
from .storage import CheckpointStore


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
    return min(size, max(1, round(math.sqrt(size / 2))))


async def summarize_conversations(
    conversations: list[Conversation], model: TextModel, *, concurrency: int = 10
) -> list[Summary]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(conversation: Conversation) -> Summary:
        async with semaphore:
            generated = await model.summarize(conversation)
            fields = generated.model_dump()
            known = set(SummaryPayload.model_fields)
            custom = {key: fields.pop(key) for key in list(fields) if key not in known}
            return Summary(chat_id=conversation.chat_id, metadata={**conversation.metadata, **custom}, **fields)

    return list(await asyncio.gather(*(one(item) for item in conversations)))


async def embed_summaries(summaries: list[Summary], embedder: Embedder, *, batch_size: int = 100) -> list[Summary]:
    missing = [item for item in summaries if item.embedding is None]
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        vectors = await embedder.embed([item.request or item.summary for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding model returned a different number of vectors")
        for item, vector in zip(batch, vectors):
            item.embedding = vector
    return summaries


async def _label_groups(
    groups: list[list[Summary]], model: TextModel, concurrency: int, level: int
) -> list[Cluster]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, members: list[Summary]) -> Cluster:
        other = [item.summary for j, group in enumerate(groups) if j != index for item in group[:2]][:10]
        async with semaphore:
            label = await model.label([item.summary for item in members[:20]], other)
        return Cluster(
            name=label.name,
            description=label.description,
            slug=_slug(label.name),
            chat_ids=[item.chat_id for item in members],
            level=level,
        )

    return list(await asyncio.gather(*(one(i, group) for i, group in enumerate(groups))))


async def cluster_summaries(
    summaries: list[Summary], model: TextModel, config: PipelineConfig
) -> list[Cluster]:
    if not summaries:
        return []
    if any(item.embedding is None for item in summaries):
        raise ValueError("Summaries must have embeddings before clustering")
    labels = _group(
        [item.embedding for item in summaries if item.embedding is not None],
        config.cluster_count or _automatic_cluster_count(len(summaries)),
        config.clustering,
    )
    groups = [[item for item, label in zip(summaries, labels) if label == group] for group in sorted(set(labels))]
    return await _label_groups(groups, model, config.concurrency, level=0)


async def build_hierarchy(
    base_clusters: list[Cluster], model: TextModel, embedder: Embedder, config: PipelineConfig
) -> list[Cluster]:
    if config.max_depth == 0 or len(base_clusters) <= 1:
        return []
    current = base_clusters
    parents: list[Cluster] = []
    for level in range(1, config.max_depth + 1):
        if len(current) <= 1:
            break
        vectors = await embedder.embed([f"{item.name}. {item.description}" for item in current])
        group_count = math.ceil(len(current) / config.children_per_parent)
        labels = _group(vectors, group_count, "kmeans")
        groups = [[item for item, label in zip(current, labels) if label == group] for group in sorted(set(labels))]
        semaphore = asyncio.Semaphore(config.concurrency)

        async def make_parent(index: int, members: list[Cluster], group_collection=groups, limiter=semaphore, depth=level) -> Cluster:
            contrast = [item.name for j, group in enumerate(group_collection) if j != index for item in group][:10]
            async with limiter:
                label = await model.label([f"{item.name}: {item.description}" for item in members], contrast)
            chat_ids = list(dict.fromkeys(chat_id for item in members for chat_id in item.chat_ids))
            parent = Cluster(name=label.name, description=label.description, slug=_slug(label.name), chat_ids=chat_ids, level=depth)
            for child in members:
                child.parent_id = parent.id
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
) -> AnalysisResult:
    """Run the complete analysis. Checkpoints resume only when inputs and config match."""
    config = config or PipelineConfig()
    ids = [item.chat_id for item in conversations]
    if len(ids) != len(set(ids)):
        raise ValueError("Conversation chat_id values must be unique")
    fingerprint = hashlib.sha256(json.dumps({
        "input": [item.model_dump(mode="json") for item in conversations],
        "config": config.model_dump(),
        "text_model": getattr(text_model, "identity", getattr(text_model, "model", type(text_model).__name__)),
        "embedder": getattr(embedder, "model", type(embedder).__name__),
    }, sort_keys=True, default=str).encode()).hexdigest()
    valid = bool(resume and checkpoints and checkpoints.read("manifest") == [{"fingerprint": fingerprint}])

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
        summaries = await summarize_conversations(conversations, text_model, concurrency=config.concurrency)
        await embed_summaries(summaries, embedder)
        save("summaries", summaries)
    elif any(item.embedding is None for item in summaries):
        await embed_summaries(summaries, embedder)
        save("summaries", summaries)

    base_clusters = load("clusters", Cluster)
    if base_clusters is None:
        base_clusters = await cluster_summaries(summaries, text_model, config)
        save("clusters", base_clusters)

    meta_clusters = load("meta_clusters", Cluster)
    if meta_clusters is None:
        meta_clusters = await build_hierarchy(base_clusters, text_model, embedder, config)
        save("clusters", base_clusters)  # parent links are assigned during hierarchy construction
        save("meta_clusters", meta_clusters)

    projected = load("dimensionality", Cluster)
    if projected is None:
        all_clusters = base_clusters + meta_clusters
        vectors = await embedder.embed([f"{item.name}. {item.description}" for item in all_clusters])
        projected = project_clusters(all_clusters, vectors, config.projection)
        save("dimensionality", projected)
    if checkpoints and not valid:
        checkpoints.write("manifest", [{"fingerprint": fingerprint}])
    return AnalysisResult(conversations, summaries, base_clusters, meta_clusters, projected)
