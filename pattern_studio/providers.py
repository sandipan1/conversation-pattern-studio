"""Adapters for text generation, embeddings, and local result caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from .models import Conversation, LabelPayload, SummaryPayload


class TextModel(Protocol):
    async def summarize(self, conversation: Conversation) -> SummaryPayload: ...
    async def label(self, examples: list[str], others: list[str]) -> LabelPayload: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class FileCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, key: object) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
        return self.path / f"{digest}.json"

    def get(self, key: object) -> dict | None:
        file = self._file(key)
        return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None

    def set(self, key: object, value: dict) -> None:
        file = self._file(key)
        temporary = file.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(file)


class OpenAITextModel:
    """Structured text generation via OpenAI or an OpenAI-compatible endpoint."""

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None, base_url: str | None = None, cache: FileCache | None = None, summary_instruction: str | None = None, label_instruction: str | None = None, summary_schema: type[SummaryPayload] = SummaryPayload):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or ("local" if base_url or os.getenv("OPENAI_BASE_URL") else None), base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self.cache = cache
        self.summary_instruction = summary_instruction
        self.label_instruction = label_instruction
        self.summary_schema = summary_schema
        self.identity = [model, summary_instruction, label_instruction, summary_schema.model_json_schema()]

    async def _json(self, system: str, user: str, cache_key: object, schema: type[SummaryPayload | LabelPayload]) -> SummaryPayload | LabelPayload:
        key = [self.model, system, cache_key]
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            try:
                return schema.model_validate(cached)
            except ValueError:
                pass
        for attempt in range(3):
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = response.choices[0].message.content
            try:
                result = schema.model_validate_json(content or "")
            except ValueError as exc:
                if attempt == 2:
                    raise ValueError("The model did not return a valid structured result after three attempts") from exc
                continue
            if self.cache:
                self.cache.set(key, result.model_dump(mode="json"))
            return result
        raise AssertionError("unreachable")

    async def summarize(self, conversation: Conversation) -> SummaryPayload:
        dialogue = "\n".join(f"{message.role}: {message.content}" for message in conversation.messages)
        system = (
            "Analyze the conversation for recurring user needs. Return a JSON object with keys "
            "summary, request, topic, languages, task, concerning_score, user_frustration, "
            "assistant_errors. Keep the summary to two sentences. Do not include direct personal "
            "identifiers such as names, addresses, phone numbers, or email addresses. "
            "Scores are integers from 1 to 5; errors and languages are arrays."
        )
        if self.summary_instruction:
            system += " " + self.summary_instruction
        system += " Respond according to this JSON schema: " + json.dumps(self.summary_schema.model_json_schema())
        result = await self._json(system, dialogue, ["summary", dialogue], self.summary_schema)
        return result

    async def label(self, examples: list[str], others: list[str]) -> LabelPayload:
        system = (
            "Name the shared user need in the examples. Return a JSON object with 'name' "
            "(a concise action phrase) and 'description' (one or two specific sentences). "
            "Use the unrelated examples to make the name distinctive. Avoid direct personal identifiers."
        )
        if self.label_instruction:
            system += " " + self.label_instruction
        prompt = json.dumps({"in_group": examples[:20], "other_groups": others[:10]}, ensure_ascii=False)
        result = await self._json(system, prompt, ["label", examples, others], LabelPayload)
        return result


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", *, api_key: str | None = None, base_url: str | None = None):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or ("local" if base_url or os.getenv("OPENAI_BASE_URL") else None), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]


class CohereEmbedder:
    """Optional semantic embeddings using Cohere's clustering input mode."""

    def __init__(self, model: str = "embed-v4.0", *, api_key: str | None = None):
        try:
            import cohere
        except ImportError as exc:
            raise RuntimeError("Install the cohere extra to use this option") from exc
        self.model = model
        self.client = cohere.ClientV2(api_key=api_key or os.getenv("COHERE_API_KEY"))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 96):
            response = await asyncio.to_thread(
                self.client.embed,
                texts=texts[offset : offset + 96],
                model=self.model,
                input_type="clustering",
                embedding_types=["float"],
            )
            vectors.extend(response.embeddings.float)
        return vectors


class LocalEmbedder:
    """Optional on-device semantic embeddings through sentence-transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the local-embeddings extra to use this option") from exc
        self.model = SentenceTransformer(model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class HashingEmbedder:
    """Offline fallback for local experiments; lexical rather than semantic."""

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()


class RuleBasedTextModel:
    """Offline preview model. Use an LLM for useful production labels."""

    async def summarize(self, conversation: Conversation) -> SummaryPayload:
        user_text = " ".join(message.content for message in conversation.messages if message.role == "user").strip()
        excerpt = (user_text[:170].rsplit(" ", 1)[0] or user_text[:170]) if len(user_text) > 170 else user_text
        return SummaryPayload(summary=excerpt or "Empty conversation", request=excerpt, task=excerpt, languages=["english"])

    async def label(self, examples: list[str], others: list[str]) -> LabelPayload:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not examples:
            return LabelPayload(name="Uncategorized", description="No examples available")
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
            matrix = vectorizer.fit_transform(examples)
            ranks = matrix.mean(axis=0).A1.argsort()[::-1]
            words = vectorizer.get_feature_names_out()
            name = " ".join(words[i] for i in ranks[:3]).title()
        except ValueError:
            name = "Other requests"
        return LabelPayload(name=name or "Other requests", description=f"{len(examples)} related conversation summaries")
