# Conversation Pattern Studio

A local workspace for finding recurring needs in chat data. It summarizes conversations, groups related requests, names the groups, builds higher-level themes, and gives you a dashboard to explore the results.

## Try it

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pattern-studio run examples/conversations.json --offline
uv run pattern-studio serve
```

Open `http://127.0.0.1:8000`. The offline run is a quick preview using simple local labels and lexical embeddings. For semantic analysis, set `OPENAI_API_KEY` and run without `--offline`:

```bash
export OPENAI_API_KEY=your-key
uv run pattern-studio run your-conversations.json --output checkpoints
uv run pattern-studio serve --dir checkpoints
```

The chat and embedding models use OpenAI by default. `OPENAI_BASE_URL` can point to a compatible API. Analysis sends conversation content to the configured model provider. Review your data handling requirements before using a hosted provider.

## Input

A JSON array or JSONL file with one object per conversation:

```json
{"chat_id":"chat-1","messages":[{"role":"user","content":"How do I export my data?"},{"role":"assistant","content":"Open Settings → Data export."}],"metadata":{"channel":"support"}}
```

`created_at` is optional on conversations and messages. The `metadata` object is optional. Claude's conversation export is supported with `--input-format claude`. Python users can load Hugging Face datasets with `Conversation.from_huggingface(...)` after installing the `huggingface` extra.

## Explore the results

The dashboard has four views: an overview of leading needs, searchable pattern cards, a map of related groups, and a conversation explorer. Import saved JSONL files in the browser or let the local server read the `checkpoints` directory. The import dialog also accepts a JSON object with `conversations`, `summaries`, `clusters`, `meta_clusters`, and `dimensionality` arrays. The built-in preview dataset lets you inspect the interface without running analysis.

## Options

```bash
uv run pattern-studio run --help
uv run pattern-studio serve --help
```

- `--clusters N` sets the base group count; the default adapts to dataset size.
- `--clustering hdbscan` enables density grouping after `uv sync --extra hdbscan`.
- `--max-depth N` controls higher-level theme grouping.
- `--embeddings local` uses sentence-transformers after `uv sync --extra local-embeddings`. `--embeddings cohere` uses Cohere after `uv sync --extra cohere` and setting `COHERE_API_KEY`.
- `--checkpoint-format sqlite`, `sql`, `parquet`, or `huggingface` changes storage. SQL database URLs require `--database-url` and `uv sync --extra sql`; database drivers are installed separately. Optional formats need their matching extras. The dashboard reads JSONL checkpoints; other formats can be read in Python or exported to JSONL.
- `--resume` reuses checkpoints only when the input, model names, and pipeline settings match.

Print a compact hierarchy with `uv run pattern-studio inspect --dir checkpoints`.

From Python:

```python
import asyncio
from pattern_studio import Conversation, analyze
from pattern_studio.providers import FileCache, OpenAIEmbedder, OpenAITextModel
from pattern_studio.storage import JSONLStore

conversations = Conversation.from_file("your-conversations.json")
result = asyncio.run(analyze(
    conversations,
    text_model=OpenAITextModel(cache=FileCache(".cache/pattern-studio")),
    embedder=OpenAIEmbedder(),
    checkpoints=JSONLStore("checkpoints"),
))
print(result.clusters)
```

For custom labels or extra summary fields, pass your own `TextModel` implementation to `analyze`, or configure `OpenAITextModel` with `summary_instruction`, `label_instruction`, and a `SummaryPayload` subclass as `summary_schema`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The project is intended for local use and does not include a package publishing workflow.
