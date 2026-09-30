# Conversation Pattern Studio

A local workspace for finding recurring needs in chat data. It summarizes conversations, groups related requests, names the groups, builds higher-level themes, and gives you a dashboard to explore the results.

## Try it

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pattern-studio run examples/hierarchy-conversations.jsonl --offline --clusters 24 --max-depth 2
uv run pattern-studio serve --dir checkpoints/hierarchy-conversations
```

Open `http://127.0.0.1:8000` and choose **Hierarchy**. The example has 120 synthetic conversations across 24 request types, enough to show several levels. The offline run uses simple local labels and lexical embeddings, so its discovered groups may differ from the illustrative demo. For semantic analysis, put your OPENAI_API key in a local `.env` file and run without `--offline`:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY to your key.
uv run pattern-studio run <your-conversations.json> --output checkpoints
uv run pattern-studio serve --dir checkpoints/your-conversations
```

The project loads `.env` from the current directory without overriding existing environment variables. It uses `gpt-6-luna` for chat analysis and `text-embedding-3-small` for embeddings by default. Use `--model` to choose another chat model; `OPENAI_BASE_URL` can point to a compatible API. Analysis sends conversation content to the configured model provider. Review your data handling requirements before using a hosted provider.

## Input

A JSON array or JSONL file with one object per conversation:

```json
{"chat_id":"chat-1","messages":[{"role":"user","content":"How do I export my data?"},{"role":"assistant","content":"Open Settings → Data export."}],"metadata":{"channel":"support"}}
```

`created_at` is optional on conversations and messages. The `metadata` object is optional. Claude's conversation export is supported with `--input-format claude`. Python users can load Hugging Face datasets with `Conversation.from_huggingface(...)` after installing the `huggingface` extra.

## Explore the results

The dashboard has five views: an overview of leading needs, searchable pattern cards, a hierarchy tree, a map of related groups, and a conversation explorer. Each CLI input is saved under its own folder inside `checkpoints/`, named after the input file by default. Use `--run-name` to choose a different folder. Point the server at the run you want to explore. The import dialog also accepts a JSON object with `conversations`, `summaries`, `clusters`, `meta_clusters`, and `dimensionality` arrays.

For a real-data example using a small sample of WildChat, see [examples/README.md](examples/README.md).

To see a clear example immediately, choose **Import data → Explore 120-chat demo**. This illustrative demo uses predefined groups: 24 base patterns, three Level 2 themes, and one Level 3 theme. It makes no model calls. The matching raw conversations are in `examples/hierarchy-conversations.jsonl`; run the CLI on that file to discover groups with the selected models. The six-conversation `examples/conversations.json` remains a minimal input example.

If a hierarchy has only one Level 2 theme containing every conversation, it has not separated broader topics. For this 120-chat example, rerun with `--clusters 24 --max-depth 2`. Save that run to a new checkpoint directory if you want to compare it with an earlier result.

## Options

```bash
uv run pattern-studio run --help
uv run pattern-studio serve --help
```

- `--clusters N` sets the base group count; the default adapts to dataset size.
- The run command shows progress for summarizing, embedding, grouping, naming themes, and placing the map. `--concurrency N` controls how many model requests run in parallel (default: 10). Try a higher value if your API provider permits it; too many requests can cause rate limits. Changing this setting does not discard compatible checkpoints.
- `--clustering hdbscan` enables density grouping after `uv sync --extra hdbscan`.
- `--max-depth N` controls higher-level theme grouping.
- `--embeddings local` uses sentence-transformers after `uv sync --extra local-embeddings`. `--embeddings cohere` uses Cohere after `uv sync --extra cohere` and setting `COHERE_API_KEY`.
- `--checkpoint-format sqlite`, `sql`, `parquet`, or `huggingface` changes storage. SQL database URLs require `--database-url` and `uv sync --extra sql`; database drivers are installed separately. Optional formats need their matching extras. The dashboard reads JSONL checkpoints; other formats can be read in Python or exported to JSONL.
- `--resume` reuses checkpoints only when the input, model names, and pipeline settings match.

Print a compact hierarchy with `uv run pattern-studio inspect --dir checkpoints/<run-name>`.

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
