# Examples

These examples show how to load conversation data, analyze recurring requests, and explore the resulting themes.

## WildChat: real conversations

[WildChat](https://huggingface.co/datasets/allenai/WildChat) is a public dataset of real user-assistant conversations. The example streams a reproducible sample of 120 conversations by default, maps each conversation's turns into the app's input format, and sends the sample through the normal summary, embedding, clustering, and hierarchy pipeline.

### Run it

From the repository root:

```bash
uv sync --extra huggingface
cp .env.example .env
# Add your OPENAI_API_KEY to .env.
uv run python examples/wildchat/run.py
```

The run uses `gpt-6-luna` for summaries and labels and `text-embedding-3-small` for embeddings. Change the sample size or model with:

```bash
uv run python examples/wildchat/run.py --limit 200 --model gpt-6-luna
```

The script streams and shuffles with a fixed seed using a small buffer; it does not download the full dataset. This makes the sample repeatable while keeping the run manageable. The dataset currently has about 529,000 rows in its `train` split, so the sample is useful for a first look, not a comprehensive analysis.

### Explore the output

The script saves the normalized sample under `examples/wildchat/` and keeps its analysis checkpoints separately under `checkpoints/`:

```text
examples/wildchat/
  conversations-120.jsonl

checkpoints/
  wildchat-120/
    conversations.jsonl  # pipeline copy used for resume
    summaries.jsonl
    clusters.jsonl
    meta_clusters.jsonl
    dimensionality.jsonl
    manifest.jsonl
```

Open the dashboard for this run:

```bash
uv run pattern-studio serve --dir checkpoints/wildchat-120
```

For a different sample size, the conversation file and default checkpoint folder reflect its count, such as `examples/wildchat/conversations-200.jsonl` and `checkpoints/wildchat-200`. Use `--run-name` to pick a checkpoint folder explicitly.

### Data handling

WildChat is distributed under the Open Data Commons Attribution License (ODC-BY). Review the [dataset card and terms](https://huggingface.co/datasets/allenai/WildChat) before use. Conversation text from the sample is sent to the configured model provider for analysis; review its data-handling terms before running the example.

## Other datasets to explore

The WildChat script above is the runnable example. These datasets offer different questions to investigate and need their own row-to-conversation mapping before use:

| Dataset | What it contains | What an example would examine |
| --- | --- | --- |
| [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | About 3,360 human judgments comparing two model conversations, with a winner recorded for each comparison. | Which tasks lead to different model preferences; map `conversation_a` and `conversation_b` separately and retain the judgment as metadata. |
| [Chatbot Arena conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | About 33,000 paired conversations with user votes and language information. | How real user requests and preferences vary across topics and languages. Access requires accepting the dataset's terms; its prompts and model outputs have different licenses. |

Neither dataset is loaded by `examples/wildchat/run.py`. A new example can transform its rows into the [conversation input format](../README.md#input), save the normalized sample in `examples/`, and write analysis results to its own folder under `checkpoints/`.
