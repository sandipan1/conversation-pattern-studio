import json

from typer.testing import CliRunner

from pattern_studio.cli import app


def test_run_requires_api_key_even_with_custom_base_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")

    result = CliRunner().invoke(app, ["run", "conversations.json"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_run_saves_checkpoints_in_input_named_folder(tmp_path):
    input_file = tmp_path / "billing-questions.json"
    input_file.write_text(json.dumps([{
        "chat_id": "one",
        "messages": [{"role": "user", "content": "How do I update my billing details?"}],
    }]))
    output = tmp_path / "checkpoints"

    result = CliRunner().invoke(app, [
        "run", str(input_file), "--offline", "--clusters", "1", "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert (output / "billing-questions" / "conversations.jsonl").exists()
    assert (output / "billing-questions" / "manifest.jsonl").exists()
