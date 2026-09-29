from typer.testing import CliRunner

from pattern_studio.cli import app


def test_run_requires_api_key_even_with_custom_base_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")

    result = CliRunner().invoke(app, ["run", "conversations.json"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output
