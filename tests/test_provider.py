from types import SimpleNamespace

import pytest

from pattern_studio.models import Conversation, Message
from pattern_studio.providers import FileCache, OpenAIEmbedder, OpenAITextModel


@pytest.mark.asyncio
async def test_model_retries_invalid_json_and_caches_valid_result(tmp_path):
    model = OpenAITextModel(api_key="unused-for-test", cache=FileCache(tmp_path))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        content = '{"unexpected": true}' if len(calls) == 1 else '{"summary": "Help with sign in"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    conversation = Conversation(chat_id="1", messages=[Message(role="user", content="Help me sign in")])
    first = await model.summarize(conversation)
    second = await model.summarize(conversation)
    assert first.summary == second.summary == "Help with sign in"
    assert len(calls) == 2
    assert all(call["model"] == "gpt-6-luna" for call in calls)
    assert all(call["reasoning_effort"] == "low" and "temperature" not in call for call in calls)


def test_openai_providers_require_key_even_with_custom_base_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAITextModel()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()
