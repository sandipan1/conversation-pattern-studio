from types import SimpleNamespace

import pytest

from pattern_studio.providers import CohereEmbedder


@pytest.mark.asyncio
async def test_cohere_embedding_batches_in_order():
    pytest.importorskip("cohere")
    embedder = CohereEmbedder(api_key="unused-for-test")
    batches = []

    class FakeClient:
        def embed(self, *, texts, model, input_type, embedding_types):
            batches.append(len(texts))
            assert model == "embed-v4.0"
            assert input_type == "clustering"
            assert embedding_types == ["float"]
            return SimpleNamespace(embeddings=SimpleNamespace(float=[[float(int(text))] for text in texts]))

    embedder.client = FakeClient()
    vectors = await embedder.embed([str(i) for i in range(100)])
    assert batches == [96, 4]
    assert vectors == [[float(i)] for i in range(100)]
