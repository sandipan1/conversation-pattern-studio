import pytest

from pattern_studio.models import Conversation, LabelPayload, Message, SummaryPayload
from pattern_studio.pipeline import PipelineConfig, analyze, project_clusters
from pattern_studio.providers import HashingEmbedder
from pattern_studio.storage import JSONLStore


class FakeTextModel:
    model = "fake-v1"

    def __init__(self):
        self.summary_calls = 0
        self.label_calls = 0

    async def summarize(self, conversation):
        self.summary_calls += 1
        content = conversation.messages[0].content
        return SummaryPayload(summary=content, request=content, task=content)

    async def label(self, examples, others):
        self.label_calls += 1
        return LabelPayload(name=examples[0][:32], description=f"{len(examples)} related requests")


def conversations():
    prompts = ["Help me reset my password", "Help me log in", "Where are my invoices", "Show my billing history", "Invite my team", "Change member roles"]
    return [Conversation(chat_id=f"chat-{i}", messages=[Message(role="user", content=text)]) for i, text in enumerate(prompts)]


@pytest.mark.asyncio
async def test_pipeline_produces_hierarchy_and_resumes_without_model_calls(tmp_path):
    model = FakeTextModel()
    store = JSONLStore(tmp_path)
    config = PipelineConfig(cluster_count=3, children_per_parent=2, max_depth=2)
    result = await analyze(conversations(), text_model=model, embedder=HashingEmbedder(), config=config, checkpoints=store)
    assert len(result.summaries) == 6
    assert len(result.clusters) == 3
    assert result.meta_clusters
    assert all(cluster.x_coord is not None for cluster in result.projected_clusters)
    assert {chat_id for cluster in result.clusters for chat_id in cluster.chat_ids} == {f"chat-{i}" for i in range(6)}
    calls = (model.summary_calls, model.label_calls)
    second = await analyze(conversations(), text_model=model, embedder=HashingEmbedder(), config=config, checkpoints=store)
    assert (model.summary_calls, model.label_calls) == calls
    assert second.as_dict() == result.as_dict()


@pytest.mark.asyncio
async def test_changed_input_invalidates_checkpoints(tmp_path):
    model = FakeTextModel()
    store = JSONLStore(tmp_path)
    items = conversations()
    await analyze(items, text_model=model, embedder=HashingEmbedder(), checkpoints=store)
    items[0].messages[0].content = "A changed request"
    await analyze(items, text_model=model, embedder=HashingEmbedder(), checkpoints=store)
    assert model.summary_calls == 12


@pytest.mark.asyncio
async def test_duplicate_ids_rejected():
    items = conversations()
    items[1].chat_id = items[0].chat_id
    with pytest.raises(ValueError, match="unique"):
        await analyze(items, text_model=FakeTextModel(), embedder=HashingEmbedder())


def test_projection_preserves_empty_and_one_point():
    from pattern_studio.models import Cluster

    assert project_clusters([], []) == []
    single = Cluster(name="A", description="A", slug="a", chat_ids=["1"])
    point = project_clusters([single], [[1.0, 2.0]])[0]
    assert point.x_coord == 0.0 and point.y_coord == 0.0


@pytest.mark.asyncio
async def test_custom_summary_fields_are_preserved_in_metadata(tmp_path):
    from pydantic import Field

    class CustomSummary(SummaryPayload):
        sentiment: str = Field(...)

    class CustomText(FakeTextModel):
        async def summarize(self, conversation):
            self.summary_calls += 1
            return CustomSummary(summary="Help requested", sentiment="neutral")

    result = await analyze(conversations()[:1], text_model=CustomText(), embedder=HashingEmbedder(), checkpoints=JSONLStore(tmp_path))
    assert result.summaries[0].metadata["sentiment"] == "neutral"
