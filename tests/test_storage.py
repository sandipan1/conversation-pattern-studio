from pattern_studio.storage import JSONLStore, MultiStore, SQLiteStore


def test_checkpoint_stores_roundtrip_nested_records(tmp_path):
    records = [{"chat_id": "one", "messages": [{"content": "Hi", "role": "user"}], "metadata": {"n": 2}}]
    for store in (JSONLStore(tmp_path / "jsonl"), SQLiteStore(tmp_path / "checkpoint.db")):
        assert store.read("conversations") is None
        store.write("conversations", records)
        assert store.read("conversations") == records
        store.write("conversations", [])
        assert store.read("conversations") == []


def test_multi_store_writes_all_and_reads_first(tmp_path):
    first = JSONLStore(tmp_path / "first")
    second = SQLiteStore(tmp_path / "second.db")
    multi = MultiStore(first, second)
    multi.write("summaries", [{"chat_id": "a"}])
    assert first.read("summaries") == second.read("summaries") == multi.read("summaries")
