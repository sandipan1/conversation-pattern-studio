import pytest

from pattern_studio.storage import SQLStore


def test_sql_store_separates_runs(tmp_path):
    pytest.importorskip("sqlalchemy")
    store = SQLStore(f"sqlite:///{tmp_path / 'analysis.db'}")
    store.write("summaries", [{"chat_id": "first"}])
    new_run = store.create_run("Second run", "another dataset")
    assert store.read("summaries") is None
    store.write("summaries", [{"chat_id": "second"}])
    assert any(row["id"] == new_run and row["name"] == "Second run" for row in store.list_runs())
    store.use_run("default")
    assert store.read("summaries") == [{"chat_id": "first"}]
    with pytest.raises(ValueError, match="Unknown run"):
        store.use_run("missing")
