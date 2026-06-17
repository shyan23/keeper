from app.api import runtime


def test_node_labels_cover_key_nodes():
    for node in ["router", "extract_text", "generate_answer"]:
        assert node in runtime.NODE_LABELS


def test_cfg_has_thread_and_progress():
    calls = []
    cfg = runtime.cfg("thread-abc", deps={"x": 1}, progress=calls.append)
    assert cfg["configurable"]["thread_id"] == "thread-abc"
    assert cfg["configurable"]["deps"] == {"x": 1}
    cfg["configurable"]["progress"]("hi")
    assert calls == ["hi"]
