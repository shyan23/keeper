from app.services.health import check_health


def test_check_health_ok():
    result = check_health()
    assert result["db"] == "ok"
    assert result["pgvector"] is True
    assert "version" in result
