from app.services import charts


def _series():
    return {
        "key": "ldl", "label": "LDL", "unit": "mg/dL",
        "ref_low": 0.0, "ref_high": 100.0,
        "points": [
            {"date": "2021-01-01", "value": 90.0, "in_range": True},
            {"date": "2022-01-01", "value": 130.0, "in_range": False},
        ],
    }


def test_render_metric_chart_returns_png_bytes():
    png = charts.render_metric_chart(_series())
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000


def test_render_metric_chart_without_reference_band():
    s = _series(); s["ref_low"] = None; s["ref_high"] = None
    png = charts.render_metric_chart(s)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
