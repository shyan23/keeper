import datetime as dt

from app.services import report


def test_resolve_timeframe_explicit_years():
    req = {"years": [2021, 2022]}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2021, 1, 1)
    assert hi == dt.date(2022, 12, 31)


def test_resolve_timeframe_last_n_years():
    req = {"last_n_years": 3}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2023, 6, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_last_n_months():
    req = {"last_n_months": 4}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2026, 2, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_months_cross_year():
    req = {"last_n_months": 8}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 3, 10))
    assert lo == dt.date(2025, 7, 10)
    assert hi == dt.date(2026, 3, 10)


def test_resolve_timeframe_none_is_all_time():
    assert report.resolve_timeframe({}, dt.date(2026, 6, 19)) == (None, None)


def test_resolve_timeframe_leap_day_guard():
    lo, hi = report.resolve_timeframe({"last_n_years": 1}, dt.date(2024, 2, 29))
    assert lo == dt.date(2023, 2, 28)
