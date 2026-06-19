"""Render a single numeric trend metric to a PNG (bytes) for the PDF. Feeds off
trends.metric_series output. Headless Agg backend — locked before pyplot import
so it never tries to open a display."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def render_metric_chart(series: dict) -> bytes:
    points = series.get("points") or []
    dates = [p["date"] for p in points]
    values = [p["value"] for p in points]
    unit = series.get("unit") or ""
    label = series.get("label") or series.get("key") or "Metric"

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(dates, values, marker="o", linewidth=1.6)
    lo, hi = series.get("ref_low"), series.get("ref_high")
    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color="tab:green", alpha=0.12, label="reference range")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{label} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{label} ({unit})" if unit else label)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
