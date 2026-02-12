from __future__ import annotations

import pandas as pd
from maxwell_demon.tools import report_stats


def test_classification_metrics_use_negative_delta_for_human() -> None:
    frame = pd.DataFrame(
        [
            {"label": "human", "delta_h": -0.3},
            {"label": "human", "delta_h": -0.1},
            {"label": "ai", "delta_h": 0.2},
            {"label": "ai", "delta_h": 0.4},
        ]
    )

    metrics = report_stats._classification_metrics(frame)

    assert metrics is not None
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["accuracy"] == 1.0


def test_generate_report_shows_updated_rule_heading() -> None:
    frame = pd.DataFrame(
        [
            {"label": "human", "delta_h": -0.1},
            {"label": "ai", "delta_h": 0.1},
        ]
    )

    report = report_stats.generate_report(frame)

    assert "Rule: delta_h < 0 is Human" in report
