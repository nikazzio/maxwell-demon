from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from maxwell_demon.tools import aggregate_docs


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filename": "a.txt",
                "label": "human",
                "mode": "diff",
                "reference": "paisa",
                "window_id": 0,
                "mean_entropy": 11.0,
                "compression_ratio": 0.90,
                "unique_ratio": 0.80,
            },
            {
                "filename": "a.txt",
                "label": "human",
                "mode": "diff",
                "reference": "paisa",
                "window_id": 1,
                "mean_entropy": 13.0,
                "compression_ratio": 0.94,
                "unique_ratio": 0.82,
            },
            {
                "filename": "b.txt",
                "label": "ai",
                "mode": "diff",
                "reference": "paisa",
                "window_id": 0,
                "mean_entropy": 10.0,
                "compression_ratio": 0.88,
                "unique_ratio": 0.75,
            },
            {
                "filename": "b.txt",
                "label": "ai",
                "mode": "diff",
                "reference": "paisa",
                "window_id": 1,
                "mean_entropy": 12.0,
                "compression_ratio": 0.92,
                "unique_ratio": 0.77,
            },
        ]
    )


def test_aggregate_defaults_from_single_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "single.csv"
    output_csv = tmp_path / "doc.csv"
    _sample_frame().to_csv(input_csv, index=False)

    aggregated, output = aggregate_docs.run_aggregation(
        input_path=input_csv,
        output_path=output_csv,
        metrics_raw=None,
        stats_raw=None,
        group_by_raw=None,
        sort_by_raw=None,
    )

    assert output == output_csv
    assert output_csv.exists()
    assert len(aggregated) == 2
    assert "n_windows" in aggregated.columns
    assert "mean_entropy__mean" in aggregated.columns
    assert "mean_entropy__median" in aggregated.columns
    assert "compression_ratio__p90" in aggregated.columns
    row_a = aggregated[aggregated["filename"] == "a.txt"].iloc[0]
    assert row_a["n_windows"] == 2
    assert row_a["mean_entropy__mean"] == pytest.approx(12.0)


def test_aggregate_multiple_csv_concat(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_csv = tmp_path / "doc.csv"
    input_dir.mkdir()
    frame = _sample_frame()
    frame.iloc[:2].to_csv(input_dir / "one.csv", index=False)
    frame.iloc[2:].to_csv(input_dir / "two.csv", index=False)

    aggregated, _ = aggregate_docs.run_aggregation(
        input_path=input_dir,
        output_path=output_csv,
        metrics_raw="mean_entropy",
        stats_raw="mean,median",
        group_by_raw="filename,label",
        sort_by_raw=None,
    )

    assert len(aggregated) == 2
    assert set(aggregated["filename"]) == {"a.txt", "b.txt"}
    assert "mean_entropy__mean" in aggregated.columns
    assert "mean_entropy__median" in aggregated.columns


def test_group_by_override_and_stats_subset(tmp_path: Path) -> None:
    input_csv = tmp_path / "single.csv"
    output_csv = tmp_path / "doc.csv"
    _sample_frame().to_csv(input_csv, index=False)

    aggregated, _ = aggregate_docs.run_aggregation(
        input_path=input_csv,
        output_path=output_csv,
        metrics_raw="mean_entropy,compression_ratio",
        stats_raw="mean,p90",
        group_by_raw="filename,label",
        sort_by_raw="label,filename",
    )

    assert "mode" not in aggregated.columns
    assert "reference" not in aggregated.columns
    assert "mean_entropy__mean" in aggregated.columns
    assert "mean_entropy__p90" in aggregated.columns
    assert "compression_ratio__p90" in aggregated.columns
    assert "mean_entropy__std" not in aggregated.columns


def test_missing_optional_group_columns_are_ignored(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"filename": "x.txt", "mean_entropy": 1.0},
            {"filename": "x.txt", "mean_entropy": 2.0},
        ]
    )
    input_csv = tmp_path / "single.csv"
    output_csv = tmp_path / "doc.csv"
    frame.to_csv(input_csv, index=False)

    aggregated, _ = aggregate_docs.run_aggregation(
        input_path=input_csv,
        output_path=output_csv,
        metrics_raw="mean_entropy",
        stats_raw="mean",
        group_by_raw="filename,label,mode,reference",
        sort_by_raw=None,
    )

    assert list(aggregated["filename"]) == ["x.txt"]
    assert aggregated.loc[0, "mean_entropy__mean"] == pytest.approx(1.5)


def test_error_on_invalid_stats_token(tmp_path: Path) -> None:
    input_csv = tmp_path / "single.csv"
    output_csv = tmp_path / "doc.csv"
    _sample_frame().to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Unsupported stats"):
        aggregate_docs.run_aggregation(
            input_path=input_csv,
            output_path=output_csv,
            metrics_raw="mean_entropy",
            stats_raw="mean,p13",
            group_by_raw="filename",
            sort_by_raw=None,
        )


def test_error_when_no_valid_metrics(tmp_path: Path) -> None:
    input_csv = tmp_path / "single.csv"
    output_csv = tmp_path / "doc.csv"
    frame = pd.DataFrame([{"filename": "a.txt", "label": "human", "mode": "diff", "reference": "paisa"}])
    frame.to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="No valid numeric metrics"):
        aggregate_docs.run_aggregation(
            input_path=input_csv,
            output_path=output_csv,
            metrics_raw="mean_entropy",
            stats_raw="mean",
            group_by_raw="filename,label",
            sort_by_raw=None,
        )
