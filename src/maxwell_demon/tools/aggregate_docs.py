"""Aggregate window-level Maxwell-Demon CSV results at document level."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_GROUP_BY = ("filename", "label", "mode", "reference")
KNOWN_NUMERIC_METRICS = (
    "mean_entropy",
    "entropy_variance",
    "compression_ratio",
    "unique_ratio",
    "delta_h",
    "burstiness_paisa",
)
DEFAULT_STATS = ("mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90")
DIRECT_STATS = {"mean", "median", "std", "min", "max"}
QUANTILE_STATS = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90}
SUPPORTED_STATS = DIRECT_STATS | set(QUANTILE_STATS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Maxwell-Demon CSV rows at document level")
    parser.add_argument("--input", required=True, help="Input CSV file or directory of CSV files")
    parser.add_argument("--output", required=True, help="Output aggregated CSV path")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated numeric metrics to aggregate (default: known metrics present)",
    )
    parser.add_argument(
        "--stats",
        default=",".join(DEFAULT_STATS),
        help="Comma-separated stats: mean,median,std,min,max,p10,p25,p75,p90",
    )
    parser.add_argument(
        "--group-by",
        default=",".join(DEFAULT_GROUP_BY),
        help="Comma-separated grouping columns (default: filename,label,mode,reference)",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Optional comma-separated output sorting columns (default: filename if present)",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _collect_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".csv" else []
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.csv") if path.is_file())
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _load_input_frame(input_path: Path) -> pd.DataFrame:
    files = _collect_csvs(input_path)
    if not files:
        raise ValueError("No CSV files found")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _warn_ignored(kind: str, items: list[str]) -> None:
    if items:
        print(f"Warning: ignoring missing {kind}: {', '.join(items)}", file=sys.stderr)


def _resolve_group_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    missing = [column for column in requested if column not in df.columns]
    _warn_ignored("group columns", missing)
    resolved = [column for column in requested if column in df.columns]
    if resolved:
        return resolved
    if "filename" in df.columns:
        return ["filename"]
    raise ValueError("No valid group-by columns found and 'filename' is missing")


def _resolve_metrics(df: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested:
        missing = [metric for metric in requested if metric not in df.columns]
        _warn_ignored("metrics", missing)
        candidates = [metric for metric in requested if metric in df.columns]
    else:
        candidates = [metric for metric in KNOWN_NUMERIC_METRICS if metric in df.columns]

    numeric = [metric for metric in candidates if is_numeric_dtype(df[metric])]
    non_numeric = [metric for metric in candidates if metric not in numeric]
    _warn_ignored("non-numeric metrics", non_numeric)
    if not numeric:
        raise ValueError("No valid numeric metrics available for aggregation")
    return numeric


def _resolve_stats(requested: list[str]) -> list[str]:
    stats = requested or list(DEFAULT_STATS)
    unknown = [stat for stat in stats if stat not in SUPPORTED_STATS]
    if unknown:
        raise ValueError(f"Unsupported stats: {', '.join(unknown)}")
    return stats


def aggregate_document_level(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[str],
    stats: list[str],
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    output = grouped.size().rename("n_windows").reset_index()

    for metric in metrics:
        for stat in stats:
            column_name = f"{metric}__{stat}"
            if stat in DIRECT_STATS:
                values = grouped[metric].agg(stat).reset_index(name=column_name)
            else:
                values = grouped[metric].quantile(QUANTILE_STATS[stat]).reset_index(name=column_name)
            output = output.merge(values, on=group_cols, how="left")

    return output


def _resolve_sort_columns(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in frame.columns]
        _warn_ignored("sort columns", missing)
        return [column for column in requested if column in frame.columns]
    if "filename" in frame.columns:
        return ["filename"]
    return []


def run_aggregation(
    *,
    input_path: str | Path,
    output_path: str | Path,
    metrics_raw: str | None,
    stats_raw: str | None,
    group_by_raw: str | None,
    sort_by_raw: str | None,
) -> tuple[pd.DataFrame, Path]:
    frame = _load_input_frame(Path(input_path))
    group_cols = _resolve_group_columns(frame, _parse_csv_list(group_by_raw))
    metrics = _resolve_metrics(frame, _parse_csv_list(metrics_raw))
    stats = _resolve_stats(_parse_csv_list(stats_raw))

    aggregated = aggregate_document_level(
        frame,
        group_cols=group_cols,
        metrics=metrics,
        stats=stats,
    )
    sort_columns = _resolve_sort_columns(aggregated, _parse_csv_list(sort_by_raw))
    if sort_columns:
        aggregated = aggregated.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output, index=False)
    return aggregated, output


def main() -> None:
    args = _parse_args()
    aggregated, output = run_aggregation(
        input_path=args.input,
        output_path=args.output,
        metrics_raw=args.metrics,
        stats_raw=args.stats,
        group_by_raw=args.group_by,
        sort_by_raw=args.sort_by,
    )
    print(f"Saved {len(aggregated)} document-level rows to {output}")


if __name__ == "__main__":
    main()
