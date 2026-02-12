"""Automatic Markdown reporting for tournament outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"

    columns = [str(col) for col in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join([header, separator, *rows])


def _descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats_cols = ["count", "mean", "median", "std", "min", "max"]
    if "label" in df.columns:
        stats = (
            df.groupby("label", dropna=False)["delta_h"]
            .agg(stats_cols)
            .reset_index()
            .sort_values("label")
        )
    else:
        aggregated = df["delta_h"].agg(stats_cols)
        stats = pd.DataFrame([aggregated.to_dict()])
        stats.insert(0, "label", "all")

    for col in ["mean", "median", "std", "min", "max"]:
        stats[col] = stats[col].astype(float).map(_format_float)
    stats["count"] = stats["count"].astype(int)
    return stats


def _classification_metrics(df: pd.DataFrame) -> dict[str, float | int] | None:
    if "label" not in df.columns:
        return None

    labels = df["label"].astype(str).str.strip().str.lower()
    valid_mask = labels.isin(["human", "ai"]) & df["delta_h"].notna()
    if not bool(valid_mask.any()):
        return None

    y_true_human = labels[valid_mask].to_numpy() == "human"
    y_pred_human = df.loc[valid_mask, "delta_h"].to_numpy(dtype=float) > 0.0

    tp = int(np.sum(y_pred_human & y_true_human))
    tn = int(np.sum((~y_pred_human) & (~y_true_human)))
    fp = int(np.sum(y_pred_human & (~y_true_human)))
    fn = int(np.sum((~y_pred_human) & y_true_human))

    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total else 0.0

    precision_den = tp + fp
    recall_den = tp + fn
    precision = float(tp / precision_den) if precision_den else 0.0
    recall = float(tp / recall_den) if recall_den else 0.0
    f1_den = precision + recall
    f1 = float((2 * precision * recall) / f1_den) if f1_den else 0.0

    return {
        "accuracy": accuracy,
        "precision_human": precision,
        "recall_human": recall,
        "f1_human": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": total,
    }


def generate_report(df: pd.DataFrame) -> str:
    """Generate a Markdown report from a tournament DataFrame."""
    if "delta_h" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'delta_h' column")

    stats = _descriptive_stats(df)
    metrics = _classification_metrics(df)

    lines: list[str] = [
        "# Maxwell-Demon Tournament Report",
        "",
        "## Descriptive Statistics",
        _markdown_table(stats),
        "",
    ]

    if metrics is None:
        lines.extend(
            [
                "## Classification Metrics (Rule: delta_h > 0 is Human)",
                "_Unavailable: missing or invalid `label` values (`human`/`ai`)._",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Classification Metrics (Rule: delta_h > 0 is Human)",
                f"- **Samples evaluated**: {metrics['n']}",
                f"- **Accuracy**: {metrics['accuracy']:.6f}",
                f"- **Precision (Human)**: {metrics['precision_human']:.6f}",
                f"- **Recall (Human)**: {metrics['recall_human']:.6f}",
                f"- **F1-Score (Human)**: {metrics['f1_human']:.6f}",
                "",
                "### Confusion Matrix (Human as Positive)",
                f"- **TP**: {metrics['tp']}",
                f"- **TN**: {metrics['tn']}",
                f"- **FP**: {metrics['fp']}",
                f"- **FN**: {metrics['fn']}",
                "",
            ]
        )

    return "\n".join(lines)


def save_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate and save a Markdown report."""
    report = generate_report(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Maxwell-Demon Markdown report")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output Markdown report path")
    return parser.parse_args()


def main() -> None:
    """Standalone CLI for report generation from CSV."""
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = pd.read_csv(input_path)
    save_report(frame, output_path)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
