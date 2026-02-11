"""Static PNG plotting for Maxwell-Demon results."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Maxwell-Demon results")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default="plot.png", help="Output image path")
    parser.add_argument("--metric", default="mean_entropy", help="Metric to plot on Y axis")
    parser.add_argument(
        "--hue",
        default="label",
        help="Column for color grouping (e.g., label, mode, filename)",
    )
    return parser.parse_args()


def _collect_csvs(input_path: Path) -> list[Path]:
    """Collect input CSVs from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.csv") if p.is_file()])
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    """Render a line plot for a chosen metric and save a PNG."""
    args = _parse_args()
    input_path = Path(args.input)
    files = _collect_csvs(input_path)
    if not files:
        raise SystemExit("No CSV files found")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    if args.metric not in df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found in CSV columns")
    if args.hue not in df.columns:
        raise SystemExit(f"Hue column '{args.hue}' not found in CSV columns")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="window_id",
        y=args.metric,
        hue=args.hue,
        estimator=None,
        alpha=0.8,
    )
    plt.title(f"{args.metric} over windows")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
