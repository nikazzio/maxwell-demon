"""Interactive HTML plotting for Maxwell-Demon results."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Maxwell-Demon results (HTML)")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default="plot.html", help="Output HTML path")
    parser.add_argument("--metric", default="mean_entropy", help="Metric to plot on Y axis")
    parser.add_argument(
        "--color",
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
    """Render an interactive line plot and save as HTML."""
    args = _parse_args()
    input_path = Path(args.input)
    files = _collect_csvs(input_path)
    if not files:
        raise SystemExit("No CSV files found")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    if args.metric not in df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found in CSV columns")
    if args.color not in df.columns:
        raise SystemExit(f"Color column '{args.color}' not found in CSV columns")

    fig = px.line(
        df,
        x="window_id",
        y=args.metric,
        color=args.color,
        hover_data=["filename", "mode", "label"],
        title=f"{args.metric} over windows",
    )
    fig.write_html(args.output)
    print(f"Saved HTML plot to {args.output}")


if __name__ == "__main__":
    main()
