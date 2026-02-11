"""Interactive phase diagram plotting for Maxwell-Demon results."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Phase diagram for Maxwell-Demon results")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default="phase.html", help="Output HTML path")
    parser.add_argument(
        "--x",
        default="mean_entropy",
        help="Column for X axis (default: mean_entropy)",
    )
    parser.add_argument(
        "--y",
        default="compression_ratio",
        help="Column for Y axis (default: compression_ratio)",
    )
    parser.add_argument(
        "--color",
        default="label",
        help="Column for color grouping (e.g., label, mode, filename)",
    )
    parser.add_argument(
        "--facet",
        default=None,
        help="Optional column to facet by (e.g., filename or mode)",
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
    """Render a phase diagram and save as HTML."""
    args = _parse_args()
    input_path = Path(args.input)
    files = _collect_csvs(input_path)
    if not files:
        raise SystemExit("No CSV files found")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    for col in [args.x, args.y, args.color]:
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in CSV columns")
    if args.facet and args.facet not in df.columns:
        raise SystemExit(f"Facet column '{args.facet}' not found in CSV columns")

    fig = px.scatter(
        df,
        x=args.x,
        y=args.y,
        color=args.color,
        facet_col=args.facet,
        hover_data=["filename", "mode", "label", "window_id"],
        title=f"Phase diagram: {args.x} vs {args.y}",
        opacity=0.7,
    )
    fig.write_html(args.output)
    print(f"Saved phase diagram to {args.output}")


if __name__ == "__main__":
    main()
