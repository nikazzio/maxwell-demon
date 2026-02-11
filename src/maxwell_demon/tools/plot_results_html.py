"""Interactive HTML plotting for Maxwell-Demon results."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.output_paths import (
    infer_dataset_name,
    line_plot_filename,
    resolve_output_template,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Maxwell-Demon results (HTML)")
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default=None, help="Output HTML path")
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
    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    input_path = Path(args.input)
    if args.output is None:
        dataset = infer_dataset_name([args.input])
        plot_dir = resolve_output_template(cfg["output"]["plot_dir"], dataset)
        output = str(Path(plot_dir) / line_plot_filename(args.metric, "html"))
    else:
        output = args.output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.write_html(output_path)
    print(f"Saved HTML plot to {output}")


if __name__ == "__main__":
    main()
