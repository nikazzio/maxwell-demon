"""Universal phase plotting for Maxwell-Demon v2.0 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.output_paths import (
    infer_dataset_name,
    phase_plot_filename,
    resolve_output_template,
)

TOURNAMENT_X = "delta_h"
TOURNAMENT_Y = "burstiness_paisa"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Universal phase diagram for Maxwell-Demon results"
    )
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--x", default=None, help="Column for X axis")
    parser.add_argument("--y", default=None, help="Column for Y axis")
    parser.add_argument(
        "--color",
        default="label",
        help="Column for color grouping (e.g., label, mode, filename)",
    )
    parser.add_argument("--facet", default=None, help="Optional column to facet by")
    parser.add_argument(
        "--density",
        action="store_true",
        help="Enable KDE-like density contours with Plotly",
    )
    parser.add_argument(
        "--density-threshold",
        type=int,
        default=3000,
        help="Auto-enable density when rows are above this threshold",
    )
    return parser.parse_args()


def _collect_csvs(input_path: Path) -> list[Path]:
    """Collect input CSVs from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.csv") if p.is_file()])
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _is_tournament_output(df: pd.DataFrame) -> bool:
    return TOURNAMENT_X in df.columns and TOURNAMENT_Y in df.columns


def _resolve_axes(df: pd.DataFrame, x: str | None, y: str | None) -> tuple[str, str]:
    if _is_tournament_output(df):
        resolved_x = x or TOURNAMENT_X
        resolved_y = y or TOURNAMENT_Y
        return resolved_x, resolved_y

    return x or "mean_entropy", y or "compression_ratio"


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col and col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in CSV columns")


def _build_hover_data(df: pd.DataFrame) -> list[str]:
    candidates = ["filename", "mode", "label", "window_id", "delta_h", "burstiness_paisa"]
    return [col for col in candidates if col in df.columns]


def main() -> None:
    """Render a universal phase diagram and save it as HTML."""
    args = _parse_args()
    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    input_path = Path(args.input)
    files = _collect_csvs(input_path)
    if not files:
        raise SystemExit("No CSV files found")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    x_col, y_col = _resolve_axes(df, args.x, args.y)

    if args.output is None:
        dataset = infer_dataset_name([args.input])
        plot_dir = resolve_output_template(cfg["output"]["plot_dir"], dataset)
        output = str(Path(plot_dir) / phase_plot_filename(x_col, y_col))
    else:
        output = args.output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_columns(df, [x_col, y_col, args.color, args.facet])

    hover_data = _build_hover_data(df)
    use_density = args.density or len(df) >= args.density_threshold

    if use_density:
        fig = px.density_contour(
            df,
            x=x_col,
            y=y_col,
            color=args.color,
            facet_col=args.facet,
            hover_data=hover_data,
            title=f"Phase density: {x_col} vs {y_col}",
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=args.color,
            facet_col=args.facet,
            hover_data=hover_data,
            title=f"Phase diagram: {x_col} vs {y_col}",
            opacity=0.55,
            render_mode="webgl",
        )

    fig.write_html(output_path)
    print(f"Saved phase diagram to {output}")


if __name__ == "__main__":
    main()
