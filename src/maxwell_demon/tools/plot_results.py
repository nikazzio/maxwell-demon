"""Static PNG plotting for Maxwell-Demon results."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.output_paths import (
    infer_dataset_name,
    line_plot_filename,
    resolve_output_template,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot Maxwell-Demon results")
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--input", required=True, help="Input CSV or folder of CSVs")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--metric", default="mean_entropy", help="Metric to plot on Y axis")
    parser.add_argument(
        "--hue",
        default="label",
        help="Column for color grouping (e.g., label, mode, filename)",
    )
    parser.add_argument(
        "--max-legend-items",
        type=int,
        default=20,
        help="Hide legend when hue has more unique values than this threshold",
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
    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    input_path = Path(args.input)
    if args.output is None:
        dataset = infer_dataset_name([args.input])
        plot_dir = resolve_output_template(cfg["output"]["plot_dir"], dataset)
        output = str(Path(plot_dir) / line_plot_filename(args.metric, "png"))
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
    if args.hue not in df.columns:
        raise SystemExit(f"Hue column '{args.hue}' not found in CSV columns")

    unique_hue_items = df[args.hue].dropna().nunique()
    show_legend = unique_hue_items <= args.max_legend_items
    if not show_legend:
        print(
            f"Hiding legend: '{args.hue}' has {unique_hue_items} unique values "
            f"(threshold={args.max_legend_items})"
        )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df,
        x="window_id",
        y=args.metric,
        hue=args.hue,
        estimator=None,
        alpha=0.8,
        legend=show_legend,
    )
    if not show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    plt.title(f"{args.metric} over windows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
