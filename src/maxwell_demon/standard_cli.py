"""Standard one-shot workflows for Maxwell-Demon."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

from .analyzer import SUPPORTED_COMPRESSION_ALGOS
from .cli import run_single_analysis
from .config import DEFAULT_CONFIG, load_config
from .output_paths import infer_dataset_name
from .tools.aggregate_docs import run_aggregation
from .tools.report_stats import save_report
from .tournament import run_tournament

DEFAULT_STANDARD: dict[str, object] = {
    "compressions": ["lzma", "gzip"],
    "human_only": {
        "aggregate_metrics": [
            "mean_entropy",
            "entropy_variance",
            "compression_ratio",
            "unique_ratio",
        ],
        "aggregate_stats": ["mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"],
        "group_by": ["filename", "label", "mode", "reference", "compression"],
    },
    "plots": {"enabled": True, "density_threshold": 2000},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard Maxwell-Demon workflows")
    parser.add_argument("--workflow", required=True, choices=["human-only", "tournament"])
    parser.add_argument("--human-input", required=True, help="Human input .txt file or folder")
    parser.add_argument("--ai-input", required=True, help="AI input .txt file or folder")
    parser.add_argument("--config", default=None, help="Path to TOML config file")
    parser.add_argument("--dataset", default=None, help="Optional dataset name override")
    parser.add_argument(
        "--compressions",
        default=None,
        help="Comma-separated compression list (default from config standard.compressions)",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Base output root. Final path is <output-root>/<dataset>/<workflow>/...",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip document-level aggregation (human-only workflow)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs only")
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_runtime_config(config_path: str | None) -> tuple[dict[str, object], str]:
    if config_path:
        return load_config(config_path), config_path
    local = Path("config.local.toml")
    if local.exists():
        return load_config(local), str(local)
    return DEFAULT_CONFIG, "DEFAULT_CONFIG"


def _extract_standard_config(cfg: dict[str, object]) -> dict[str, object]:
    raw = cfg.get("standard", {})
    if not isinstance(raw, dict):
        raw = {}
    human_only = raw.get("human_only", {})
    if not isinstance(human_only, dict):
        human_only = {}
    plots = raw.get("plots", {})
    if not isinstance(plots, dict):
        plots = {}
    return {
        "compressions": raw.get("compressions", DEFAULT_STANDARD["compressions"]),
        "human_only": {
            "aggregate_metrics": human_only.get(
                "aggregate_metrics", DEFAULT_STANDARD["human_only"]["aggregate_metrics"]
            ),
            "aggregate_stats": human_only.get(
                "aggregate_stats", DEFAULT_STANDARD["human_only"]["aggregate_stats"]
            ),
            "group_by": human_only.get("group_by", DEFAULT_STANDARD["human_only"]["group_by"]),
        },
        "plots": {
            "enabled": plots.get("enabled", DEFAULT_STANDARD["plots"]["enabled"]),
            "density_threshold": plots.get(
                "density_threshold", DEFAULT_STANDARD["plots"]["density_threshold"]
            ),
        },
    }


def _resolve_compressions(
    requested: str | None, standard_cfg: dict[str, object]
) -> list[str]:
    if requested:
        selected = _parse_csv_list(requested)
    else:
        configured = standard_cfg.get("compressions", DEFAULT_STANDARD["compressions"])
        if not isinstance(configured, list):
            raise ValueError("standard.compressions must be a list of compression names")
        selected = [str(item).strip() for item in configured if str(item).strip()]
    if not selected:
        raise ValueError("At least one compression must be provided")
    invalid = [item for item in selected if item not in SUPPORTED_COMPRESSION_ALGOS]
    if invalid:
        supported = ",".join(sorted(SUPPORTED_COMPRESSION_ALGOS))
        raise ValueError(f"Unsupported compression(s): {', '.join(invalid)} (supported: {supported})")
    return selected


def _resolve_dataset_name(dataset: str | None, human_input: str, ai_input: str) -> str:
    if dataset:
        cleaned = dataset.strip()
        if not cleaned:
            raise ValueError("--dataset cannot be empty")
        return cleaned
    return infer_dataset_name([human_input, ai_input])


def _ensure_reference_paths(cfg: dict[str, object], *, tournament_mode: bool) -> None:
    reference_cfg = cfg["reference"]
    paisa_path = Path(str(reference_cfg["paisa_path"]))
    if not paisa_path.exists():
        raise FileNotFoundError(f"Missing PAISA reference dictionary: {paisa_path}")
    if tournament_mode:
        synthetic_path = Path(str(reference_cfg["synthetic_path"]))
        if not synthetic_path.exists():
            raise FileNotFoundError(f"Missing synthetic reference dictionary: {synthetic_path}")


def _write_boxplot(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if metric not in df.columns or "label" not in df.columns or df.empty:
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="label", y=metric, order=["ai", "human"])
    sample_size = min(len(df), 4000)
    if sample_size > 0:
        sns.stripplot(
            data=df.sample(sample_size, random_state=42),
            x="label",
            y=metric,
            order=["ai", "human"],
            alpha=0.14,
            size=2,
            color="black",
        )
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()


def _write_phase_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output_path: Path,
    density_threshold: int,
    title_prefix: str,
) -> None:
    required = [x_col, y_col, "label"]
    if any(column not in df.columns for column in required) or df.empty:
        return
    hover_data = [col for col in ["filename", "window_id", "label", "compression"] if col in df.columns]
    if len(df) >= density_threshold:
        fig = px.density_contour(
            df,
            x=x_col,
            y=y_col,
            color="label",
            hover_data=hover_data,
            title=f"{title_prefix}: {x_col} vs {y_col}",
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="label",
            hover_data=hover_data,
            title=f"{title_prefix}: {x_col} vs {y_col}",
            opacity=0.55,
            render_mode="webgl",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)


def _write_separation_table(df: pd.DataFrame, output_path: Path) -> None:
    if "label" not in df.columns or df.empty:
        return
    labels = df["label"].astype(str).str.lower().str.strip()
    human = df[labels == "human"]
    ai = df[labels == "ai"]
    if human.empty or ai.empty:
        return

    rows: list[dict[str, object]] = []
    metric_sources = [col for col in df.columns if col.endswith("__median")]
    if "delta_h" in df.columns:
        metric_sources.append("delta_h")
    for source_col in metric_sources:
        metric_name = source_col.replace("__median", "")
        h_values = human[source_col].dropna().astype(float)
        a_values = ai[source_col].dropna().astype(float)
        if h_values.empty or a_values.empty:
            continue
        h_med = float(h_values.median())
        a_med = float(a_values.median())
        h_mean = float(h_values.mean())
        a_mean = float(a_values.mean())
        h_std = float(h_values.std(ddof=1)) if len(h_values) > 1 else 0.0
        a_std = float(a_values.std(ddof=1)) if len(a_values) > 1 else 0.0
        pooled_var = ((h_std**2) + (a_std**2)) / 2
        effect = (h_mean - a_mean) / (pooled_var**0.5) if pooled_var > 0 else 0.0
        h_q1, h_q3 = float(h_values.quantile(0.25)), float(h_values.quantile(0.75))
        a_q1, a_q3 = float(a_values.quantile(0.25)), float(a_values.quantile(0.75))
        overlap = max(0.0, min(h_q3, a_q3) - max(h_q1, a_q1))
        union = max(h_q3, a_q3) - min(h_q1, a_q1)
        overlap_ratio = overlap / union if union > 0 else 0.0
        rows.append(
            {
                "metric": metric_name,
                "human_median": h_med,
                "ai_median": a_med,
                "median_diff_human_minus_ai": h_med - a_med,
                "cohen_d_human_minus_ai": effect,
                "iqr_overlap_ratio": overlap_ratio,
            }
        )
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("metric").to_csv(output_path, index=False)


def _merge_labeled_csv(human_csv: Path, ai_csv: Path, output_csv: Path) -> pd.DataFrame:
    human = pd.read_csv(human_csv)
    ai = pd.read_csv(ai_csv)
    combined = pd.concat([human, ai], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_standard_workflow(
    *,
    workflow: str,
    human_input: str | Path,
    ai_input: str | Path,
    config_path: str | None = None,
    dataset: str | None = None,
    compressions_raw: str | None = None,
    output_root: str | Path = "results",
    skip_plots: bool = False,
    skip_aggregate: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    cfg, resolved_config_path = _load_runtime_config(config_path)
    standard_cfg = _extract_standard_config(cfg)
    compressions = _resolve_compressions(compressions_raw, standard_cfg)
    dataset_name = _resolve_dataset_name(dataset, str(human_input), str(ai_input))
    _ensure_reference_paths(cfg, tournament_mode=(workflow == "tournament"))

    base_dir = Path(output_root) / dataset_name / workflow
    data_dir = base_dir / "data"
    plot_dir = base_dir / "plot"
    manifest_path = base_dir / "run_manifest.json"
    artifacts: list[str] = []

    manifest: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "dataset": dataset_name,
        "config_source": resolved_config_path,
        "human_input": str(human_input),
        "ai_input": str(ai_input),
        "compressions": compressions,
        "output_base": str(base_dir),
        "dry_run": dry_run,
        "artifacts": artifacts,
    }

    if dry_run:
        for compression in compressions:
            if workflow == "human-only":
                artifacts.extend(
                    [
                        str(data_dir / f"labeled_{compression}.csv"),
                        str(data_dir / f"doc_level_{compression}.csv"),
                        str(data_dir / f"separation_{compression}.csv"),
                        str(plot_dir / f"box_mean_entropy_by_label_{compression}.png"),
                        str(plot_dir / f"box_compression_ratio_by_label_{compression}.png"),
                        str(
                            plot_dir
                            / f"phase_density_mean_entropy_vs_compression_ratio_by_label_{compression}.html"
                        ),
                    ]
                )
            else:
                artifacts.extend(
                    [
                        str(data_dir / f"delta_{compression}.csv"),
                        str(data_dir / f"delta_{compression}.md"),
                        str(data_dir / f"separation_{compression}.csv"),
                        str(
                            plot_dir
                            / f"phase_delta_h_vs_burstiness_paisa_{compression}.html"
                        ),
                        str(plot_dir / f"box_delta_h_by_label_{compression}.png"),
                    ]
                )
        return manifest

    window = int(cfg["analysis"]["window"])
    step = int(cfg["analysis"]["step"])
    log_base = float(cfg["analysis"]["log_base"])
    tokenization = cfg["tokenization"]
    if not isinstance(tokenization, dict):
        raise ValueError("Invalid config section: tokenization")

    for index, compression in enumerate(compressions, start=1):
        print(f"[{index}/{len(compressions)}] workflow={workflow} compression={compression}")
        if workflow == "human-only":
            tmp_human = data_dir / f".tmp_human_{compression}.csv"
            tmp_ai = data_dir / f".tmp_ai_{compression}.csv"
            run_single_analysis(
                input_path=human_input,
                mode="diff",
                window=window,
                step=step,
                output_path=tmp_human,
                output_dir=None,
                label="human",
                reference_name="paisa",
                ref_dict_path=None,
                log_base=log_base,
                compression=compression,
                cfg=cfg,
            )
            run_single_analysis(
                input_path=ai_input,
                mode="diff",
                window=window,
                step=step,
                output_path=tmp_ai,
                output_dir=None,
                label="ai",
                reference_name="paisa",
                ref_dict_path=None,
                log_base=log_base,
                compression=compression,
                cfg=cfg,
            )
            labeled_out = data_dir / f"labeled_{compression}.csv"
            labeled_frame = _merge_labeled_csv(tmp_human, tmp_ai, labeled_out)
            tmp_human.unlink(missing_ok=True)
            tmp_ai.unlink(missing_ok=True)
            artifacts.append(str(labeled_out))

            if not skip_aggregate:
                human_only_cfg = standard_cfg["human_only"]
                metrics_raw = ",".join(human_only_cfg["aggregate_metrics"])
                stats_raw = ",".join(human_only_cfg["aggregate_stats"])
                group_by_raw = ",".join(human_only_cfg["group_by"])
                doc_out = data_dir / f"doc_level_{compression}.csv"
                aggregated, _ = run_aggregation(
                    input_path=labeled_out,
                    output_path=doc_out,
                    metrics_raw=metrics_raw,
                    stats_raw=stats_raw,
                    group_by_raw=group_by_raw,
                    sort_by_raw="filename,label",
                )
                artifacts.append(str(doc_out))
                separation_out = data_dir / f"separation_{compression}.csv"
                _write_separation_table(aggregated, separation_out)
                if separation_out.exists():
                    artifacts.append(str(separation_out))
            else:
                aggregated = pd.DataFrame()

            plots_cfg = standard_cfg["plots"]
            plots_enabled = bool(plots_cfg["enabled"]) and not skip_plots
            if plots_enabled:
                if not aggregated.empty:
                    mean_col = (
                        "mean_entropy__median"
                        if "mean_entropy__median" in aggregated.columns
                        else "mean_entropy__mean"
                    )
                    comp_col = (
                        "compression_ratio__median"
                        if "compression_ratio__median" in aggregated.columns
                        else "compression_ratio__mean"
                    )
                    box_entropy = plot_dir / f"box_mean_entropy_by_label_{compression}.png"
                    _write_boxplot(
                        aggregated,
                        mean_col,
                        box_entropy,
                        f"Document-level entropy by label ({compression})",
                    )
                    if box_entropy.exists():
                        artifacts.append(str(box_entropy))
                    box_compression = plot_dir / f"box_compression_ratio_by_label_{compression}.png"
                    _write_boxplot(
                        aggregated,
                        comp_col,
                        box_compression,
                        f"Document-level compression ratio by label ({compression})",
                    )
                    if box_compression.exists():
                        artifacts.append(str(box_compression))

                phase_out = (
                    plot_dir
                    / f"phase_density_mean_entropy_vs_compression_ratio_by_label_{compression}.html"
                )
                _write_phase_plot(
                    labeled_frame,
                    x_col="mean_entropy",
                    y_col="compression_ratio",
                    output_path=phase_out,
                    density_threshold=int(plots_cfg["density_threshold"]),
                    title_prefix="Window-level phase",
                )
                if phase_out.exists():
                    artifacts.append(str(phase_out))
            continue

        tournament_out = data_dir / f"delta_{compression}.csv"
        tournament_frame = run_tournament(
            human_input=human_input,
            ai_input=ai_input,
            paisa_ref_path=cfg["reference"]["paisa_path"],
            synthetic_ref_path=cfg["reference"]["synthetic_path"],
            window_size=window,
            step=step,
            log_base=log_base,
            output_path=tournament_out,
            compression=compression,
            tokenization=tokenization,
        )
        artifacts.append(str(tournament_out))
        report_out = data_dir / f"delta_{compression}.md"
        save_report(tournament_frame, report_out)
        artifacts.append(str(report_out))
        separation_out = data_dir / f"separation_{compression}.csv"
        _write_separation_table(tournament_frame, separation_out)
        if separation_out.exists():
            artifacts.append(str(separation_out))

        plots_cfg = standard_cfg["plots"]
        plots_enabled = bool(plots_cfg["enabled"]) and not skip_plots
        if plots_enabled:
            phase_out = plot_dir / f"phase_delta_h_vs_burstiness_paisa_{compression}.html"
            _write_phase_plot(
                tournament_frame,
                x_col="delta_h",
                y_col="burstiness_paisa",
                output_path=phase_out,
                density_threshold=int(plots_cfg["density_threshold"]),
                title_prefix="Tournament phase",
            )
            if phase_out.exists():
                artifacts.append(str(phase_out))
            box_out = plot_dir / f"box_delta_h_by_label_{compression}.png"
            _write_boxplot(
                tournament_frame,
                "delta_h",
                box_out,
                f"Tournament delta_h by label ({compression})",
            )
            if box_out.exists():
                artifacts.append(str(box_out))

    _write_manifest(manifest_path, manifest)
    artifacts.append(str(manifest_path))
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = run_standard_workflow(
        workflow=args.workflow,
        human_input=args.human_input,
        ai_input=args.ai_input,
        config_path=args.config,
        dataset=args.dataset,
        compressions_raw=args.compressions,
        output_root=args.output_root,
        skip_plots=args.skip_plots,
        skip_aggregate=args.skip_aggregate,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    print(f"Completed workflow '{args.workflow}' for dataset '{manifest['dataset']}'")

