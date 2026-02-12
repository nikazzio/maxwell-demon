"""Dedicated CLI entrypoint for tournament workflow."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from .analyzer import SUPPORTED_COMPRESSION_ALGOS
from .config import DEFAULT_CONFIG, load_config
from .output_paths import infer_dataset_name, resolve_output_template, tournament_output_filename
from .tools.report_stats import save_report
from .tournament import run_tournament


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Maxwell-Demon Tournament")
    parser.add_argument("--human-input", required=True, help="Human input .txt file or folder")
    parser.add_argument("--ai-input", required=True, help="AI input .txt file or folder")
    parser.add_argument("--output", default=None, help="Tournament CSV output path")
    parser.add_argument("--window", type=int, default=None, help="Window size in tokens")
    parser.add_argument("--step", type=int, default=None, help="Step size in tokens")
    parser.add_argument("--log-base", type=float, default=None, help="Log base for surprisal")
    parser.add_argument(
        "--compression",
        choices=list(SUPPORTED_COMPRESSION_ALGOS),
        default=None,
        help="Compression algorithm (protocol v2.0: lzma)",
    )
    parser.add_argument("--config", default=None, help="Path to TOML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)

    window = args.window if args.window is not None else cfg["analysis"]["window"]
    step = args.step if args.step is not None else cfg["analysis"]["step"]
    log_base = args.log_base if args.log_base is not None else cfg["analysis"]["log_base"]
    compression = (
        args.compression if args.compression is not None else cfg["compression"]["algorithm"]
    )
    tokenization_cfg = cfg["tokenization"]
    if not isinstance(tokenization_cfg, Mapping):
        raise SystemExit("Invalid config section: tokenization")

    if args.output is None:
        dataset = infer_dataset_name([args.human_input, args.ai_input])
        output_dir = resolve_output_template(cfg["output"]["data_dir"], dataset)
        output = str(Path(output_dir) / tournament_output_filename())
    else:
        output = args.output

    frame = run_tournament(
        human_input=Path(args.human_input),
        ai_input=Path(args.ai_input),
        paisa_ref_path=cfg["reference"]["paisa_path"],
        synthetic_ref_path=cfg["reference"]["synthetic_path"],
        window_size=window,
        step=step,
        log_base=log_base,
        compression=compression,
        tokenization=tokenization_cfg,
        output_path=output,
    )
    print(f"Saved {len(frame)} rows to {output}")

    report_path = Path(output).with_suffix(".md")
    try:
        save_report(frame, report_path)
        print(f"Saved report to {report_path}")
    except Exception as exc:  # pragma: no cover - best effort reporting
        print(f"Warning: failed to generate report at {report_path}: {exc}")


if __name__ == "__main__":
    main()
