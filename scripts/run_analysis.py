"""Unified entrypoint for Maxwell-Demon v2.0 analysis workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from maxwell_demon.analyzer import SUPPORTED_COMPRESSION_ALGOS
from maxwell_demon.cli import run_single_analysis
from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.output_paths import (
    infer_dataset_name,
    resolve_output_template,
    single_output_filename,
    tournament_output_filename,
)
from maxwell_demon.tournament import run_tournament


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Maxwell-Demon analysis workflows")
    parser.add_argument(
        "--workflow",
        choices=["single", "tournament"],
        required=True,
        help="Choose analysis workflow",
    )
    parser.add_argument("--config", default=None, help="Path to TOML config")

    parser.add_argument("--input", default=None, help="Single workflow input file/folder")
    parser.add_argument("--mode", choices=["raw", "diff"], default="raw")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--output-dir", default=None, help="Output directory for per-file CSVs")
    parser.add_argument("--label", choices=["human", "ai"], default=None)
    parser.add_argument("--reference", choices=["paisa", "synthetic"], default="paisa")
    parser.add_argument("--ref-dict", default=None, help="Optional explicit reference dict path")

    parser.add_argument(
        "--human-input",
        default=None,
        help="Tournament workflow human input (.txt or folder)",
    )
    parser.add_argument(
        "--ai-input",
        default=None,
        help="Tournament workflow AI input (.txt or folder)",
    )

    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--log-base", type=float, default=None)
    parser.add_argument("--compression", choices=list(SUPPORTED_COMPRESSION_ALGOS), default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = DEFAULT_CONFIG
    if args.config:
        cfg = load_config(args.config)

    window = args.window if args.window is not None else cfg["analysis"]["window"]
    step = args.step if args.step is not None else cfg["analysis"]["step"]
    log_base = args.log_base if args.log_base is not None else cfg["analysis"]["log_base"]
    compression = (
        args.compression if args.compression is not None else cfg["compression"]["algorithm"]
    )

    if args.workflow == "single":
        if not args.input:
            raise SystemExit("--input is required for --workflow single")
        if args.output is None:
            dataset = infer_dataset_name([args.input])
            output_dir = resolve_output_template(cfg["output"]["data_dir"], dataset)
            output = str(Path(output_dir) / single_output_filename(args.mode, args.reference))
        else:
            output = args.output
        total_rows, output = run_single_analysis(
            input_path=args.input,
            mode=args.mode,
            window=window,
            step=step,
            output_path=output,
            output_dir=args.output_dir,
            label=args.label,
            reference_name=args.reference,
            ref_dict_path=args.ref_dict,
            log_base=log_base,
            compression=compression,
            cfg=cfg,
        )
        print(f"Saved {total_rows} rows to {output}")
        return

    if not args.human_input or not args.ai_input:
        raise SystemExit("--human-input and --ai-input are required for --workflow tournament")
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
        tokenization=cfg["tokenization"],
        output_path=output,
    )
    print(f"Saved {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
