"""CLI entry point for Maxwell-Demon single analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .analyzer import SUPPORTED_COMPRESSION_ALGOS, analyze_tokens, preprocess_text
from .config import DEFAULT_CONFIG, load_config
from .metrics import load_ref_dict
from .output_paths import (
    infer_dataset_name,
    resolve_output_template,
    single_output_filename,
)

REFERENCE_NAMES = ("paisa", "synthetic")


def _collect_input_files(input_path: Path) -> list[Path]:
    """Collect input .txt files from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.txt") if p.is_file()])
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Maxwell-Demon CLI")
    parser.add_argument("--input", required=True, help="Input .txt file or folder")
    parser.add_argument("--mode", choices=["raw", "diff"], default=None)
    parser.add_argument("--window", type=int, default=None, help="Window size in tokens")
    parser.add_argument("--step", type=int, default=None, help="Step size in tokens")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to write one CSV per input file",
    )
    parser.add_argument("--label", choices=["human", "ai"], default=None)
    parser.add_argument(
        "--reference",
        choices=REFERENCE_NAMES,
        default=None,
        help="Reference dictionary to use in diff mode",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Run explicit human-reference-only analysis (diff mode with PAISA reference)",
    )
    parser.add_argument(
        "--ref-dict",
        dest="ref_dict",
        default=None,
        help="Optional explicit path to a reference dictionary JSON",
    )
    parser.add_argument(
        "--log-base",
        type=float,
        default=None,
        help="Log base for entropy/surprisal (e.g., 2 for bits)",
    )
    parser.add_argument(
        "--compression",
        choices=list(SUPPORTED_COMPRESSION_ALGOS),
        default=None,
        help="Compression algorithm for ratio (default from config)",
    )
    parser.add_argument("--config", help="Path to TOML config file")
    return parser.parse_args()


def _reference_path_from_config(cfg: dict[str, object], reference_name: str) -> str:
    reference_cfg = cfg["reference"]
    key = f"{reference_name}_path"
    value = reference_cfg[key]
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Missing config value: reference.{key}")
    return value


def _resolve_mode_reference(args: argparse.Namespace) -> tuple[str, str]:
    mode = args.mode or "raw"
    reference = args.reference or "paisa"

    if not args.human_only:
        return mode, reference

    if args.mode not in (None, "diff"):
        raise SystemExit("--human-only is incompatible with --mode raw")
    if args.reference not in (None, "paisa"):
        raise SystemExit("--human-only is incompatible with --reference synthetic")

    return "diff", "paisa"


def run_single_analysis(
    *,
    input_path: str | Path,
    mode: str,
    window: int,
    step: int,
    output_path: str | Path,
    output_dir: str | Path | None,
    label: str | None,
    reference_name: str,
    ref_dict_path: str | None,
    log_base: float,
    compression: str,
    cfg: dict[str, object],
) -> tuple[int, Path]:
    """Execute single raw/diff analysis and save CSV output(s)."""
    files = _collect_input_files(Path(input_path))
    if not files:
        raise SystemExit("No .txt files found in input")

    if log_base <= 0 or log_base == 1.0:
        raise SystemExit("--log-base must be > 0 and != 1")

    if mode == "diff":
        path = ref_dict_path or _reference_path_from_config(cfg, reference_name)
        ref_dict = load_ref_dict(path)
    else:
        ref_dict = None

    rows: list[dict[str, object]] = []
    out_dir = Path(output_dir) if output_dir else None

    tokenization_cfg = cfg["tokenization"]
    if not isinstance(tokenization_cfg, Mapping):
        raise SystemExit("Invalid config section: tokenization")

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = preprocess_text(text, tokenization=tokenization_cfg)
        window_results = analyze_tokens(
            tokens=tokens,
            mode=mode,
            window_size=window,
            step=step,
            ref_dict=ref_dict,
            log_base=log_base,
            compression=compression,
        )

        file_rows: list[dict[str, object]] = []
        for row in window_results:
            record = {
                "filename": file_path.name,
                "window_id": row["window_id"],
                "mean_entropy": row["mean_entropy"],
                "entropy_variance": row["entropy_variance"],
                "compression_ratio": row["compression_ratio"],
                "unique_ratio": row["unique_ratio"],
                "mode": mode,
                "label": label,
                "log_base": log_base,
                "compression": compression,
                "reference": reference_name if mode == "diff" else None,
            }
            file_rows.append(record)
            rows.append(record)

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{file_path.stem}.csv"
            pd.DataFrame(file_rows).to_csv(out_file, index=False)

    output = Path(output_path)
    if out_dir is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        output = out_dir

    return len(rows), output


def main() -> None:
    """Run CLI workflow for single analysis."""
    args = _parse_args()

    cfg = DEFAULT_CONFIG
    if args.config:
        cfg = load_config(args.config)
    mode, reference = _resolve_mode_reference(args)

    window = args.window if args.window is not None else cfg["analysis"]["window"]
    step = args.step if args.step is not None else cfg["analysis"]["step"]
    log_base = args.log_base if args.log_base is not None else cfg["analysis"]["log_base"]
    compression = (
        args.compression if args.compression is not None else cfg["compression"]["algorithm"]
    )
    if args.output is None:
        dataset = infer_dataset_name([args.input])
        output_dir = resolve_output_template(cfg["output"]["data_dir"], dataset)
        output = str(
            Path(output_dir)
            / single_output_filename(
                mode,
                reference,
                human_only=args.human_only,
            )
        )
    else:
        output = args.output

    total_rows, output = run_single_analysis(
        input_path=args.input,
        mode=mode,
        window=window,
        step=step,
        output_path=output,
        output_dir=args.output_dir,
        label=args.label,
        reference_name=reference,
        ref_dict_path=args.ref_dict,
        log_base=log_base,
        compression=compression,
        cfg=cfg,
    )
    print(f"Saved {total_rows} rows to {output}")


if __name__ == "__main__":
    main()
