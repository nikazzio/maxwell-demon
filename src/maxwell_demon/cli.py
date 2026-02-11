"""CLI entry point for Maxwell-Demon."""

import argparse
from pathlib import Path

import pandas as pd

from .analyzer import analyze_tokens, preprocess_text
from .config import DEFAULT_CONFIG, load_config
from .metrics import (
    build_ref_dict,
    build_ref_dict_from_frequency_list,
    download_frequency_list,
    load_ref_dict,
    save_ref_dict,
)


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
    parser.add_argument("--input", required=False, help="Input .txt file or folder")
    parser.add_argument("--mode", choices=["raw", "diff"], default=None)
    parser.add_argument("--window", type=int, default=None, help="Window size in tokens")
    parser.add_argument("--step", type=int, default=None, help="Step size in tokens")
    parser.add_argument("--output", required=False, help="Output CSV path")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to write one CSV per input file",
    )
    parser.add_argument("--label", choices=["human", "ai"], default=None)
    parser.add_argument("--ref-dict", dest="ref_dict", help="Path to reference dictionary JSON")
    parser.add_argument(
        "--build-ref-dict",
        dest="build_ref_dict",
        help="Build reference dictionary from corpus text file",
    )
    parser.add_argument(
        "--build-ref-dict-from-freq",
        dest="build_ref_dict_from_freq",
        help="Build reference dictionary from a frequency list file",
    )
    parser.add_argument(
        "--freq-url",
        dest="freq_url",
        default=None,
        help="Download frequency list from URL before building dict",
    )
    parser.add_argument(
        "--freq-cache",
        dest="freq_cache",
        default="data/frequency_list.txt",
        help="Local path to save downloaded frequency list",
    )
    parser.add_argument(
        "--ref-dict-out",
        dest="ref_dict_out",
        default="ref_dict.json",
        help="Output JSON for built reference dictionary",
    )
    parser.add_argument(
        "--log-base",
        type=float,
        default=None,
        help="Log base for entropy/surprisal (e.g., 2 for bits)",
    )
    parser.add_argument(
        "--compression",
        choices=["zlib", "gzip", "bz2", "lzma"],
        default=None,
        help="Compression algorithm for ratio",
    )
    parser.add_argument(
        "--unknown-prob",
        type=float,
        default=None,
        help="Fallback probability for unseen tokens (diff mode)",
    )
    parser.add_argument(
        "--smoothing-k",
        type=float,
        default=None,
        help="Add-k smoothing for reference dictionary building",
    )
    parser.add_argument("--config", help="Path to TOML config file")
    return parser.parse_args()


def main() -> None:
    """Run CLI workflow: parse args, analyze input, write CSV."""
    args = _parse_args()

    cfg = DEFAULT_CONFIG
    if args.config:
        cfg = load_config(args.config)

    mode = args.mode if args.mode is not None else cfg["analysis"]["mode"]
    window = args.window if args.window is not None else cfg["analysis"]["window"]
    step = args.step if args.step is not None else cfg["analysis"]["step"]
    log_base = args.log_base if args.log_base is not None else cfg["analysis"]["log_base"]
    compression = (
        args.compression if args.compression is not None else cfg["compression"]["algorithm"]
    )
    unknown_prob = (
        args.unknown_prob if args.unknown_prob is not None else cfg["reference"]["unknown_prob"]
    )
    smoothing_k = (
        args.smoothing_k if args.smoothing_k is not None else cfg["reference"]["smoothing_k"]
    )
    ref_dict_path = args.ref_dict if args.ref_dict is not None else cfg["reference"]["path"]

    if args.build_ref_dict:
        corpus_path = Path(args.build_ref_dict)
        ref_dict = build_ref_dict(str(corpus_path), smoothing_k=smoothing_k)
        save_ref_dict(ref_dict, args.ref_dict_out)
        print(f"Reference dictionary saved to {args.ref_dict_out} ({len(ref_dict)} tokens)")
        return
    if args.build_ref_dict_from_freq:
        freq_path = Path(args.build_ref_dict_from_freq)
        if args.freq_url:
            freq_path = download_frequency_list(args.freq_url, args.freq_cache)
        ref_dict = build_ref_dict_from_frequency_list(str(freq_path))
        save_ref_dict(ref_dict, args.ref_dict_out)
        print(f"Reference dictionary saved to {args.ref_dict_out} ({len(ref_dict)} tokens)")
        return

    if not args.input:
        raise SystemExit("--input is required unless using --build-ref-dict")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("results.csv")
    output_dir = Path(args.output_dir) if args.output_dir else None

    if log_base <= 0 or log_base == 1.0:
        raise SystemExit("--log-base must be > 0 and != 1")

    files = _collect_input_files(input_path)
    if not files:
        raise SystemExit("No .txt files found in input")

    ref_dict: dict[str, float] | None = None
    if mode == "diff":
        if not ref_dict_path:
            raise SystemExit("--ref-dict is required for diff mode")
        ref_dict = load_ref_dict(ref_dict_path)

    rows: list[dict[str, object]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = preprocess_text(text)
        window_results = analyze_tokens(
            tokens=tokens,
            mode=mode,
            window_size=window,
            step=step,
            ref_dict=ref_dict,
            log_base=log_base,
            compression=compression,
            unknown_prob=unknown_prob,
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
                "label": args.label,
                "log_base": log_base,
                "compression": compression,
            }
            file_rows.append(record)
            rows.append(record)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{file_path.stem}.csv"
            pd.DataFrame(file_rows).to_csv(out_file, index=False)

    if not output_dir:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
    else:
        print(f"Saved {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
