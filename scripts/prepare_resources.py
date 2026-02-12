"""Prepare PAISA corpus and gold-standard reference dictionaries for v2.0."""

from __future__ import annotations

import argparse
import gzip
import shutil
from collections.abc import Mapping
from pathlib import Path
from urllib.request import urlretrieve

from maxwell_demon.analyzer import preprocess_text
from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.metrics import build_ref_dict_from_tokens, save_ref_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PAISA resources and generate human/synthetic reference dictionaries"
    )
    parser.add_argument("--config", default=None, help="Path to TOML config")
    parser.add_argument("--paisa-url", default=None, help="PAISA corpus URL (override config)")
    parser.add_argument(
        "--paisa-corpus-out",
        default=None,
        help="Downloaded PAISA corpus path (override config)",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--synthetic-input",
        default=None,
        help="Synthetic corpus source (.txt file or directory with .txt files)",
    )
    source_group.add_argument(
        "--synthetic-url",
        default=None,
        help="Synthetic corpus URL (override config)",
    )
    parser.add_argument(
        "--synthetic-corpus-out",
        default=None,
        help="Downloaded synthetic corpus path (override config)",
    )
    parser.add_argument(
        "--human-dict-out",
        default=None,
        help="Output JSON for human gold dictionary (override config)",
    )
    parser.add_argument(
        "--synthetic-dict-out",
        default=None,
        help="Output JSON for synthetic gold dictionary (override config)",
    )
    parser.add_argument(
        "--smoothing-k",
        type=float,
        default=None,
        help="Optional add-k smoothing override for reference dictionaries",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing local corpus files without downloading",
    )
    return parser.parse_args()


def _collect_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.txt") if p.is_file()])
    raise FileNotFoundError(f"Path not found: {path}")


def _load_tokens_from_text_files(
    files: list[Path],
    *,
    tokenization_cfg: Mapping[str, object],
) -> list[str]:
    tokens: list[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens.extend(preprocess_text(text, tokenization=tokenization_cfg))
    return tokens


def _download_corpus(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output_path)
    if _is_gzip_file(output_path):
        decompressed_tmp = output_path.with_suffix(f"{output_path.suffix}.tmp")
        with gzip.open(output_path, "rb") as source, decompressed_tmp.open("wb") as target:
            shutil.copyfileobj(source, target)
        decompressed_tmp.replace(output_path)
    return output_path


def _download_paisa(url: str, output_path: Path) -> Path:
    return _download_corpus(url, output_path)


def _is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _read_text_maybe_gzip(path: Path) -> str:
    if _is_gzip_file(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_synthetic_tokens(
    args: argparse.Namespace,
    reference_cfg: dict[str, object],
    *,
    tokenization_cfg: Mapping[str, object],
) -> list[str]:
    if args.synthetic_input:
        synthetic_files = _collect_text_files(Path(args.synthetic_input))
        if not synthetic_files:
            raise SystemExit("No synthetic .txt files found")
        return _load_tokens_from_text_files(synthetic_files, tokenization_cfg=tokenization_cfg)

    synthetic_url = args.synthetic_url or reference_cfg["synthetic_url"]
    synthetic_corpus_out = args.synthetic_corpus_out or reference_cfg["synthetic_corpus_path"]
    synthetic_corpus_path = Path(synthetic_corpus_out)
    if args.skip_download:
        if not synthetic_corpus_path.exists():
            raise SystemExit(f"Missing synthetic corpus file: {synthetic_corpus_path}")
    else:
        synthetic_corpus_path = _download_corpus(str(synthetic_url), synthetic_corpus_path)
    return preprocess_text(_read_text_maybe_gzip(synthetic_corpus_path), tokenization=tokenization_cfg)


def main() -> None:
    args = _parse_args()

    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    reference_cfg = cfg["reference"]
    tokenization_cfg = cfg["tokenization"]
    if not isinstance(tokenization_cfg, Mapping):
        raise SystemExit("Invalid config section: tokenization")
    paisa_url = args.paisa_url or reference_cfg["paisa_url"]
    paisa_corpus_out = args.paisa_corpus_out or reference_cfg["paisa_corpus_path"]
    human_dict_out = args.human_dict_out or reference_cfg["paisa_path"]
    synthetic_dict_out = args.synthetic_dict_out or reference_cfg["synthetic_path"]
    smoothing_k = (
        float(args.smoothing_k)
        if args.smoothing_k is not None
        else float(reference_cfg["smoothing_k"])
    )
    if smoothing_k < 0:
        raise SystemExit("--smoothing-k must be >= 0")

    paisa_corpus_path = Path(paisa_corpus_out)
    if args.skip_download:
        if not paisa_corpus_path.exists():
            raise SystemExit(f"Missing PAISA corpus file: {paisa_corpus_path}")
    else:
        paisa_corpus_path = _download_corpus(str(paisa_url), paisa_corpus_path)

    paisa_tokens = preprocess_text(
        _read_text_maybe_gzip(paisa_corpus_path),
        tokenization=tokenization_cfg,
    )
    synthetic_tokens = _load_synthetic_tokens(
        args,
        reference_cfg,
        tokenization_cfg=tokenization_cfg,
    )

    human_dict = build_ref_dict_from_tokens(paisa_tokens, smoothing_k=smoothing_k)
    synthetic_dict = build_ref_dict_from_tokens(synthetic_tokens, smoothing_k=smoothing_k)

    save_ref_dict(human_dict, str(human_dict_out))
    save_ref_dict(synthetic_dict, str(synthetic_dict_out))

    print(
        "Saved resources:",
        f"PAISA tokens={len(paisa_tokens)}",
        f"human_dict={human_dict_out} ({len(human_dict)} tokens)",
        f"synthetic_dict={synthetic_dict_out} ({len(synthetic_dict)} tokens)",
    )


if __name__ == "__main__":
    main()
