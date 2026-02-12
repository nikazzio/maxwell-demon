"""Prepare PAISA corpus and gold-standard reference dictionaries for v2.0."""

from __future__ import annotations

import argparse
import gzip
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from maxwell_demon.analyzer import preprocess_text
from maxwell_demon.config import DEFAULT_CONFIG, load_config
from maxwell_demon.metrics import save_ref_dict
from tqdm import tqdm

DEFAULT_USER_AGENT = "maxwell-demon/1.2 (+https://github.com/nikazzio/maxwell-demon)"
DOWNLOAD_CHUNK_SIZE = 1024 * 64


def _log(message: str) -> None:
    print(f"[prepare_resources] {message}", flush=True)


def _tokenization_summary(tokenization_cfg: Mapping[str, object]) -> str:
    method = tokenization_cfg.get("method", "unknown")
    encoding_name = tokenization_cfg.get("encoding_name", "unknown")
    include_punctuation = tokenization_cfg.get("include_punctuation", "unknown")
    return (
        f"method={method}, encoding_name={encoding_name}, "
        f"include_punctuation={include_punctuation}"
    )


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
    source_group = parser.add_mutually_exclusive_group(required=False)
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
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--only-human",
        action="store_true",
        help="Build only the human reference dictionary",
    )
    mode_group.add_argument(
        "--only-synthetic",
        action="store_true",
        help="Build only the synthetic reference dictionary",
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
    parser.add_argument(
        "--user-agent",
        default=None,
        help="Optional User-Agent for HTTP corpus downloads",
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
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    token_total = 0
    for file_path in files:
        _log(f"Tokenizing file: {file_path}")
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_tokens = preprocess_text(text, tokenization=tokenization_cfg)
        counts.update(file_tokens)
        token_total += len(file_tokens)
    return counts, token_total


def _download_corpus(url: str, output_path: Path, *, user_agent: str = DEFAULT_USER_AGENT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": user_agent})
    _log(f"Downloading corpus from {url} -> {output_path}")
    try:
        with urlopen(request) as source, output_path.open("wb") as target:
            raw_total = getattr(getattr(source, "headers", None), "get", lambda _k: None)(
                "Content-Length"
            )
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {output_path.name}",
                leave=False,
            ) as progress:
                while True:
                    chunk = source.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    target.write(chunk)
                    progress.update(len(chunk))
        _log(f"Download completed: {output_path}")
    except HTTPError as exc:
        raise SystemExit(
            f"Failed to download corpus from '{url}' (HTTP {exc.code}). "
            "Use --paisa-url/--synthetic-url to override, or --skip-download with a local file."
        ) from exc
    except URLError as exc:
        raise SystemExit(
            f"Failed to download corpus from '{url}' ({exc}). "
            "Check connectivity or use --skip-download with a local file."
        ) from exc
    if _is_gzip_file(output_path):
        _log(f"Detected gzip payload, decompressing: {output_path}")
        decompressed_tmp = output_path.with_suffix(f"{output_path.suffix}.tmp")
        with gzip.open(output_path, "rb") as source, decompressed_tmp.open("wb") as target:
            shutil.copyfileobj(source, target)
        decompressed_tmp.replace(output_path)
        _log(f"Decompression completed: {output_path}")
    return output_path


def _download_paisa(
    url: str,
    output_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Path:
    return _download_corpus(url, output_path, user_agent=user_agent)


def _is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _read_text_maybe_gzip(path: Path) -> str:
    if _is_gzip_file(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_lines_maybe_gzip(path: Path):
    if _is_gzip_file(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                yield line
        return
    with path.open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            yield line


def _count_tokens_in_corpus_file(
    path: Path,
    *,
    tokenization_cfg: Mapping[str, object],
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    token_total = 0
    _log(f"Streaming tokenization from corpus: {path}")
    for line in tqdm(_iter_lines_maybe_gzip(path), desc=f"Tokenizing {path.name}", unit="lines"):
        line_tokens = preprocess_text(line, tokenization=tokenization_cfg)
        counts.update(line_tokens)
        token_total += len(line_tokens)
    return counts, token_total


def _build_ref_dict_from_counter(counts: Counter[str], *, smoothing_k: float) -> dict[str, float]:
    if smoothing_k < 0:
        raise ValueError("smoothing_k must be >= 0")
    total = sum(counts.values())
    if total == 0:
        return {}
    if smoothing_k == 0:
        return {token: count / total for token, count in counts.items()}
    vocab_size = len(counts)
    denom = total + smoothing_k * vocab_size
    return {token: (count + smoothing_k) / denom for token, count in counts.items()}


def _load_synthetic_tokens(
    args: argparse.Namespace,
    reference_cfg: dict[str, object],
    *,
    tokenization_cfg: Mapping[str, object],
    user_agent: str,
) -> tuple[Counter[str], int]:
    _log("Preparing synthetic corpus tokens")
    if args.synthetic_input:
        synthetic_files = _collect_text_files(Path(args.synthetic_input))
        if not synthetic_files:
            raise SystemExit("No synthetic .txt files found")
        _log(f"Reading synthetic text files: {len(synthetic_files)}")
        return _load_tokens_from_text_files(synthetic_files, tokenization_cfg=tokenization_cfg)

    synthetic_url = args.synthetic_url or reference_cfg["synthetic_url"]
    synthetic_corpus_out = args.synthetic_corpus_out or reference_cfg["synthetic_corpus_path"]
    synthetic_corpus_path = Path(synthetic_corpus_out)
    if args.skip_download:
        if not synthetic_corpus_path.exists():
            raise SystemExit(f"Missing synthetic corpus file: {synthetic_corpus_path}")
    else:
        synthetic_corpus_path = _download_corpus(
            str(synthetic_url),
            synthetic_corpus_path,
            user_agent=user_agent,
        )
    return _count_tokens_in_corpus_file(synthetic_corpus_path, tokenization_cfg=tokenization_cfg)


def _resolve_build_modes(args: argparse.Namespace) -> tuple[bool, bool]:
    only_human = bool(getattr(args, "only_human", False))
    only_synthetic = bool(getattr(args, "only_synthetic", False))
    build_human = not only_synthetic
    build_synthetic = not only_human
    return build_human, build_synthetic


def main() -> None:
    args = _parse_args()
    _log("Starting resource preparation")

    cfg = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    reference_cfg = cfg["reference"]
    tokenization_cfg = cfg["tokenization"]
    if not isinstance(tokenization_cfg, Mapping):
        raise SystemExit("Invalid config section: tokenization")
    _log(f"Tokenization config: {_tokenization_summary(tokenization_cfg)}")
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
    user_agent = args.user_agent or DEFAULT_USER_AGENT
    build_human, build_synthetic = _resolve_build_modes(args)
    if build_synthetic and not args.synthetic_input and not args.synthetic_url:
        raise SystemExit(
            "Synthetic source is required unless --only-human is set "
            "(use --synthetic-input or --synthetic-url)"
        )

    messages: list[str] = ["Saved resources:"]

    if build_human:
        _log("Building human reference dictionary")
        paisa_corpus_path = Path(paisa_corpus_out)
        if args.skip_download:
            if not paisa_corpus_path.exists():
                raise SystemExit(f"Missing PAISA corpus file: {paisa_corpus_path}")
            _log(f"Using local PAISA corpus: {paisa_corpus_path}")
        else:
            paisa_corpus_path = _download_corpus(
                str(paisa_url),
                paisa_corpus_path,
                user_agent=user_agent,
            )
        _log(f"Tokenizing PAISA corpus: {paisa_corpus_path}")
        paisa_counts, paisa_token_total = _count_tokens_in_corpus_file(
            paisa_corpus_path,
            tokenization_cfg=tokenization_cfg,
        )
        _log(f"PAISA tokenization completed: {paisa_token_total} tokens")
        _log("Computing human reference probabilities")
        human_dict = _build_ref_dict_from_counter(paisa_counts, smoothing_k=smoothing_k)
        _log(f"Saving human reference dictionary: {human_dict_out}")
        save_ref_dict(human_dict, str(human_dict_out))
        messages.append(
            f"PAISA tokens={paisa_token_total} human_dict={human_dict_out} ({len(human_dict)} tokens)"
        )
        _log("Human reference dictionary saved")

    if build_synthetic:
        _log("Building synthetic reference dictionary")
        synthetic_counts, synthetic_token_total = _load_synthetic_tokens(
            args,
            reference_cfg,
            tokenization_cfg=tokenization_cfg,
            user_agent=user_agent,
        )
        _log(f"Synthetic tokenization completed: {synthetic_token_total} tokens")
        _log("Computing synthetic reference probabilities")
        synthetic_dict = _build_ref_dict_from_counter(synthetic_counts, smoothing_k=smoothing_k)
        _log(f"Saving synthetic reference dictionary: {synthetic_dict_out}")
        save_ref_dict(synthetic_dict, str(synthetic_dict_out))
        messages.append(
            f"synthetic_dict={synthetic_dict_out} ({len(synthetic_dict)} tokens)"
        )
        _log("Synthetic reference dictionary saved")

    print(*messages)


if __name__ == "__main__":
    main()
