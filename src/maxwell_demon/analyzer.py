"""Sliding-window analyzer for Maxwell-Demon."""

from __future__ import annotations

import bz2
import gzip
import lzma
import re
import zlib
from collections.abc import Mapping
from typing import Any

from .metrics import (
    calculate_shannon_entropy,
    entropy_variance_from_tokens,
    surprisal_stats_from_ref,
)

TOKEN_CLEAN_RE = re.compile(r"[^\w\s]+", re.UNICODE)
SUPPORTED_COMPRESSION_ALGOS = ("lzma", "gzip", "bz2", "zlib")
DEFAULT_TOKENIZATION: dict[str, Any] = {
    "method": "legacy",
    "encoding_name": "cl100k_base",
    "include_punctuation": True,
}


def _legacy_preprocess_text(text: str) -> list[str]:
    """Legacy tokenizer: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = TOKEN_CLEAN_RE.sub(" ", text)
    return text.split()


def _resolve_tokenization_config(tokenization: Mapping[str, object] | None) -> dict[str, Any]:
    if tokenization is None:
        return DEFAULT_TOKENIZATION.copy()
    return {
        "method": tokenization.get("method", DEFAULT_TOKENIZATION["method"]),
        "encoding_name": tokenization.get(
            "encoding_name",
            DEFAULT_TOKENIZATION["encoding_name"],
        ),
        "include_punctuation": tokenization.get(
            "include_punctuation",
            DEFAULT_TOKENIZATION["include_punctuation"],
        ),
    }


def _tiktoken_preprocess_text(
    text: str,
    *,
    encoding_name: str,
    include_punctuation: bool,
) -> list[str]:
    try:
        import tiktoken
    except ModuleNotFoundError:
        # Compatibility fallback for environments where optional runtime deps are missing.
        return _legacy_preprocess_text(text)

    prepared_text = text if include_punctuation else TOKEN_CLEAN_RE.sub(" ", text)
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(prepared_text)
    return [
        encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        for token_id in token_ids
    ]


def preprocess_text(text: str, tokenization: Mapping[str, object] | None = None) -> list[str]:
    """Tokenize text according to configured tokenization method."""
    tokenization_cfg = _resolve_tokenization_config(tokenization)
    method = str(tokenization_cfg["method"]).lower()
    if method == "legacy":
        return _legacy_preprocess_text(text)
    if method == "tiktoken":
        return _tiktoken_preprocess_text(
            text,
            encoding_name=str(tokenization_cfg["encoding_name"]),
            include_punctuation=bool(tokenization_cfg["include_punctuation"]),
        )
    raise ValueError("tokenization method must be one of: legacy, tiktoken")


def _compression_ratio(window_text: str, algorithm: str = "lzma") -> float:
    """Compute compression ratio for a text window using a configured algorithm."""
    if algorithm not in SUPPORTED_COMPRESSION_ALGOS:
        raise ValueError("compression algorithm must be one of: lzma, gzip, bz2, zlib")

    raw_bytes = window_text.encode("utf-8")
    if len(raw_bytes) == 0:
        return 0.0

    if algorithm == "lzma":
        compressed = lzma.compress(raw_bytes)
    elif algorithm == "gzip":
        compressed = gzip.compress(raw_bytes)
    elif algorithm == "bz2":
        compressed = bz2.compress(raw_bytes)
    else:
        compressed = zlib.compress(raw_bytes)
    return len(compressed) / len(raw_bytes)


def _iter_window_tokens(tokens: list[str], window_size: int, step: int) -> list[list[str]]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive integers")
    if not tokens:
        return []

    if len(tokens) < window_size:
        return [tokens]

    windows: list[list[str]] = []
    for start in range(0, len(tokens) - window_size + 1, step):
        windows.append(tokens[start : start + window_size])
    return windows


def _analyze_window(
    window_tokens: list[str],
    *,
    mode: str,
    ref_dict: dict[str, float] | None,
    log_base: float,
    compression: str,
    unknown_prob: float,
) -> dict[str, float]:
    window_text = " ".join(window_tokens)

    if mode == "raw":
        mean_entropy = calculate_shannon_entropy(window_tokens, log_base)
        entropy_variance = entropy_variance_from_tokens(window_tokens, log_base)
    elif mode == "diff":
        if ref_dict is None:
            raise ValueError("ref_dict is required for diff mode")
        mean_entropy, entropy_variance = surprisal_stats_from_ref(
            window_tokens,
            ref_dict,
            log_base,
            unknown_prob,
        )
    else:
        raise ValueError("mode must be 'raw' or 'diff'")

    compression_ratio = _compression_ratio(window_text, compression)
    unique_ratio = len(set(window_tokens)) / len(window_tokens) if window_tokens else 0.0
    return {
        "mean_entropy": mean_entropy,
        "entropy_variance": entropy_variance,
        "compression_ratio": compression_ratio,
        "unique_ratio": unique_ratio,
    }


def analyze_tokens(
    tokens: list[str],
    mode: str,
    window_size: int,
    step: int,
    ref_dict: dict[str, float] | None = None,
    log_base: float = 2.718281828459045,
    compression: str = "lzma",
    unknown_prob: float = 1e-10,
    precomputed_windows: list[list[str]] | None = None,
) -> list[dict[str, float]]:
    """Analyze tokens with a sliding window and return per-window metrics."""
    windows = (
        precomputed_windows
        if precomputed_windows is not None
        else _iter_window_tokens(
            tokens,
            window_size,
            step,
        )
    )

    results: list[dict[str, float]] = []
    for window_id, window_tokens in enumerate(windows):
        row = _analyze_window(
            window_tokens,
            mode=mode,
            ref_dict=ref_dict,
            log_base=log_base,
            compression=compression,
            unknown_prob=unknown_prob,
        )
        results.append({"window_id": window_id, **row})
    return results


def analyze_tokens_batch(
    tokens: list[str],
    *,
    mode: str,
    window_size: int,
    step: int,
    ref_dicts: Mapping[str, dict[str, float]] | None = None,
    log_base: float = 2.718281828459045,
    compression: str = "lzma",
    unknown_prob: float = 1e-10,
) -> dict[str, list[dict[str, float]]]:
    """Batch analyzer optimized for tournament calls over the same token windows."""
    windows = _iter_window_tokens(tokens, window_size, step)

    if mode == "raw":
        return {
            "raw": analyze_tokens(
                tokens,
                mode="raw",
                window_size=window_size,
                step=step,
                log_base=log_base,
                compression=compression,
                unknown_prob=unknown_prob,
                precomputed_windows=windows,
            )
        }

    if mode != "diff":
        raise ValueError("mode must be 'raw' or 'diff'")
    if not ref_dicts:
        raise ValueError("ref_dicts is required for diff batch mode")

    results: dict[str, list[dict[str, float]]] = {}
    for name, ref_dict in ref_dicts.items():
        results[name] = analyze_tokens(
            tokens,
            mode="diff",
            window_size=window_size,
            step=step,
            ref_dict=ref_dict,
            log_base=log_base,
            compression=compression,
            unknown_prob=unknown_prob,
            precomputed_windows=windows,
        )
    return results
