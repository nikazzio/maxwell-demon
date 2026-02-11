"""Sliding-window analyzer for Maxwell-Demon."""

import bz2
import gzip
import lzma
import re
import zlib

from .metrics import (
    calculate_shannon_entropy,
    entropy_variance_from_tokens,
    surprisal_stats_from_ref,
)

TOKEN_CLEAN_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def preprocess_text(text: str) -> list[str]:
    """Lowercase, remove basic punctuation, and split into tokens."""
    text = text.lower()
    text = TOKEN_CLEAN_RE.sub(" ", text)
    tokens = text.split()
    return tokens


def _compression_ratio(window_text: str, algorithm: str) -> float:
    """Compute compression ratio for a text window using a given algorithm."""
    raw_bytes = window_text.encode("utf-8")
    if len(raw_bytes) == 0:
        return 0.0
    if algorithm == "zlib":
        compressed = zlib.compress(raw_bytes)
    elif algorithm == "gzip":
        compressed = gzip.compress(raw_bytes)
    elif algorithm == "bz2":
        compressed = bz2.compress(raw_bytes)
    elif algorithm == "lzma":
        compressed = lzma.compress(raw_bytes)
    else:
        raise ValueError("compression algorithm must be one of: zlib, gzip, bz2, lzma")
    return len(compressed) / len(raw_bytes)


def analyze_tokens(
    tokens: list[str],
    mode: str,
    window_size: int,
    step: int,
    ref_dict: dict[str, float] | None = None,
    log_base: float = 2.718281828459045,
    compression: str = "zlib",
    unknown_prob: float = 1e-10,
) -> list[dict[str, float]]:
    """Analyze tokens with a sliding window and return per-window metrics."""
    results: list[dict[str, float]] = []
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive integers")

    if len(tokens) < window_size and tokens:
        window_starts = [0]
    else:
        window_starts = range(0, max(len(tokens) - window_size + 1, 0), step)

    for start in window_starts:
        window_tokens = tokens[start : start + window_size]
        window_text = " ".join(window_tokens)

        if mode == "raw":
            mean_entropy = calculate_shannon_entropy(window_tokens, log_base)
            entropy_variance = entropy_variance_from_tokens(window_tokens, log_base)
        elif mode == "diff":
            if ref_dict is None:
                raise ValueError("ref_dict is required for diff mode")
            mean_entropy, entropy_variance = surprisal_stats_from_ref(
                window_tokens, ref_dict, log_base, unknown_prob
            )
        else:
            raise ValueError("mode must be 'raw' or 'diff'")

        compression_ratio = _compression_ratio(window_text, compression)
        unique_ratio = len(set(window_tokens)) / len(window_tokens) if window_tokens else 0.0

        results.append(
            {
                "window_id": len(results),
                "mean_entropy": mean_entropy,
                "entropy_variance": entropy_variance,
                "compression_ratio": compression_ratio,
                "unique_ratio": unique_ratio,
            }
        )

    return results
