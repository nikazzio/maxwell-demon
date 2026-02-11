"""Entropy and surprisal metrics for Maxwell-Demon."""

import json
import math
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.stats import entropy as scipy_entropy

EPSILON = 1e-10


def _validate_log_base(log_base: float) -> None:
    """Validate log base to avoid invalid or degenerate values."""
    if log_base <= 0 or log_base == 1.0:
        raise ValueError("log_base must be > 0 and != 1")


def calculate_shannon_entropy(tokens: list[str], log_base: float = math.e) -> float:
    """Compute Shannon entropy for a token list."""
    _validate_log_base(log_base)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    probs = np.clip(probs, EPSILON, 1.0)
    return float(scipy_entropy(probs, base=log_base))


def _surprisal_from_probs(
    tokens: list[str],
    prob_lookup: dict[str, float],
    log_base: float,
    unknown_prob: float,
) -> np.ndarray:
    """Compute token-level surprisal given a probability lookup."""
    _validate_log_base(log_base)
    if not tokens:
        return np.array([], dtype=float)
    probs = np.array([prob_lookup.get(t, unknown_prob) for t in tokens], dtype=float)
    probs = np.clip(probs, EPSILON, 1.0)
    if log_base == math.e:
        return -np.log(probs)
    return -np.log(probs) / math.log(log_base)


def calculate_surprisal(
    token: str,
    ref_dict: dict[str, float],
    log_base: float = math.e,
    unknown_prob: float = EPSILON,
) -> float:
    """Compute surprisal for a single token against a reference dictionary."""
    _validate_log_base(log_base)
    p = ref_dict.get(token, unknown_prob)
    p = max(p, EPSILON)
    if log_base == math.e:
        return -math.log(p)
    return -math.log(p) / math.log(log_base)


def entropy_variance_from_tokens(tokens: list[str], log_base: float = math.e) -> float:
    """Compute variance of token surprisal within a window."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    prob_lookup = {t: c / total for t, c in counts.items()}
    surprisals = _surprisal_from_probs(tokens, prob_lookup, log_base, EPSILON)
    if surprisals.size == 0:
        return 0.0
    return float(np.var(surprisals))


def surprisal_stats_from_ref(
    tokens: list[str],
    ref_dict: dict[str, float],
    log_base: float = math.e,
    unknown_prob: float = EPSILON,
) -> tuple[float, float]:
    """Compute mean and variance of surprisal for a window vs a reference dict."""
    surprisals = _surprisal_from_probs(tokens, ref_dict, log_base, unknown_prob)
    if surprisals.size == 0:
        return 0.0, 0.0
    return float(np.mean(surprisals)), float(np.var(surprisals))


def build_ref_dict(corpus_path: str, smoothing_k: float = 0.0) -> dict[str, float]:
    """Build a token->probability reference dictionary from a corpus text file."""
    from .analyzer import preprocess_text

    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess_text(text)
    counts = Counter(tokens)
    if smoothing_k < 0:
        raise ValueError("smoothing_k must be >= 0")
    vocab_size = len(counts)
    total = sum(counts.values())
    if total == 0:
        return {}

    if smoothing_k == 0:
        return {t: c / total for t, c in counts.items()}

    denom = total + smoothing_k * vocab_size
    return {t: (c + smoothing_k) / denom for t, c in counts.items()}


def download_frequency_list(url: str, output_path: str) -> Path:
    """Download a frequency list file from a URL."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, out_path)
    return out_path


def build_ref_dict_from_frequency_list(path: str) -> dict[str, float]:
    """Build a token->probability dictionary from a 'word frequency' list file."""
    total = 0
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word, freq = parts[0], parts[1]
            try:
                count = int(freq)
            except ValueError:
                continue
            counts[word] = count
            total += count

    if total == 0:
        return {}
    return {w: c / total for w, c in counts.items()}


def save_ref_dict(ref_dict: dict[str, float], output_path: str) -> None:
    """Persist a reference dictionary as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_dict, f, ensure_ascii=False, indent=2)


def load_ref_dict(path: str) -> dict[str, float]:
    """Load a reference dictionary from JSON, enforcing token->prob mapping."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Reference dictionary JSON must be an object of token->prob.")
    return {str(k): float(v) for k, v in data.items()}
