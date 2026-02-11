"""Maxwell-Demon: Entropy-based text analysis toolkit."""

from .analyzer import analyze_tokens, analyze_tokens_batch, preprocess_text
from .metrics import (
    build_ref_dict,
    build_ref_dict_from_tokens,
    calculate_shannon_entropy,
    calculate_surprisal,
    entropy_variance_from_tokens,
    load_ref_dict,
    save_ref_dict,
    surprisal_stats_from_ref,
)
from .tournament import run_tournament

__all__ = [
    "analyze_tokens",
    "analyze_tokens_batch",
    "preprocess_text",
    "build_ref_dict",
    "build_ref_dict_from_tokens",
    "calculate_shannon_entropy",
    "calculate_surprisal",
    "entropy_variance_from_tokens",
    "load_ref_dict",
    "save_ref_dict",
    "surprisal_stats_from_ref",
    "run_tournament",
]
