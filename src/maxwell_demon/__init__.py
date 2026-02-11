"""Maxwell-Demon: Entropy-based text analysis toolkit."""

from .analyzer import analyze_tokens, preprocess_text
from .metrics import (
    build_ref_dict,
    build_ref_dict_from_frequency_list,
    calculate_shannon_entropy,
    calculate_surprisal,
    download_frequency_list,
    entropy_variance_from_tokens,
    load_ref_dict,
    save_ref_dict,
    surprisal_stats_from_ref,
)

__all__ = [
    "analyze_tokens",
    "preprocess_text",
    "build_ref_dict",
    "build_ref_dict_from_frequency_list",
    "calculate_shannon_entropy",
    "calculate_surprisal",
    "download_frequency_list",
    "entropy_variance_from_tokens",
    "load_ref_dict",
    "save_ref_dict",
    "surprisal_stats_from_ref",
]
