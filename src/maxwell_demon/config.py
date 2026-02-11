"""Config loading for Maxwell-Demon."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib

DEFAULT_CONFIG: dict[str, object] = {
    "analysis": {
        "mode": "raw",
        "window": 50,
        "step": 10,
        "log_base": 2.718281828459045,
    },
    "compression": {
        "algorithm": "zlib",
    },
    "reference": {
        "path": None,
        "freq_url": None,
        "freq_cache": "data/frequency_list.txt",
        "smoothing_k": 0.0,
        "unknown_prob": 1e-10,
    },
}


def load_config(path: str | Path) -> dict[str, object]:
    """Load TOML config, merging with defaults."""
    cfg_path = Path(path)
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    merged: dict[str, object] = {
        "analysis": {**DEFAULT_CONFIG["analysis"], **data.get("analysis", {})},
        "compression": {**DEFAULT_CONFIG["compression"], **data.get("compression", {})},
        "reference": {**DEFAULT_CONFIG["reference"], **data.get("reference", {})},
    }
    _validate_config(merged)
    return merged


def _validate_config(cfg: dict[str, object]) -> None:
    analysis = cfg["analysis"]
    window = analysis["window"]
    step = analysis["step"]
    log_base = analysis["log_base"]

    if not isinstance(window, int) or window <= 0:
        raise ValueError("analysis.window must be a positive integer")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("analysis.step must be a positive integer")
    if not isinstance(log_base, (int, float)) or log_base <= 0 or log_base == 1.0:
        raise ValueError("analysis.log_base must be > 0 and != 1")

    compression = cfg["compression"]["algorithm"]
    if compression not in {"zlib", "gzip", "bz2", "lzma"}:
        raise ValueError("compression.algorithm must be one of: zlib, gzip, bz2, lzma")

    reference = cfg["reference"]
    smoothing_k = reference["smoothing_k"]
    unknown_prob = reference["unknown_prob"]
    freq_url = reference["freq_url"]
    freq_cache = reference["freq_cache"]
    if not isinstance(smoothing_k, (int, float)) or smoothing_k < 0:
        raise ValueError("reference.smoothing_k must be >= 0")
    if not isinstance(unknown_prob, (int, float)) or unknown_prob <= 0:
        raise ValueError("reference.unknown_prob must be > 0")
    if freq_url is not None and not isinstance(freq_url, str):
        raise ValueError("reference.freq_url must be a string or null")
    if not isinstance(freq_cache, str) or not freq_cache:
        raise ValueError("reference.freq_cache must be a non-empty string")
