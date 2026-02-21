"""Config loading for Maxwell-Demon."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib

DEFAULT_CONFIG: dict[str, object] = {
    "analysis": {
        "window": 50,
        "step": 10,
        "log_base": 2.718281828459045,
    },
    "compression": {
        "algorithm": "lzma",
    },
    "tokenization": {
        "method": "tiktoken",
        "encoding_name": "cl100k_base",
        "include_punctuation": True,
        "fallback_to_legacy_if_tiktoken_missing": True,
    },
    "reference": {
        "paisa_path": "data/reference/paisa_ref_dict.json",
        "synthetic_path": "data/reference/synthetic_ref_dict.json",
        "paisa_url": "https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3/paisa.raw.utf8.gz",
        "paisa_corpus_path": "data/reference/paisa_corpus.txt",
        "synthetic_url": "https://example.com/synthetic_corpus.txt",
        "synthetic_corpus_path": "data/reference/synthetic_corpus.txt",
        "smoothing_k": 1.0,
    },
    "output": {
        "data_dir": "results/{dataset}/data",
        "plot_dir": "results/{dataset}/plot",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",
    },
    "shadow_dataset": {
        "model": "gpt-4.1-mini",
        "temperature": 0.8,
        "incipit_chars": 100,
        "max_output_tokens": 1800,
        "system_prompt": (
            "Agisci come un saggista esperto. Scrivi saggi lunghi, complessi, "
            "senza elenchi puntati, ricchi di subordinate e vocabolario vario."
        ),
        "user_prompt_template": (
            "Scrivi un saggio originale di circa 1000 parole basato su questo titolo: "
            "'{TITLE}'. Usa questo incipit come ispirazione per il tono: '{INCIPIT}'."
        ),
    },
    "standard": {
        "compressions": ["lzma", "gzip"],
        "human_only": {
            "aggregate_metrics": [
                "mean_entropy",
                "entropy_variance",
                "compression_ratio",
                "unique_ratio",
            ],
            "aggregate_stats": ["mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"],
            "group_by": ["filename", "label", "mode", "reference", "compression"],
        },
        "plots": {
            "enabled": True,
            "density_threshold": 2000,
        },
    },
}


def load_config(path: str | Path) -> dict[str, object]:
    """Load TOML config, merging with defaults."""
    cfg_path = Path(path)
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    merged: dict[str, object] = {
        "analysis": {**DEFAULT_CONFIG["analysis"], **data.get("analysis", {})},
        "compression": {**DEFAULT_CONFIG["compression"], **data.get("compression", {})},
        "tokenization": {**DEFAULT_CONFIG["tokenization"], **data.get("tokenization", {})},
        "reference": {**DEFAULT_CONFIG["reference"], **data.get("reference", {})},
        "output": {**DEFAULT_CONFIG["output"], **data.get("output", {})},
        "openai": {**DEFAULT_CONFIG["openai"], **data.get("openai", {})},
        "shadow_dataset": {**DEFAULT_CONFIG["shadow_dataset"], **data.get("shadow_dataset", {})},
        "standard": {
            **DEFAULT_CONFIG["standard"],
            **data.get("standard", {}),
        },
    }
    if isinstance(merged["standard"], dict):
        raw_standard = merged["standard"]
        default_standard = DEFAULT_CONFIG["standard"]
        if isinstance(raw_standard.get("human_only"), dict):
            raw_standard["human_only"] = {
                **default_standard["human_only"],
                **raw_standard["human_only"],
            }
        if isinstance(raw_standard.get("plots"), dict):
            raw_standard["plots"] = {**default_standard["plots"], **raw_standard["plots"]}
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
    if compression not in {"lzma", "gzip", "bz2", "zlib"}:
        raise ValueError("compression.algorithm must be one of: lzma, gzip, bz2, zlib")

    tokenization = cfg["tokenization"]
    method = tokenization["method"]
    encoding_name = tokenization["encoding_name"]
    include_punctuation = tokenization["include_punctuation"]
    fallback_to_legacy = tokenization["fallback_to_legacy_if_tiktoken_missing"]
    if method not in {"legacy", "tiktoken"}:
        raise ValueError("tokenization.method must be one of: legacy, tiktoken")
    if not isinstance(encoding_name, str) or not encoding_name.strip():
        raise ValueError("tokenization.encoding_name must be a non-empty string")
    if not isinstance(include_punctuation, bool):
        raise ValueError("tokenization.include_punctuation must be a boolean")
    if not isinstance(fallback_to_legacy, bool):
        raise ValueError(
            "tokenization.fallback_to_legacy_if_tiktoken_missing must be a boolean"
        )

    reference = cfg["reference"]
    paisa_path = reference["paisa_path"]
    synthetic_path = reference["synthetic_path"]
    paisa_url = reference["paisa_url"]
    paisa_corpus_path = reference["paisa_corpus_path"]
    synthetic_url = reference["synthetic_url"]
    synthetic_corpus_path = reference["synthetic_corpus_path"]
    smoothing_k = reference["smoothing_k"]
    if not isinstance(paisa_path, str) or not paisa_path:
        raise ValueError("reference.paisa_path must be a non-empty string")
    if not isinstance(synthetic_path, str) or not synthetic_path:
        raise ValueError("reference.synthetic_path must be a non-empty string")
    if not isinstance(paisa_url, str) or not paisa_url:
        raise ValueError("reference.paisa_url must be a non-empty string")
    if not isinstance(paisa_corpus_path, str) or not paisa_corpus_path:
        raise ValueError("reference.paisa_corpus_path must be a non-empty string")
    if not isinstance(synthetic_url, str) or not synthetic_url:
        raise ValueError("reference.synthetic_url must be a non-empty string")
    if not isinstance(synthetic_corpus_path, str) or not synthetic_corpus_path:
        raise ValueError("reference.synthetic_corpus_path must be a non-empty string")
    if not isinstance(smoothing_k, (int, float)) or float(smoothing_k) < 0:
        raise ValueError("reference.smoothing_k must be a non-negative number")

    output = cfg["output"]
    for key in ("data_dir", "plot_dir"):
        value = output[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"output.{key} must be a non-empty string")

    openai_cfg = cfg["openai"]
    if not isinstance(openai_cfg["api_key_env"], str) or not openai_cfg["api_key_env"].strip():
        raise ValueError("openai.api_key_env must be a non-empty string")
    if not isinstance(openai_cfg["api_key"], str):
        raise ValueError("openai.api_key must be a string")

    shadow_cfg = cfg["shadow_dataset"]
    if not isinstance(shadow_cfg["model"], str) or not shadow_cfg["model"].strip():
        raise ValueError("shadow_dataset.model must be a non-empty string")
    if (
        not isinstance(shadow_cfg["temperature"], (int, float))
        or not 0 <= float(shadow_cfg["temperature"]) <= 2
    ):
        raise ValueError("shadow_dataset.temperature must be between 0 and 2")
    if not isinstance(shadow_cfg["incipit_chars"], int) or shadow_cfg["incipit_chars"] <= 0:
        raise ValueError("shadow_dataset.incipit_chars must be a positive integer")
    if not isinstance(shadow_cfg["max_output_tokens"], int) or shadow_cfg["max_output_tokens"] <= 0:
        raise ValueError("shadow_dataset.max_output_tokens must be a positive integer")
    if not isinstance(shadow_cfg["system_prompt"], str) or not shadow_cfg["system_prompt"].strip():
        raise ValueError("shadow_dataset.system_prompt must be a non-empty string")
    if (
        not isinstance(shadow_cfg["user_prompt_template"], str)
        or not shadow_cfg["user_prompt_template"].strip()
    ):
        raise ValueError("shadow_dataset.user_prompt_template must be a non-empty string")

    standard_cfg = cfg["standard"]
    if not isinstance(standard_cfg, dict):
        raise ValueError("standard must be a TOML table")
    compressions = standard_cfg["compressions"]
    if not isinstance(compressions, list) or not compressions:
        raise ValueError("standard.compressions must be a non-empty list")
    if any(not isinstance(item, str) or item not in {"lzma", "gzip", "bz2", "zlib"} for item in compressions):
        raise ValueError("standard.compressions items must be among: lzma, gzip, bz2, zlib")

    human_only_cfg = standard_cfg["human_only"]
    if not isinstance(human_only_cfg, dict):
        raise ValueError("standard.human_only must be a TOML table")
    for key in ("aggregate_metrics", "aggregate_stats", "group_by"):
        value = human_only_cfg[key]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"standard.human_only.{key} must be a non-empty string list")

    plots_cfg = standard_cfg["plots"]
    if not isinstance(plots_cfg, dict):
        raise ValueError("standard.plots must be a TOML table")
    if not isinstance(plots_cfg["enabled"], bool):
        raise ValueError("standard.plots.enabled must be a boolean")
    if not isinstance(plots_cfg["density_threshold"], int) or plots_cfg["density_threshold"] <= 0:
        raise ValueError("standard.plots.density_threshold must be a positive integer")
