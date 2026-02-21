"""Helpers to derive dataset-aware output paths."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

GENERIC_NAMES = {"", ".", "data", "results", "plot", "plots", "human", "ai"}


def _sanitize_dataset_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or "default"


def _candidate_from_path(path: Path) -> str | None:
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if path.suffix:
        return path.parent.name
    return path.name


def infer_dataset_name(paths: list[str | Path]) -> str:
    """Infer a stable dataset name from one or more input paths."""
    candidates: list[str] = []
    for raw in paths:
        candidate = _candidate_from_path(Path(raw))
        if candidate is None:
            continue
        normalized = _sanitize_dataset_name(candidate)
        if normalized in GENERIC_NAMES:
            continue
        candidates.append(normalized)

    if not candidates:
        return "default"

    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def resolve_output_template(template: str, dataset: str) -> str:
    """Render an output path template with the inferred dataset name."""
    if "{dataset}" in template:
        return template.format(dataset=dataset)
    return template


def _slug(value: str) -> str:
    return _sanitize_dataset_name(value).replace("-", "_")


def single_output_filename(
    mode: str,
    reference: str | None = None,
    *,
    human_only: bool = False,
) -> str:
    """Build an explicit filename for single workflow outputs."""
    if human_only:
        return "single_human_only_paisa.csv"
    if mode == "diff" and reference:
        return f"single_diff_{_slug(reference)}.csv"
    return f"single_{_slug(mode)}.csv"


def tournament_output_filename() -> str:
    """Build the default filename for tournament outputs."""
    return "final_delta.csv"


def line_plot_filename(metric: str, extension: str) -> str:
    """Build a default filename for line plots."""
    return f"line_{_slug(metric)}.{extension.lstrip('.')}"


def phase_plot_filename(x_col: str, y_col: str) -> str:
    """Build a default filename for phase plots."""
    return f"phase_{_slug(x_col)}_vs_{_slug(y_col)}.html"
