"""Tournament orchestrator for deterministic Human-vs-AI discernment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analyzer import analyze_tokens_batch, preprocess_text
from .metrics import load_ref_dict


def _collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.txt") if p.is_file()])
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _compute_delta_rows(
    *,
    filename: str,
    label: str,
    paisa_rows: list[dict[str, float]],
    synthetic_rows: list[dict[str, float]],
) -> list[dict[str, object]]:
    if len(paisa_rows) != len(synthetic_rows):
        raise ValueError("Mismatched window counts between reference analyses")

    output: list[dict[str, object]] = []
    for paisa_row, synthetic_row in zip(paisa_rows, synthetic_rows, strict=True):
        delta_h = float(paisa_row["mean_entropy"] - synthetic_row["mean_entropy"])
        output.append(
            {
                "filename": filename,
                "window_id": int(paisa_row["window_id"]),
                "label": label,
                "delta_h": delta_h,
                "burstiness_paisa": float(paisa_row["entropy_variance"]),
            }
        )
    return output


def run_tournament(
    *,
    human_input: str | Path,
    ai_input: str | Path,
    paisa_ref_path: str | Path,
    synthetic_ref_path: str | Path,
    window_size: int,
    step: int,
    log_base: float,
    output_path: str | Path = "tournament_results.csv",
    compression: str = "lzma",
) -> pd.DataFrame:
    """Run cross-reference tournament and persist deterministic delta output."""
    ref_paisa = load_ref_dict(str(paisa_ref_path))
    ref_synthetic = load_ref_dict(str(synthetic_ref_path))

    grouped_files = {
        "human": _collect_input_files(Path(human_input)),
        "ai": _collect_input_files(Path(ai_input)),
    }

    rows: list[dict[str, object]] = []
    for label, files in grouped_files.items():
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            tokens = preprocess_text(text)
            by_ref = analyze_tokens_batch(
                tokens,
                mode="diff",
                window_size=window_size,
                step=step,
                ref_dicts={"paisa": ref_paisa, "synthetic": ref_synthetic},
                log_base=log_base,
                compression=compression,
            )
            rows.extend(
                _compute_delta_rows(
                    filename=file_path.name,
                    label=label,
                    paisa_rows=by_ref["paisa"],
                    synthetic_rows=by_ref["synthetic"],
                )
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    return frame
