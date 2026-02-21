from __future__ import annotations

from pathlib import Path

from maxwell_demon import metrics
from maxwell_demon.standard_cli import run_standard_workflow


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    human_dir = tmp_path / "dataset" / "human"
    ai_dir = tmp_path / "dataset" / "ai"
    _write_text(human_dir / "001_human.txt", "parola umana parola umana naturale lessico umano")
    _write_text(ai_dir / "001_ai.txt", "testo generato macchina token modello artificiale")
    return human_dir, ai_dir


def _build_refs(tmp_path: Path) -> tuple[Path, Path]:
    paisa = tmp_path / "refs" / "paisa.json"
    synthetic = tmp_path / "refs" / "synthetic.json"
    paisa.parent.mkdir(parents=True, exist_ok=True)
    metrics.save_ref_dict(
        {"parola": 0.25, "umana": 0.25, "naturale": 0.25, "lessico": 0.25},
        str(paisa),
    )
    metrics.save_ref_dict(
        {"testo": 0.2, "generato": 0.2, "macchina": 0.2, "token": 0.2, "modello": 0.2},
        str(synthetic),
    )
    return paisa, synthetic


def _write_config(tmp_path: Path, paisa: Path, synthetic: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[analysis]
window = 2
step = 1
log_base = 2.0

[compression]
algorithm = "lzma"

[tokenization]
method = "legacy"
encoding_name = "cl100k_base"
include_punctuation = true
fallback_to_legacy_if_tiktoken_missing = true

[reference]
paisa_path = "{paisa}"
synthetic_path = "{synthetic}"
paisa_url = "https://example.test/paisa.gz"
paisa_corpus_path = "data/reference/paisa.txt"
synthetic_url = "https://example.test/synthetic.gz"
synthetic_corpus_path = "data/reference/synthetic.txt"
smoothing_k = 1.0

[output]
data_dir = "results/{{dataset}}/data"
plot_dir = "results/{{dataset}}/plot"

[standard]
compressions = ["lzma", "gzip"]

[standard.human_only]
aggregate_metrics = ["mean_entropy", "compression_ratio"]
aggregate_stats = ["mean", "median", "p90"]
group_by = ["filename", "label", "mode", "reference", "compression"]

[standard.plots]
enabled = true
density_threshold = 1
""",
        encoding="utf-8",
    )
    return cfg


def test_standard_human_only_generates_workflow_scoped_outputs(tmp_path: Path) -> None:
    human_dir, ai_dir = _build_dataset(tmp_path)
    paisa, synthetic = _build_refs(tmp_path)
    cfg = _write_config(tmp_path, paisa, synthetic)
    output_root = tmp_path / "results"

    manifest = run_standard_workflow(
        workflow="human-only",
        human_input=human_dir,
        ai_input=ai_dir,
        config_path=str(cfg),
        dataset="demo_set",
        output_root=output_root,
    )

    base = output_root / "demo_set" / "human-only"
    assert manifest["workflow"] == "human-only"
    assert (base / "data" / "labeled_lzma.csv").exists()
    assert (base / "data" / "doc_level_lzma.csv").exists()
    assert (base / "data" / "separation_lzma.csv").exists()
    assert (base / "plot" / "box_mean_entropy_by_label_lzma.png").exists()
    assert (base / "plot" / "phase_density_mean_entropy_vs_compression_ratio_by_label_lzma.html").exists()
    assert (base / "data" / "labeled_gzip.csv").exists()
    assert (base / "run_manifest.json").exists()


def test_standard_tournament_generates_workflow_scoped_outputs(tmp_path: Path) -> None:
    human_dir, ai_dir = _build_dataset(tmp_path)
    paisa, synthetic = _build_refs(tmp_path)
    cfg = _write_config(tmp_path, paisa, synthetic)
    output_root = tmp_path / "results"

    manifest = run_standard_workflow(
        workflow="tournament",
        human_input=human_dir,
        ai_input=ai_dir,
        config_path=str(cfg),
        dataset="demo_set",
        output_root=output_root,
    )

    base = output_root / "demo_set" / "tournament"
    assert manifest["workflow"] == "tournament"
    assert (base / "data" / "delta_lzma.csv").exists()
    assert (base / "data" / "delta_lzma.md").exists()
    assert (base / "plot" / "phase_delta_h_vs_burstiness_paisa_lzma.html").exists()
    assert (base / "plot" / "box_delta_h_by_label_lzma.png").exists()
    assert (base / "data" / "delta_gzip.csv").exists()
    assert (base / "run_manifest.json").exists()


def test_standard_dry_run_only_returns_plan(tmp_path: Path) -> None:
    human_dir, ai_dir = _build_dataset(tmp_path)
    paisa, synthetic = _build_refs(tmp_path)
    cfg = _write_config(tmp_path, paisa, synthetic)
    output_root = tmp_path / "results"

    manifest = run_standard_workflow(
        workflow="human-only",
        human_input=human_dir,
        ai_input=ai_dir,
        config_path=str(cfg),
        dataset="demo_set",
        output_root=output_root,
        dry_run=True,
    )

    assert manifest["dry_run"] is True
    assert len(manifest["artifacts"]) > 0
    assert not (output_root / "demo_set" / "human-only" / "run_manifest.json").exists()
