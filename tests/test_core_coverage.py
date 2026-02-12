from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from maxwell_demon import analyzer, cli, config, metrics, tournament


def _base_args(**overrides: object) -> SimpleNamespace:
    args = {
        "config": None,
        "mode": "raw",
        "window": 3,
        "step": 1,
        "output": None,
        "output_dir": None,
        "label": "human",
        "reference": "paisa",
        "ref_dict": None,
        "log_base": 2.0,
        "compression": "lzma",
        "input": None,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_load_config_merges_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[analysis]
window = 5

[reference]
paisa_path = "paisa.json"
""",
        encoding="utf-8",
    )

    loaded = config.load_config(cfg)

    assert loaded["analysis"]["window"] == 5
    assert loaded["compression"]["algorithm"] == "lzma"
    assert loaded["reference"]["paisa_path"] == "paisa.json"
    assert loaded["reference"]["synthetic_path"] == "data/reference/synthetic_ref_dict.json"
    assert isinstance(loaded["reference"]["paisa_url"], str)
    assert loaded["reference"]["paisa_corpus_path"] == "data/reference/paisa_corpus.txt"
    assert isinstance(loaded["reference"]["synthetic_url"], str)
    assert loaded["reference"]["synthetic_corpus_path"] == "data/reference/synthetic_corpus.txt"
    assert loaded["reference"]["smoothing_k"] == 1.0
    assert loaded["tokenization"]["method"] == "tiktoken"
    assert loaded["tokenization"]["encoding_name"] == "cl100k_base"
    assert loaded["tokenization"]["include_punctuation"] is True
    assert loaded["output"]["data_dir"] == "results/{dataset}/data"
    assert loaded["output"]["plot_dir"] == "results/{dataset}/plot"


def test_load_config_accepts_supported_compression_algorithms(tmp_path: Path) -> None:
    cfg = tmp_path / "ok.toml"
    cfg.write_text("[compression]\nalgorithm = 'zlib'\n", encoding="utf-8")

    loaded = config.load_config(cfg)

    assert loaded["compression"]["algorithm"] == "zlib"


def test_load_config_rejects_unknown_compression(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[compression]\nalgorithm = 'snappy'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compression.algorithm"):
        config.load_config(cfg)


def test_load_config_rejects_negative_reference_smoothing(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_smoothing.toml"
    cfg.write_text("[reference]\nsmoothing_k = -0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reference.smoothing_k"):
        config.load_config(cfg)


def test_load_config_rejects_unknown_tokenization_method(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_tokenization.toml"
    cfg.write_text("[tokenization]\nmethod = 'unknown'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tokenization.method"):
        config.load_config(cfg)


def test_metrics_ref_dict_build_and_json_roundtrip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a a b", encoding="utf-8")

    ref = metrics.build_ref_dict(str(corpus), smoothing_k=0.0)
    out = tmp_path / "ref.json"
    metrics.save_ref_dict(ref, str(out))
    loaded = metrics.load_ref_dict(str(out))

    assert pytest.approx(sum(ref.values())) == 1.0
    assert loaded == ref


def test_analyzer_compression_and_batch_diff() -> None:
    tokens = ["uno", "due", "uno", "tre"]

    for algo in ("lzma", "gzip", "bz2", "zlib"):
        ratio = analyzer._compression_ratio(" ".join(tokens), algo)
        assert ratio > 0

    with pytest.raises(ValueError, match="compression algorithm"):
        analyzer._compression_ratio(" ".join(tokens), "snappy")

    by_ref = analyzer.analyze_tokens_batch(
        tokens,
        mode="diff",
        window_size=2,
        step=1,
        ref_dicts={"paisa": {"uno": 0.6, "due": 0.2, "tre": 0.2}},
    )
    assert "paisa" in by_ref
    assert by_ref["paisa"]
    assert "mean_entropy" in by_ref["paisa"][0]


def test_preprocess_text_legacy_mode_preserves_current_behavior() -> None:
    text = "Hello, WORLD! Ciao?"
    tokens = analyzer.preprocess_text(text, tokenization={"method": "legacy"})
    assert tokens == ["hello", "world", "ciao"]


def test_collect_input_files_file_dir_and_missing(tmp_path: Path) -> None:
    one = tmp_path / "one.txt"
    one.write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    two = sub / "two.txt"
    two.write_text("y", encoding="utf-8")

    as_file = cli._collect_input_files(one)
    as_dir = cli._collect_input_files(tmp_path)

    assert as_file == [one]
    assert as_dir == [one, two]

    with pytest.raises(FileNotFoundError):
        cli._collect_input_files(tmp_path / "missing")


def test_cli_main_raw_generates_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("a b c d e", encoding="utf-8")
    out = tmp_path / "out.csv"

    args = _base_args(input=str(input_file), output=str(out), mode="raw")
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    cli.main()

    assert out.exists()
    frame = pd.read_csv(out)
    assert {"filename", "mean_entropy", "compression", "mode"}.issubset(frame.columns)
    assert frame["compression"].iloc[0] == "lzma"


def test_cli_main_diff_uses_reference_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("a b c", encoding="utf-8")

    ref_path = tmp_path / "paisa.json"
    metrics.save_ref_dict({"a": 0.5, "b": 0.3, "c": 0.2}, str(ref_path))

    cfg = config.DEFAULT_CONFIG.copy()
    cfg["reference"] = {
        "paisa_path": str(ref_path),
        "synthetic_path": str(tmp_path / "synthetic.json"),
    }

    monkeypatch.setattr(cli, "DEFAULT_CONFIG", cfg)
    args = _base_args(input=str(input_file), mode="diff", output=str(tmp_path / "out.csv"))
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    cli.main()

    frame = pd.read_csv(tmp_path / "out.csv")
    assert frame["mode"].iloc[0] == "diff"
    assert frame["reference"].iloc[0] == "paisa"


def test_run_tournament_generates_expected_columns(tmp_path: Path) -> None:
    human = tmp_path / "human"
    ai = tmp_path / "ai"
    human.mkdir()
    ai.mkdir()

    (human / "h1.txt").write_text("parola parola umana", encoding="utf-8")
    (ai / "a1.txt").write_text("token sintetico sintetico", encoding="utf-8")

    paisa_ref = tmp_path / "paisa.json"
    synthetic_ref = tmp_path / "synthetic.json"
    metrics.save_ref_dict({"parola": 0.5, "umana": 0.5}, str(paisa_ref))
    metrics.save_ref_dict({"token": 0.5, "sintetico": 0.5}, str(synthetic_ref))

    out = tmp_path / "tournament_results.csv"
    frame = tournament.run_tournament(
        human_input=human,
        ai_input=ai,
        paisa_ref_path=paisa_ref,
        synthetic_ref_path=synthetic_ref,
        window_size=2,
        step=1,
        log_base=2.0,
        output_path=out,
    )

    assert out.exists()
    assert len(frame) > 0
    assert {"filename", "window_id", "label", "delta_h", "burstiness_paisa"}.issubset(frame.columns)


def _load_prepare_resources_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_resources.py"
    spec = importlib.util.spec_from_file_location("prepare_resources", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/prepare_resources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_resources_reads_local_gzip_with_skip_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_resources = _load_prepare_resources_module()
    paisa_gz = tmp_path / "paisa_corpus.txt"
    with gzip.open(paisa_gz, "wt", encoding="utf-8") as handle:
        handle.write("parola parola umana")

    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "s1.txt").write_text("token sintetico", encoding="utf-8")

    human_dict_out = tmp_path / "paisa_ref.json"
    synthetic_dict_out = tmp_path / "synthetic_ref.json"
    args = SimpleNamespace(
        config=None,
        paisa_url=None,
        paisa_corpus_out=str(paisa_gz),
        synthetic_input=str(synthetic_dir),
        human_dict_out=str(human_dict_out),
        synthetic_dict_out=str(synthetic_dict_out),
        smoothing_k=0.0,
        skip_download=True,
    )
    monkeypatch.setattr(prepare_resources, "_parse_args", lambda: args)

    prepare_resources.main()

    human_ref = metrics.load_ref_dict(str(human_dict_out))
    synthetic_ref = metrics.load_ref_dict(str(synthetic_dict_out))
    assert "parola" in human_ref
    assert "sintetico" in synthetic_ref


def test_prepare_resources_download_decompresses_gzip_to_plain_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_resources = _load_prepare_resources_module()
    gz_payload = gzip.compress(b"umano umano lessico")

    def fake_urlretrieve(_: str, output_path: Path) -> tuple[str, object]:
        Path(output_path).write_bytes(gz_payload)
        return str(output_path), None

    monkeypatch.setattr(prepare_resources, "urlretrieve", fake_urlretrieve)
    out_path = tmp_path / "paisa_corpus.txt"
    downloaded = prepare_resources._download_paisa("https://example.test/paisa.gz", out_path)

    assert downloaded == out_path
    assert out_path.read_text(encoding="utf-8") == "umano umano lessico"


def test_prepare_resources_downloads_synthetic_from_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_resources = _load_prepare_resources_module()
    payloads = {
        "https://example.test/paisa.gz": gzip.compress(b"parola umana parola"),
        "https://example.test/synthetic.gz": gzip.compress(b"sintetico token sintetico"),
    }

    def fake_urlretrieve(url: str, output_path: Path) -> tuple[str, object]:
        Path(output_path).write_bytes(payloads[url])
        return str(output_path), None

    monkeypatch.setattr(prepare_resources, "urlretrieve", fake_urlretrieve)

    args = SimpleNamespace(
        config=None,
        paisa_url="https://example.test/paisa.gz",
        paisa_corpus_out=str(tmp_path / "paisa.txt"),
        synthetic_input=None,
        synthetic_url="https://example.test/synthetic.gz",
        synthetic_corpus_out=str(tmp_path / "synthetic.txt"),
        human_dict_out=str(tmp_path / "human.json"),
        synthetic_dict_out=str(tmp_path / "synthetic.json"),
        smoothing_k=0.0,
        skip_download=False,
    )
    monkeypatch.setattr(prepare_resources, "_parse_args", lambda: args)

    prepare_resources.main()

    human_ref = metrics.load_ref_dict(str(tmp_path / "human.json"))
    synthetic_ref = metrics.load_ref_dict(str(tmp_path / "synthetic.json"))
    assert "umana" in human_ref
    assert "sintetico" in synthetic_ref
