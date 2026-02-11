from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from maxwell_demon import analyzer, cli, config, metrics


def _base_args(**overrides: object) -> SimpleNamespace:
    args = {
        "config": None,
        "mode": "raw",
        "window": 3,
        "step": 1,
        "output": None,
        "output_dir": None,
        "label": "human",
        "ref_dict": None,
        "build_ref_dict": None,
        "build_ref_dict_from_freq": None,
        "freq_url": None,
        "freq_cache": "data/frequency_list.txt",
        "ref_dict_out": "ref_dict.json",
        "log_base": 2.0,
        "compression": "zlib",
        "unknown_prob": 1e-10,
        "smoothing_k": 0.0,
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
unknown_prob = 0.01
""",
        encoding="utf-8",
    )

    loaded = config.load_config(cfg)

    assert loaded["analysis"]["mode"] == "raw"
    assert loaded["analysis"]["window"] == 5
    assert loaded["compression"]["algorithm"] == "zlib"
    assert loaded["reference"]["unknown_prob"] == 0.01


def test_load_config_rejects_invalid_values(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[analysis]\nwindow = 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="analysis.window"):
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


def test_metrics_frequency_list_ignores_invalid_rows(tmp_path: Path) -> None:
    freq = tmp_path / "freq.txt"
    freq.write_text("ciao 10\ninvalid\nmondo notanint\nmondo 5\n", encoding="utf-8")

    ref = metrics.build_ref_dict_from_frequency_list(str(freq))

    assert ref == {"ciao": 10 / 15, "mondo": 5 / 15}


def test_download_frequency_list_uses_urlretrieve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_urlretrieve(url: str, out_path: Path) -> tuple[str, object]:
        calls.append((url, out_path))
        out_path.write_text("x 1\n", encoding="utf-8")
        return str(out_path), None

    monkeypatch.setattr(metrics, "urlretrieve", fake_urlretrieve)

    out = metrics.download_frequency_list("https://example.com/freq", str(tmp_path / "freq.txt"))

    assert out.exists()
    assert calls[0][0] == "https://example.com/freq"


def test_analyzer_compression_and_modes() -> None:
    tokens = ["uno", "due", "uno", "tre"]

    for algo in ["zlib", "gzip", "bz2", "lzma"]:
        ratio = analyzer._compression_ratio(" ".join(tokens), algo)
        assert ratio > 0

    raw = analyzer.analyze_tokens(tokens, mode="raw", window_size=10, step=2)
    assert len(raw) == 1
    assert "mean_entropy" in raw[0]

    with pytest.raises(ValueError, match="ref_dict is required"):
        analyzer.analyze_tokens(tokens, mode="diff", window_size=2, step=1)

    with pytest.raises(ValueError, match="mode must be"):
        analyzer.analyze_tokens(tokens, mode="bad", window_size=2, step=1)


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


def test_cli_main_build_ref_dict_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a a b", encoding="utf-8")
    out = tmp_path / "dict.json"

    args = _base_args(build_ref_dict=str(corpus), ref_dict_out=str(out))
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    cli.main()

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["a"] > loaded["b"]
    assert "Reference dictionary saved" in capsys.readouterr().out


def test_cli_main_build_ref_dict_from_freq_with_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloaded = tmp_path / "downloaded.txt"
    out = tmp_path / "dict.json"

    def fake_download(url: str, cache: str) -> Path:
        assert url == "https://example.com/freq"
        assert cache == str(tmp_path / "cache.txt")
        downloaded.write_text("a 2\nb 1\n", encoding="utf-8")
        return downloaded

    monkeypatch.setattr(cli, "download_frequency_list", fake_download)

    args = _base_args(
        build_ref_dict_from_freq=str(tmp_path / "ignored.txt"),
        freq_url="https://example.com/freq",
        freq_cache=str(tmp_path / "cache.txt"),
        ref_dict_out=str(out),
    )
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    cli.main()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"a": 2 / 3, "b": 1 / 3}


def test_cli_main_raw_generates_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("a b c d e", encoding="utf-8")
    out = tmp_path / "out.csv"

    args = _base_args(input=str(input_file), output=str(out), mode="raw")
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    cli.main()

    assert out.exists()
    frame = pd.read_csv(out)
    assert set(["filename", "mean_entropy", "compression", "mode"]).issubset(frame.columns)
    assert frame["mode"].iloc[0] == "raw"


def test_cli_main_diff_requires_ref_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("a b c", encoding="utf-8")

    args = _base_args(input=str(input_file), mode="diff", ref_dict=None)
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    with pytest.raises(SystemExit, match="--ref-dict is required"):
        cli.main()
