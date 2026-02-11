from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_shadow_dataset.py"
SPEC = spec_from_file_location("generate_shadow_dataset", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load generate_shadow_dataset.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_human_to_ai_filename_canonical() -> None:
    assert MODULE._human_to_ai_filename(Path("001_human.txt")) == "001_ai.txt"


def test_human_to_ai_filename_legacy_suffix() -> None:
    assert MODULE._human_to_ai_filename(Path("051_human_Title.txt")) == "051_ai_Title.txt"


def test_build_incipit_normalizes_spaces() -> None:
    text = "  Prima riga\n\nseconda   riga  "
    assert MODULE._build_incipit(text, 15) == "Prima riga seco"


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {
        "openai": {"api_key_env": "OPENAI_API_KEY", "api_key": ""},
        "shadow_dataset": MODULE.DEFAULT_SHADOW_CONFIG,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    key = MODULE._resolve_api_key(cfg)

    assert key == "env-key"


def test_load_shadow_config_with_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[shadow_dataset]\nmodel = 'gpt-test'\n", encoding="utf-8")

    cfg = MODULE._load_shadow_config(cfg_file)

    assert cfg["shadow_dataset"]["model"] == "gpt-test"
    assert cfg["shadow_dataset"]["temperature"] == MODULE.DEFAULT_SHADOW_CONFIG["temperature"]


def test_is_temperature_unsupported_error_detects_message() -> None:
    exc = Exception("Unsupported parameter: 'temperature' is not supported with this model.")
    assert MODULE._is_temperature_unsupported_error(exc) is True


def test_create_response_skips_temperature_when_requested() -> None:
    class _FakeResponses:
        def __init__(self) -> None:
            self.payload: dict[str, object] = {}

        def create(self, **kwargs: object) -> dict[str, object]:
            self.payload = kwargs
            return {"ok": True}

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    client = _FakeClient()
    _ = MODULE._create_response(
        client=client,
        model="gpt-5-mini",
        temperature=0.8,
        max_output_tokens=300,
        system_prompt="system",
        user_prompt="user",
        include_temperature=False,
    )

    assert "temperature" not in client.responses.payload


def test_resolve_only_id_from_only_file() -> None:
    assert MODULE._resolve_only_id(None, "009_human.txt") == "009"


def test_select_human_files_by_only_id(tmp_path: Path) -> None:
    files = [
        tmp_path / "001_human.txt",
        tmp_path / "002_human.txt",
    ]

    selected = MODULE._select_human_files(files, "002", None)

    assert selected == [tmp_path / "002_human.txt"]


def test_select_human_files_raises_for_missing_id(tmp_path: Path) -> None:
    files = [tmp_path / "001_human.txt"]

    with pytest.raises(SystemExit, match="No human file found"):
        MODULE._select_human_files(files, "999", None)
