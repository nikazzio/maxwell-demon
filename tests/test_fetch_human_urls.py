from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scripts_fetch_human.py"
SPEC = spec_from_file_location("scripts_fetch_human", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load scripts_fetch_human.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_urls_from_txt_ignores_comments_and_empty_lines(tmp_path: Path) -> None:
    urls_path = tmp_path / "urls.txt"
    urls_path.write_text(
        "# comment\n\nhttps://example.com/a\n  https://example.com/b  \n",
        encoding="utf-8",
    )

    rows = MODULE._load_urls(urls_path)

    assert rows == [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]


def test_load_urls_from_json_returns_list_of_objects(tmp_path: Path) -> None:
    urls_path = tmp_path / "urls.json"
    urls_path.write_text('[{"url": "https://example.com/a", "id": "001"}]', encoding="utf-8")

    rows = MODULE._load_urls(urls_path)

    assert rows == [{"url": "https://example.com/a", "id": "001"}]


def test_load_urls_rejects_unsupported_extension(tmp_path: Path) -> None:
    urls_path = tmp_path / "urls.csv"
    urls_path.write_text("https://example.com/a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported URL file format"):
        MODULE._load_urls(urls_path)


def test_resolve_article_id_prefers_empty_stub(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "001_human.txt").write_text("", encoding="utf-8")
    (human_dir / "002_human.txt").write_text("filled", encoding="utf-8")

    article_id = MODULE._resolve_article_id({}, human_dir)

    assert article_id == "001"


def test_resolve_article_id_uses_next_available_if_no_stub(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "001_human.txt").write_text("filled", encoding="utf-8")
    (human_dir / "003_human_title.txt").write_text("filled", encoding="utf-8")

    article_id = MODULE._resolve_article_id({}, human_dir)

    assert article_id == "004"


def test_upsert_metadata_row_updates_existing_row(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "id,title,human_source,ai_model,notes\n001,Old,blog,,\n",
        encoding="utf-8",
    )

    MODULE._upsert_metadata_row(
        metadata, {"id": "001", "title": "New", "source_type": "editoriale"}
    )

    text = metadata.read_text(encoding="utf-8")
    assert "001,New,editoriale,," in text
    assert "001,Old,blog,," not in text
