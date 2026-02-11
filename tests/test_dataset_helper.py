from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scripts_dataset.py"
SPEC = spec_from_file_location("scripts_dataset", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load scripts_dataset.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_collect_ids_accepts_only_canonical_names(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "001_human.txt").write_text("a", encoding="utf-8")
    (human_dir / "002_human_Title.txt").write_text("b", encoding="utf-8")

    ids = MODULE._collect_ids(human_dir, "human")

    assert ids == {"001"}


def test_count_empty_stubs_counts_only_canonical_stub_files(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "001_human.txt").write_text("", encoding="utf-8")
    (human_dir / "002_human.txt").write_text("filled", encoding="utf-8")
    (human_dir / "003_human_Title.txt").write_text("", encoding="utf-8")

    count = MODULE._count_empty_stubs(human_dir, "human")

    assert count == 1


def test_count_legacy_files_counts_suffix_names(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "001_human.txt").write_text("a", encoding="utf-8")
    (human_dir / "002_human_Title.txt").write_text("b", encoding="utf-8")

    count = MODULE._count_legacy_files(human_dir, "human")

    assert count == 1
