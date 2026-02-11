"""Dataset helper utilities for Maxwell-Demon."""

import argparse
import csv
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Dataset helper for Maxwell-Demon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Create a dataset skeleton")
    init_p.add_argument("--name", required=True, help="Dataset folder name")
    init_p.add_argument("--count", type=int, default=50, help="Number of pairs to scaffold")

    check_p = sub.add_parser("check", help="Validate a dataset structure and pairing")
    check_p.add_argument("--name", required=True, help="Dataset folder name")

    sub.add_parser("audit", help="Audit all datasets under data/")

    return parser.parse_args()


def _dataset_root(name: str) -> Path:
    """Return dataset root path under data/."""
    return Path("data") / name


def init_dataset(name: str, count: int) -> None:
    """Create a dataset skeleton with human/ai folders and metadata."""
    root = _dataset_root(name)
    human = root / "human"
    ai = root / "ai"
    human.mkdir(parents=True, exist_ok=True)
    ai.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        idx = f"{i:03d}"
        (human / f"{idx}_human.txt").touch(exist_ok=True)
        (ai / f"{idx}_ai.txt").touch(exist_ok=True)

    metadata = root / "metadata.csv"
    if not metadata.exists():
        metadata.write_text("id,title,human_source,ai_model,notes\n", encoding="utf-8")

    print(f"Created dataset skeleton at {root} with {count} pairs")


def _read_metadata_ids(path: Path) -> set[str]:
    """Read IDs from metadata.csv (if present)."""
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        ids = set()
        for row in reader:
            if "id" in row and row["id"]:
                ids.add(row["id"].strip())
        return ids


def check_dataset(name: str) -> None:
    """Validate a dataset's file pairing and metadata coverage."""
    root = _dataset_root(name)
    human = root / "human"
    ai = root / "ai"

    if not root.exists():
        raise SystemExit(f"Dataset not found: {root}")
    if not human.is_dir() or not ai.is_dir():
        raise SystemExit("Missing human/ or ai/ directories")

    human_ids = {p.stem.replace("_human", "") for p in human.glob("*_human.txt")}
    ai_ids = {p.stem.replace("_ai", "") for p in ai.glob("*_ai.txt")}

    missing_ai = sorted(human_ids - ai_ids)
    missing_human = sorted(ai_ids - human_ids)

    if missing_ai:
        print("Missing AI files for IDs:")
        for mid in missing_ai:
            print(f"  {mid}")
    if missing_human:
        print("Missing human files for IDs:")
        for mid in missing_human:
            print(f"  {mid}")

    total_pairs = len(human_ids & ai_ids)
    print(f"Found {total_pairs} paired IDs")

    metadata_path = root / "metadata.csv"
    metadata_ids = _read_metadata_ids(metadata_path)
    if metadata_ids:
        missing_meta = sorted((human_ids & ai_ids) - metadata_ids)
        if missing_meta:
            print("Missing metadata rows for IDs:")
            for mid in missing_meta:
                print(f"  {mid}")


def audit_all() -> None:
    """Validate all datasets under data/."""
    data_root = Path("data")
    if not data_root.exists():
        raise SystemExit("data/ directory not found")

    datasets = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not datasets:
        raise SystemExit("No datasets found under data/")

    for ds in datasets:
        print(f"== {ds.name} ==")
        check_dataset(ds.name)


def main() -> None:
    """Dispatch CLI subcommands."""
    args = _parse_args()
    if args.cmd == "init":
        init_dataset(args.name, args.count)
    elif args.cmd == "check":
        check_dataset(args.name)
    elif args.cmd == "audit":
        audit_all()


if __name__ == "__main__":
    main()
