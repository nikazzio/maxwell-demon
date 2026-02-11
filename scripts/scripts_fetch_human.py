"""Download human articles into a dataset from URL lists (.json or .txt)."""

import argparse
import csv
import json
import re
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch human articles into a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/")
    parser.add_argument(
        "--urls",
        required=True,
        help="Path to URL list (.json list of objects or .txt with one URL per line)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=800,
        help="Minimum word count to keep an article",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download retries per URL",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between retries",
    )
    parser.add_argument(
        "--fail-log",
        default=None,
        help="Path to failure log file (default: data/<dataset>/fetch_failures.log)",
    )
    parser.add_argument(
        "--overwrite-existing-id",
        action="store_true",
        help="Overwrite existing non-empty <id>_human.txt when --id is provided or resolved",
    )
    parser.add_argument(
        "--only-id",
        default=None,
        help="Process only the URL entry targeting this dataset ID (e.g. 012)",
    )
    parser.add_argument(
        "--only-file",
        default=None,
        help="Process only one target file (e.g. 012_human.txt), mapped to its ID",
    )
    return parser.parse_args()


def _load_urls_from_json(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("URLs JSON must be a list of objects")
    return data


def _load_urls_from_txt(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            items.append({"url": raw})
    return items


def _load_urls(path: Path) -> list[dict[str, str]]:
    ext = path.suffix.lower()
    if ext == ".json":
        return _load_urls_from_json(path)
    if ext == ".txt":
        return _load_urls_from_txt(path)
    raise ValueError("Unsupported URL file format. Use .json or .txt")


def _parse_human_id(path: Path) -> str | None:
    match = re.match(r"^(\d+)_human(?:_.+)?\.txt$", path.name)
    if not match:
        return None
    return match.group(1).zfill(3)


def _human_id_set(human_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in human_dir.glob("*.txt"):
        parsed = _parse_human_id(path)
        if parsed is not None:
            ids.add(parsed)
    return ids


def _find_empty_stub_ids(human_dir: Path) -> list[str]:
    empty_ids: list[str] = []
    for path in sorted(human_dir.glob("*_human.txt")):
        parsed = _parse_human_id(path)
        if parsed is None:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            empty_ids.append(parsed)
    return empty_ids


def _normalize_item_id(raw: str) -> str:
    text = raw.strip()
    if not text.isdigit():
        raise ValueError(f"Invalid id '{raw}'. Expected numeric value.")
    return text.zfill(3)


def _resolve_only_id(only_id: str | None, only_file: str | None) -> str | None:
    if only_id and only_file:
        raise ValueError("Use only one selector: --only-id or --only-file")
    if only_id:
        return _normalize_item_id(only_id)
    if not only_file:
        return None
    parsed = _parse_human_id(Path(only_file))
    if parsed is None:
        raise ValueError("--only-file must look like NNN_human.txt")
    return parsed


def _next_available_id(human_dir: Path) -> str:
    existing = _human_id_set(human_dir)
    if not existing:
        return "001"
    max_id = max(int(x) for x in existing)
    return f"{max_id + 1:03d}"


def _resolve_article_id(item: dict[str, str], human_dir: Path) -> str:
    raw_item_id = item.get("id")
    if raw_item_id:
        return _normalize_item_id(raw_item_id)
    empty_stub_ids = _find_empty_stub_ids(human_dir)
    if empty_stub_ids:
        return empty_stub_ids[0]
    return _next_available_id(human_dir)


def _filter_urls_for_target_id(urls: list[dict[str, str]], target_id: str) -> list[dict[str, str]]:
    matching = []
    for item in urls:
        raw_item_id = item.get("id")
        if not raw_item_id:
            continue
        try:
            normalized = _normalize_item_id(raw_item_id)
        except ValueError:
            continue
        if normalized == target_id:
            matching.append(item)
    if matching:
        return matching

    # TXT lists usually do not have explicit IDs: map dataset ID -> 1-based URL index.
    index = int(target_id) - 1
    if 0 <= index < len(urls):
        return [{**urls[index], "id": target_id}]
    raise SystemExit(
        f"No URL entries with id={target_id}. "
        "Provide IDs in JSON or ensure the TXT list has enough URL rows."
    )


def _upsert_metadata_row(path: Path, row: dict[str, str]) -> None:
    header = ["id", "title", "human_source", "ai_model", "notes"]
    target_id = row.get("id", "").strip()
    if not target_id:
        return

    rows: list[dict[str, str]] = []
    if not path.exists():
        path.write_text(",".join(header) + "\n", encoding="utf-8")
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                rows.append(
                    {
                        "id": old.get("id", "").strip(),
                        "title": old.get("title", "").strip(),
                        "human_source": old.get("human_source", "").strip(),
                        "ai_model": old.get("ai_model", "").strip(),
                        "notes": old.get("notes", "").strip(),
                    }
                )

    updated = False
    for old in rows:
        if old["id"] == target_id:
            old["title"] = row.get("title", old["title"]).strip()
            old["human_source"] = row.get("source_type", old["human_source"]).strip()
            updated = True
            break

    if not updated:
        rows.append(
            {
                "id": target_id,
                "title": row.get("title", "").strip(),
                "human_source": row.get("source_type", "").strip(),
                "ai_model": "",
                "notes": "",
            }
        )

    rows.sort(key=lambda r: (int(r["id"]) if r["id"].isdigit() else 10**9, r["id"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    try:
        from newspaper import Article
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for article parsing. Install with: "
            "pip install lxml_html_clean (or pip install -e .[dev])."
        ) from exc

    dataset_root = Path("data") / args.dataset
    human_dir = dataset_root / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "metadata.csv"
    fail_log = Path(args.fail_log) if args.fail_log else dataset_root / "fetch_failures.log"

    urls = _load_urls(Path(args.urls))
    target_id = _resolve_only_id(args.only_id, args.only_file)
    if target_id is not None:
        urls = _filter_urls_for_target_id(urls, target_id)

    for item in urls:
        url = item.get("url")
        if not url:
            print("Skipping entry without url")
            continue

        article = Article(url)
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                article.download()
                article.parse()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(args.retry_delay)
        if last_error is not None:
            msg = f"Error downloading {url}: {last_error}"
            print(msg)
            fail_log.parent.mkdir(parents=True, exist_ok=True)
            with fail_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            continue

        if len(article.text.split()) < args.min_words:
            print(f"Skipped (too short): {article.title}")
            continue

        article_id = _resolve_article_id(item, human_dir)
        title = item.get("title") or article.title or "untitled"
        file_path = human_dir / f"{article_id}_human.txt"
        if file_path.exists() and file_path.read_text(encoding="utf-8").strip():
            if not args.overwrite_existing_id:
                print(
                    f"Skipping ID {article_id}: {file_path.name} already has content "
                    "(use --overwrite-existing-id to replace)."
                )
                continue
            print(f"Overwriting existing content for ID {article_id}")
        file_path.write_text(article.text, encoding="utf-8")

        _upsert_metadata_row(
            metadata_path,
            {
                "id": article_id,
                "title": title,
                "source_type": item.get("source_type", ""),
            },
        )

        print(f"Downloaded: {title}")


if __name__ == "__main__":
    main()
