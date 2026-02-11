"""Download human articles into a dataset from a JSON URL list."""

import argparse
import json
from pathlib import Path
import time

from newspaper import Article


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch human articles into a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/")
    parser.add_argument("--urls", required=True, help="Path to JSON list of URLs")
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
    return parser.parse_args()


def _load_urls(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("URLs JSON must be a list of objects")
    return data


def _safe_title(text: str) -> str:
    cleaned = "".join(c for c in text if c.isalnum() or c == " ").strip()
    return cleaned.replace(" ", "_")


def _write_metadata_row(path: Path, row: dict[str, str]) -> None:
    if not path.exists():
        path.write_text("id,title,human_source,ai_model,notes\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"{row.get('id','')},{row.get('title','')},{row.get('source_type','')},,\n"
        )


def main() -> None:
    args = _parse_args()
    dataset_root = Path("data") / args.dataset
    human_dir = dataset_root / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "metadata.csv"
    fail_log = Path(args.fail_log) if args.fail_log else dataset_root / "fetch_failures.log"

    urls = _load_urls(Path(args.urls))

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

        article_id = item.get("id")
        if not article_id:
            article_id = f"{len(list(human_dir.glob('*_human.txt'))) + 1:03d}"

        title = item.get("title") or article.title or "untitled"
        safe_title = _safe_title(title)
        suffix = f"_{safe_title}" if safe_title else ""
        filename = f"{article_id}_human{suffix}.txt"
        file_path = human_dir / filename
        file_path.write_text(article.text, encoding="utf-8")

        _write_metadata_row(
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
