"""Generate AI shadow texts from human dataset files using OpenAI APIs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib


DEFAULT_SHADOW_CONFIG: dict[str, object] = {
    "model": "gpt-4.1-mini",
    "temperature": 0.8,
    "incipit_chars": 100,
    "max_output_tokens": 1800,
    "system_prompt": (
        "Agisci come un saggista esperto. Scrivi saggi lunghi, complessi, "
        "senza elenchi puntati, ricchi di subordinate e vocabolario vario."
    ),
    "user_prompt_template": (
        "Scrivi un saggio originale di circa 1000 parole basato su questo titolo: "
        "'{TITLE}'. Usa questo incipit come ispirazione per il tono: '{INCIPIT}'."
    ),
}

DEFAULT_OPENAI_CONFIG: dict[str, object] = {
    "api_key_env": "OPENAI_API_KEY",
    "api_key": "",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI shadow dataset from human texts")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/")
    parser.add_argument(
        "--config",
        default="config.local.toml",
        help="TOML config path (default: config.local.toml)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite non-empty output files in data/<dataset>/ai",
    )
    parser.add_argument(
        "--fail-log",
        default=None,
        help="Failure log path (default: data/<dataset>/shadow_failures.log)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without calling the API or writing files",
    )
    parser.add_argument(
        "--only-id",
        default=None,
        help="Process only one dataset ID from human files (e.g. 012)",
    )
    parser.add_argument(
        "--only-file",
        default=None,
        help="Process only one human filename (e.g. 012_human.txt)",
    )
    return parser.parse_args()


def _load_toml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_shadow_config(path: Path) -> dict[str, object]:
    raw = _load_toml(path)
    shadow = raw.get("shadow_dataset", {}) if isinstance(raw, dict) else {}
    openai_cfg = raw.get("openai", {}) if isinstance(raw, dict) else {}

    if not isinstance(shadow, dict):
        raise ValueError("[shadow_dataset] must be a TOML table")
    if not isinstance(openai_cfg, dict):
        raise ValueError("[openai] must be a TOML table")

    cfg: dict[str, object] = {
        "shadow_dataset": {**DEFAULT_SHADOW_CONFIG, **shadow},
        "openai": {**DEFAULT_OPENAI_CONFIG, **openai_cfg},
    }
    _validate_shadow_config(cfg)
    return cfg


def _validate_shadow_config(cfg: dict[str, object]) -> None:
    shadow = cfg["shadow_dataset"]
    model = shadow["model"]
    temperature = shadow["temperature"]
    incipit_chars = shadow["incipit_chars"]
    max_output_tokens = shadow["max_output_tokens"]
    system_prompt = shadow["system_prompt"]
    user_prompt_template = shadow["user_prompt_template"]

    if not isinstance(model, str) or not model.strip():
        raise ValueError("shadow_dataset.model must be a non-empty string")
    if not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("shadow_dataset.temperature must be between 0 and 2")
    if not isinstance(incipit_chars, int) or incipit_chars <= 0:
        raise ValueError("shadow_dataset.incipit_chars must be a positive integer")
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise ValueError("shadow_dataset.max_output_tokens must be a positive integer")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("shadow_dataset.system_prompt must be a non-empty string")
    if not isinstance(user_prompt_template, str) or not user_prompt_template.strip():
        raise ValueError("shadow_dataset.user_prompt_template must be a non-empty string")

    openai_cfg = cfg["openai"]
    api_key_env = openai_cfg["api_key_env"]
    api_key = openai_cfg["api_key"]
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("openai.api_key_env must be a non-empty string")
    if not isinstance(api_key, str):
        raise ValueError("openai.api_key must be a string")


def _resolve_api_key(cfg: dict[str, object]) -> str:
    openai_cfg = cfg["openai"]
    env_name = str(openai_cfg["api_key_env"]).strip()
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env

    from_config = str(openai_cfg.get("api_key", "")).strip()
    if from_config:
        return from_config

    raise SystemExit(
        f"OpenAI API key not found. Set env var {env_name} or openai.api_key in local config."
    )


def _human_to_ai_filename(human_file: Path) -> str:
    stem = human_file.stem
    if stem.endswith("_human"):
        return f"{stem[:-6]}_ai.txt"
    if "_human_" in stem:
        return f"{stem.replace('_human_', '_ai_', 1)}.txt"
    return f"{stem}_ai.txt"


def _extract_dataset_id_from_human_file(path: Path) -> str | None:
    match = re.match(r"^(\d+)_human(?:_.+)?\.txt$", path.name)
    if not match:
        return None
    return match.group(1).zfill(3)


def _resolve_only_id(only_id: str | None, only_file: str | None) -> str | None:
    if only_id and only_file:
        raise ValueError("Use only one selector: --only-id or --only-file")
    if only_id:
        text = only_id.strip()
        if not text.isdigit():
            raise ValueError("--only-id must be numeric")
        return text.zfill(3)
    if not only_file:
        return None
    parsed = _extract_dataset_id_from_human_file(Path(only_file))
    if parsed is None:
        raise ValueError("--only-file must look like NNN_human.txt")
    return parsed


def _select_human_files(
    files: list[Path], only_id: str | None, only_file: str | None
) -> list[Path]:
    target_id = _resolve_only_id(only_id, only_file)
    if target_id is None:
        return files
    selected = [p for p in files if _extract_dataset_id_from_human_file(p) == target_id]
    if not selected:
        raise SystemExit(f"No human file found for ID {target_id}")
    return selected


def _is_non_empty(path: Path) -> bool:
    return path.exists() and bool(path.read_text(encoding="utf-8").strip())


def _build_title(human_file: Path) -> str:
    return human_file.stem.replace("_", " ").strip()


def _build_incipit(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    return normalized[:max_chars]


def _extract_output_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            maybe_text = getattr(part, "text", None)
            if isinstance(maybe_text, str) and maybe_text.strip():
                chunks.append(maybe_text.strip())
    return "\n".join(chunks).strip()


def _is_temperature_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "temperature" in message and "not supported" in message


def _create_response(
    *,
    client: object,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system_prompt: str,
    user_prompt: str,
    include_temperature: bool,
) -> object:
    payload: dict[str, object] = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if include_temperature:
        payload["temperature"] = temperature
    return client.responses.create(**payload)


def _log_failure(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def main() -> None:
    args = _parse_args()

    cfg = _load_shadow_config(Path(args.config))
    api_key = _resolve_api_key(cfg)

    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Missing OpenAI SDK. Install dependencies with: pip install -e .[dev]"
            ) from exc
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise SystemExit(
                "Missing tqdm dependency. Install dependencies with: pip install -e .[dev]"
            ) from exc
    else:
        # Keep dry-run usable even without external dependencies.
        tqdm = lambda x, **_: x  # noqa: E731
        OpenAI = None  # type: ignore[assignment]

    dataset_root = Path("data") / args.dataset
    human_dir = dataset_root / "human"
    ai_dir = dataset_root / "ai"
    if not human_dir.is_dir():
        raise SystemExit(f"Missing input directory: {human_dir}")
    ai_dir.mkdir(parents=True, exist_ok=True)

    fail_log = Path(args.fail_log) if args.fail_log else dataset_root / "shadow_failures.log"

    human_files = sorted(human_dir.glob("*.txt"))
    if not human_files:
        raise SystemExit(f"No .txt files found in {human_dir}")
    human_files = _select_human_files(human_files, args.only_id, args.only_file)

    shadow_cfg = cfg["shadow_dataset"]
    model = str(shadow_cfg["model"])
    temperature = float(shadow_cfg["temperature"])
    max_output_tokens = int(shadow_cfg["max_output_tokens"])
    incipit_chars = int(shadow_cfg["incipit_chars"])
    system_prompt = str(shadow_cfg["system_prompt"])
    user_prompt_template = str(shadow_cfg["user_prompt_template"])

    client = OpenAI(api_key=api_key) if OpenAI is not None else None

    generated = 0
    skipped_existing = 0
    skipped_empty_human = 0
    failed = 0

    iterator = tqdm(human_files, desc="Generating shadow dataset", unit="file")
    for human_file in iterator:
        human_text = human_file.read_text(encoding="utf-8").strip()
        if not human_text:
            skipped_empty_human += 1
            continue

        ai_file = ai_dir / _human_to_ai_filename(human_file)
        if _is_non_empty(ai_file) and not args.overwrite_existing:
            skipped_existing += 1
            continue

        title = _build_title(human_file)
        incipit = _build_incipit(human_text, incipit_chars)
        user_prompt = user_prompt_template.format(TITLE=title, INCIPIT=incipit)

        if args.dry_run:
            print(f"[DRY RUN] Would generate: {ai_file}")
            generated += 1
            continue

        assert client is not None
        try:
            response = _create_response(
                client=client,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                include_temperature=True,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_temperature_unsupported_error(exc):
                failed += 1
                message = f"{human_file.name}\t{ai_file.name}\t{exc}"
                print(f"Error: {message}")
                _log_failure(fail_log, message)
                continue
            try:
                response = _create_response(
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    include_temperature=False,
                )
            except Exception as retry_exc:  # noqa: BLE001
                failed += 1
                message = f"{human_file.name}\t{ai_file.name}\t{retry_exc}"
                print(f"Error: {message}")
                _log_failure(fail_log, message)
                continue
        try:
            output_text = _extract_output_text(response)
            if not output_text:
                raise RuntimeError("OpenAI response did not include output text")
            ai_file.write_text(output_text.strip() + "\n", encoding="utf-8")
            generated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            message = f"{human_file.name}\t{ai_file.name}\t{exc}"
            print(f"Error: {message}")
            _log_failure(fail_log, message)

    print("\nShadow dataset generation completed")
    print(f"Generated: {generated}")
    print(f"Skipped existing AI files: {skipped_existing}")
    print(f"Skipped empty human files: {skipped_empty_human}")
    print(f"Failures: {failed}")
    if failed:
        print(f"Failure log: {fail_log}")


if __name__ == "__main__":
    main()
