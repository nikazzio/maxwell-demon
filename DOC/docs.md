# CLI and Configuration Specification

## 1. Configuration Schema (TOML)

Reference template: `config.example.toml`.

| Section | Formal scope |
|---|---|
| `[analysis]` | Sliding-window inference parameters (`window`, `step`, `log_base`) |
| `[compression]` | Compression backend selection (`algorithm`) |
| `[tokenization]` | Tokenization strategy and encoder settings |
| `[reference]` | Reference-corpus sources and dictionary paths |
| `[output]` | Dataset-templated output directories (`data_dir`, `plot_dir`) |
| `[openai]` | API integration settings |
| `[shadow_dataset]` | Synthetic generation defaults |

### 1.1 `[tokenization]` semantics

| Key | Definition |
|---|---|
| `method` | Tokenization backend. Supported values: `tiktoken`, `legacy`. |
| `encoding_name` | Encoder name used by `tiktoken` mode (default: `cl100k_base`). |
| `include_punctuation` | If `true`, punctuation is retained in tokenization. In `legacy` mode this flag is ignored. |
| `fallback_to_legacy_if_tiktoken_missing` | If `true`, missing `tiktoken` dependency triggers automatic legacy fallback with warning; if `false`, execution fails. |

Notes:

- `tiktoken` mode returns token strings decoded from token ids, preserving `list[str]` compatibility with entropy metrics.
- `legacy` mode preserves historical behavior (lowercasing + regex punctuation removal + whitespace split).
- Runtime/config default is `method = "tiktoken"`.
- Fallback behavior is explicit and configurable via `fallback_to_legacy_if_tiktoken_missing`.

### 1.2 `[reference]` semantics

| Key | Definition |
|---|---|
| `paisa_path` | Target path for the human reference dictionary JSON |
| `synthetic_path` | Target path for the synthetic reference dictionary JSON |
| `paisa_url` | Remote source URI for the PAISA corpus |
| `paisa_corpus_path` | Local cache path for PAISA text |
| `synthetic_url` | Remote source URI for synthetic corpus text |
| `synthetic_corpus_path` | Local cache path for synthetic text |
| `smoothing_k` | Global add-k smoothing coefficient applied symmetrically to both dictionaries (recommended: `1.0`) |

### 1.3 `[output]` semantics

| Key | Definition |
|---|---|
| `data_dir` | CSV output template, typically `results/{dataset}/data` |
| `plot_dir` | Plot output template, typically `results/{dataset}/plot` |

`{dataset}` is inferred from input path topology when available.

## 2. Canonical Execution Order

1. `scripts/prepare_resources.py`
2. `maxwell-demon-tournament`
3. `maxwell-demon-phase`

## 3. `prepare_resources.py`

Reference command:

```bash
python scripts/prepare_resources.py \
  --synthetic-input data/<dataset>/ai \
  --config config.example.toml
```

### 3.1 Arguments

| Argument | Requirement | Semantics |
|---|---|---|
| `--synthetic-input` | conditional | Local synthetic corpus source (`.txt` file or directory). Mutually exclusive with `--synthetic-url`. |
| `--synthetic-url` | conditional | Remote synthetic corpus source. Mutually exclusive with `--synthetic-input`. |
| `--synthetic-corpus-out` | optional | Overrides cached synthetic corpus path. |
| `--only-human` | optional | Build only the human reference dictionary (synthetic source not required). |
| `--only-synthetic` | optional | Build only the synthetic reference dictionary. |
| `--config` | optional | Path to TOML configuration. |
| `--paisa-url` | optional | Overrides `reference.paisa_url`. |
| `--paisa-corpus-out` | optional | Overrides cached PAISA corpus path. |
| `--human-dict-out` | optional | Overrides `reference.paisa_path`. |
| `--synthetic-dict-out` | optional | Overrides `reference.synthetic_path`. |
| `--smoothing-k` | optional | CLI override of smoothing coefficient. If omitted, `reference.smoothing_k` is used. |
| `--skip-download` | optional | Enforces local-cache usage for remote corpus sources. |

### 3.2 Outputs

- `data/reference/paisa_ref_dict.json`
- `data/reference/synthetic_ref_dict.json`

Human-only mode:

```bash
python scripts/prepare_resources.py --only-human --config config.example.toml
```

In this mode, only `reference.paisa_path` is produced.

Tokenization coherence guarantee:

- `prepare_resources.py` uses the same `[tokenization]` configuration used later by analysis CLIs.
- This prevents dictionary/analysis token-space mismatch.

## 4. `maxwell-demon-tournament`

Reference command:

```bash
maxwell-demon-tournament \
  --human-input data/<dataset>/human \
  --ai-input data/<dataset>/ai \
  --config config.example.toml
```

### 4.1 Arguments

| Argument | Requirement | Semantics |
|---|---|---|
| `--human-input` | required | Human corpus input (`.txt` file or directory). |
| `--ai-input` | required | Synthetic corpus input (`.txt` file or directory). |
| `--config` | optional | Path to TOML configuration. |
| `--output` | optional | Explicit tournament CSV output path. |
| `--window` | optional | Overrides `analysis.window`. |
| `--step` | optional | Overrides `analysis.step`. |
| `--log-base` | optional | Overrides `analysis.log_base`. |
| `--compression` | optional | Overrides compression backend. |

Tokenization policy:

- No dedicated CLI flag is required; tournament tokenization is resolved from `[tokenization]` in TOML.

### 4.2 Output artifact

- `results/<dataset>/data/final_delta.csv`
- `results/<dataset>/data/final_delta.md` (auto-generated statistical report)

### 4.3 Output schema

| Column | Statistical meaning |
|---|---|
| `filename` | Source text identifier |
| `window_id` | Sliding-window ordinal index |
| `label` | Ground-truth class (`human`, `ai`) |
| `delta_h` | Differential entropy statistic: `H_human_ref - H_synthetic_ref` |
| `burstiness_paisa` | Surprisal variance under the human reference |

## 5. `maxwell-demon-phase`

Reference command:

```bash
maxwell-demon-phase \
  --input results/<dataset>/data \
  --config config.example.toml
```

### 5.1 Arguments

| Argument | Requirement | Semantics |
|---|---|---|
| `--input` | required | Single CSV file or directory of CSV files. |
| `--config` | optional | Path to TOML configuration. |
| `--output` | optional | Explicit HTML output path. |
| `--x` | optional | X-axis variable. |
| `--y` | optional | Y-axis variable. |
| `--color` | optional | Grouping variable for color encoding. |
| `--facet` | optional | Optional faceting variable. |
| `--density` | optional | Forces density contour mode. |
| `--density-threshold` | optional | Row-count threshold for density auto-switch. |

Tournament-aware default axes:

- `x = delta_h`
- `y = burstiness_paisa`

Default artifact:

- `results/<dataset>/plot/phase_delta_h_vs_burstiness_paisa.html`

## 6. `maxwell-demon` (single-run diagnostic mode)

This interface provides local metric inspection (`raw`/`diff`) and does not substitute tournament inference.

| Argument | Requirement | Semantics |
|---|---|---|
| `--input` | required | Input `.txt` file or directory. |
| `--mode` | optional | `raw` or `diff`. |
| `--window` | optional | Window length in tokens. |
| `--step` | optional | Window shift in tokens. |
| `--output` | optional | Aggregated CSV output path. |
| `--output-dir` | optional | Per-file CSV output directory. |
| `--label` | optional | Optional label annotation (`human`, `ai`). |
| `--reference` | optional | Reference selector for `diff` mode (`paisa`, `synthetic`). |
| `--ref-dict` | optional | Explicit reference dictionary JSON path. |
| `--log-base` | optional | Logarithm base for entropy/surprisal. |
| `--compression` | optional | Compression backend. |
| `--config` | optional | Path to TOML configuration. |

Tokenization policy:

- Single analysis uses `[tokenization]` from the loaded config.

## 7. Compression Policy

Protocol default: **LZMA**.

`gzip`, `bz2`, and `zlib` are maintained for controlled comparative analyses and reproducibility studies.

## 8. Automatic and Standalone Reporting

`maxwell-demon-tournament` generates a Markdown report immediately after CSV export.
The report path is derived from the CSV output path by replacing the extension with `.md`.

Example:

- CSV: `results/dataset_it_01/data/final_delta.csv`
- report: `results/dataset_it_01/data/final_delta.md`

### 8.1 Report content

The report includes:

- descriptive statistics on `delta_h` (`count`, `mean`, `median`, `std`, `min`, `max`), grouped by `label` if available;
- rule-based classification metrics (current implementation in `maxwell-demon-report`: `delta_h < 0 => human`);
- confusion matrix (`TP`, `TN`, `FP`, `FN`) when valid labels are present.

### 8.2 Standalone CLI

The same report can be generated manually from any existing CSV:

```bash
maxwell-demon-report --input <tournament.csv> --output <report.md>
```
