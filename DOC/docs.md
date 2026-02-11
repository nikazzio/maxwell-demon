# CLI and Configuration Specification

## 1. Configuration Schema (TOML)

Reference template: `config.example.toml`.

| Section | Formal scope |
|---|---|
| `[analysis]` | Sliding-window inference parameters (`window`, `step`, `log_base`) |
| `[compression]` | Compression backend selection (`algorithm`) |
| `[reference]` | Reference-corpus sources and dictionary paths |
| `[output]` | Dataset-templated output directories (`data_dir`, `plot_dir`) |
| `[openai]` | API integration settings |
| `[shadow_dataset]` | Synthetic generation defaults |

### 1.1 `[reference]` semantics

| Key | Definition |
|---|---|
| `paisa_path` | Target path for the human reference dictionary JSON |
| `synthetic_path` | Target path for the synthetic reference dictionary JSON |
| `paisa_url` | Remote source URI for the PAISA corpus |
| `paisa_corpus_path` | Local cache path for PAISA text |
| `synthetic_url` | Remote source URI for synthetic corpus text |
| `synthetic_corpus_path` | Local cache path for synthetic text |
| `smoothing_k` | Global add-k smoothing coefficient applied symmetrically to both dictionaries (recommended: `1.0`) |

### 1.2 `[output]` semantics

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

### 4.2 Output artifact

- `results/<dataset>/data/final_delta.csv`

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

## 7. Compression Policy

Protocol default: **LZMA**.

`gzip`, `bz2`, and `zlib` are maintained for controlled comparative analyses and reproducibility studies.
