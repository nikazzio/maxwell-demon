# Maxwell-Demon

![License](https://img.shields.io/github/license/nikazzio/maxwell-demon) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![CI](https://img.shields.io/github/actions/workflow/status/nikazzio/maxwell-demon/ci.yml?branch=main) ![Coverage](https://img.shields.io/codecov/c/github/nikazzio/maxwell-demon)

Maxwell-Demon is a Python package and CLI toolkit for binary discrimination between human-authored and machine-generated text via a dual-reference entropy protocol.

The primary decision statistic is the window-wise entropy differential:

$$
\Delta H(T, W) = H_{\text{Human Ref}}(T, W) - H_{\text{Synthetic Ref}}(T, W)
$$

where each entropy term is the mean token surprisal under a calibrated reference distribution.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Core Pipeline

| Command | Role in the protocol |
|---|---|
| `python scripts/prepare_resources.py` | Reference calibration (human + synthetic dictionaries) |
| `maxwell-demon-tournament` | Dual-reference scoring and delta extraction |
| `maxwell-demon-report` | Standalone Markdown report generation from a tournament CSV |
| `maxwell-demon-phase` | Phase-space rendering (`delta_h`, `burstiness_paisa`) |

## Quickstart (Operational)

This is the shortest end-to-end path from raw files to interpretable outputs.

### 0. Prepare a paired dataset

Expected structure:

```text
data/<dataset>/
  human/
    001_human.txt
  ai/
    001_ai.txt
```

### 1. Build reference dictionaries

```bash
python scripts/prepare_resources.py \
  --synthetic-input data/<dataset>/ai \
  --config config.example.toml
```

Required outputs:

- `data/reference/paisa_ref_dict.json`
- `data/reference/synthetic_ref_dict.json`

### 2. Run dual-reference tournament

```bash
maxwell-demon-tournament \
  --human-input data/<dataset>/human \
  --ai-input data/<dataset>/ai \
  --config config.example.toml
```

Default outputs:

- `results/<dataset>/data/final_delta.csv`
- `results/<dataset>/data/final_delta.md`

### 3. Render phase-space plot

```bash
maxwell-demon-phase \
  --input results/<dataset>/data \
  --config config.example.toml
```

Default output:

- `results/<dataset>/plot/phase_delta_h_vs_burstiness_paisa.html`

## Output Interpretation

Core columns in tournament CSV:

- `delta_h = H_human_ref - H_synthetic_ref`
- `burstiness_paisa = Var(-log P_human_ref(token))`
- `label` (if present): expected class (`human`/`ai`)

Practical reading:

- lower `delta_h` means lower surprisal under human reference relative to synthetic reference;
- higher `delta_h` means lower surprisal under synthetic reference relative to human reference;
- higher `burstiness_paisa` means stronger local surprisal fluctuation under the human model.
- default decision rule (if a hard threshold is needed): `delta_h < 0 => human`.

Do not interpret single windows in isolation; inspect distributions per file and per class.

## Statistical Caveats

- `delta_h = 0` is a useful default boundary but not universally optimal; calibrate on a validation set.
- Window-level rows are not independent observations from the same document; avoid overconfident significance claims.
- Domain and genre shift can change token distributions and degrade discrimination quality.
- OOV/rare-token behavior depends on smoothing and tokenization settings.
- Keep tokenization and smoothing identical between reference building and runtime analysis.

## Minimal Reproducible Run

### 1. Calibrate References

Using local synthetic text:

```bash
python scripts/prepare_resources.py \
  --synthetic-input data/dataset_it_01/ai \
  --config config.example.toml
```

Using remote synthetic text:

```bash
python scripts/prepare_resources.py \
  --synthetic-url https://example.com/synthetic_corpus.txt.gz \
  --config config.example.toml
```

Human-only fallback (when no synthetic corpus is available):

```bash
python scripts/prepare_resources.py \
  --only-human \
  --config config.example.toml
```

### 2. Execute Tournament

```bash
maxwell-demon-tournament \
  --human-input data/dataset_it_01/human \
  --ai-input data/dataset_it_01/ai \
  --config config.example.toml
```

Default artifact:

- `results/dataset_it_01/data/final_delta.csv`
- `results/dataset_it_01/data/final_delta.md` (auto-generated report)

### 3. Inspect Phase Space

```bash
maxwell-demon-phase \
  --input results/dataset_it_01/data \
  --config config.example.toml
```

Default artifact:

- `results/dataset_it_01/plot/phase_delta_h_vs_burstiness_paisa.html`

## Compression Regime

Protocol default: `lzma`.

Alternative codecs (`gzip`, `bz2`, `zlib`) are available for ablation and sensitivity analyses, but `lzma` is the operational baseline.

## Configuration Model

Canonical template: `config.example.toml`.

Top-level sections:

- `[analysis]`
- `[compression]`
- `[tokenization]`
- `[reference]`
- `[output]`
- `[openai]`
- `[shadow_dataset]`

Tokenization defaults:

- `method = "tiktoken"` (recommended)
- `encoding_name = "cl100k_base"`
- `include_punctuation = true`

Backward-compatible mode is available with `method = "legacy"` (lowercase + regex punctuation stripping).
For statistical consistency, reference-dictionary construction and runtime analysis both use the same tokenization configuration.

Output paths are dataset-aware through templating:

- data: `results/{dataset}/data`
- plots: `results/{dataset}/plot`

## Auxiliary Interfaces

- `maxwell-demon`: single-run diagnostics (`raw`, `diff`)
- `maxwell-demon-plot`: static PNG trajectory plot
- `maxwell-demon-plot-html`: interactive HTML trajectory plot
- `maxwell-demon-report`: standalone Markdown report tool (`--input`, `--output`)
- `scripts/run_analysis.py`: wrapper for single/tournament execution modes

## Verification

```bash
.venv/bin/ruff check .
PYTHONPATH=src .venv/bin/python -m pytest tests
```

## Documentation Map

- `DOC/theoretical_framework.md`
- `DOC/docs.md`
- `DOC/guide.md`

## License

MIT (`LICENSE`).
