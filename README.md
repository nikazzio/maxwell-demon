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
| `maxwell-demon-phase` | Phase-space rendering (`delta_h`, `burstiness_paisa`) |

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

### 2. Execute Tournament

```bash
maxwell-demon-tournament \
  --human-input data/dataset_it_01/human \
  --ai-input data/dataset_it_01/ai \
  --config config.example.toml
```

Default artifact:

- `results/dataset_it_01/data/final_delta.csv`

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
- `[reference]`
- `[output]`
- `[openai]`
- `[shadow_dataset]`

Output paths are dataset-aware through templating:

- data: `results/{dataset}/data`
- plots: `results/{dataset}/plot`

## Auxiliary Interfaces

- `maxwell-demon`: single-run diagnostics (`raw`, `diff`)
- `maxwell-demon-plot`: static PNG trajectory plot
- `maxwell-demon-plot-html`: interactive HTML trajectory plot
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
