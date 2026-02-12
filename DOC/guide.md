# Operational Protocol

This document specifies the recommended end-to-end execution protocol for Maxwell-Demon experiments.

## 1. Environment and Reproducibility Preconditions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Ensure all runs are executed from the same environment to avoid dependency-induced drift.

## 2. Dataset Validation Stage

Objective: obtain a class-balanced corpus with deterministic human/AI pairing.

Expected structure:

```text
data/<dataset>/
  human/
  ai/
  metadata.csv
```

Reference commands:

```bash
python scripts/scripts_dataset.py init --name dataset_it_01 --count 50
python scripts/scripts_dataset.py check --name dataset_it_01
```

Acceptance criteria:

- exactly 50 files matching `NNN_human.txt`
- exactly 50 files matching `NNN_ai.txt`
- one-to-one ID correspondence across classes

Any pairing inconsistency invalidates downstream comparative analysis.

## 3. Reference Calibration Stage

Objective: estimate two token-probability dictionaries under identical preprocessing and smoothing assumptions.

Tokenization constraint:

- calibration and inference must share the same tokenization policy from `[tokenization]` in `config.toml`;
- default protocol uses `tiktoken` (`cl100k_base`, punctuation included);
- `legacy` mode is available for backward compatibility experiments.

Local synthetic source:

```bash
python scripts/prepare_resources.py \
  --synthetic-input data/dataset_it_01/ai \
  --config config.example.toml
```

Remote synthetic source:

```bash
python scripts/prepare_resources.py \
  --synthetic-url https://example.com/synthetic_corpus.txt.gz \
  --config config.example.toml
```

Mandatory artifacts:

- `data/reference/paisa_ref_dict.json`
- `data/reference/synthetic_ref_dict.json`

Without both references, the dual-attractor hypothesis is not operationally testable.

## 4. Tournament Inference Stage

Objective: compute window-level differential entropy and export the consolidated score table.

```bash
maxwell-demon-tournament \
  --human-input data/dataset_it_01/human \
  --ai-input data/dataset_it_01/ai \
  --config config.example.toml
```

Default artifact:

- `results/dataset_it_01/data/final_delta.csv`
- `results/dataset_it_01/data/final_delta.md` (automatic report generated after CSV export)

## 5. Phase-Space Analysis Stage

Objective: evaluate separability in the joint space of entropy differential and surprisal variance.

```bash
maxwell-demon-phase \
  --input results/dataset_it_01/data \
  --config config.example.toml
```

Default artifact:

- `results/dataset_it_01/plot/phase_delta_h_vs_burstiness_paisa.html`

## 5.1 Standalone Reporting (optional)

To regenerate a report from an existing CSV without rerunning the tournament:

```bash
maxwell-demon-report \
  --input results/dataset_it_01/data/final_delta.csv \
  --output results/dataset_it_01/data/final_delta.md
```

## 6. Interpretation Heuristics

- `delta_h > 0`: increased compatibility with human-reference statistics.
- `delta_h \approx 0` or `delta_h < 0`: increased compatibility with synthetic-reference statistics.
- large `burstiness_paisa`: elevated local surprisal heterogeneity under the human model.
- small `burstiness_paisa`: smoother surprisal profile under the human model.

Inference should be distributional (across windows and files), not pointwise.

## 7. Recommended Minimal Execution Trace

| Stage | Command | Expected endpoint |
|---|---|---|
| validation | `scripts_dataset.py check` | paired and balanced dataset |
| calibration | `prepare_resources.py` | both reference dictionaries persisted |
| inference | `maxwell-demon-tournament` | `final_delta.csv` + `final_delta.md` generated |
| visualization | `maxwell-demon-phase` | phase-space map rendered |

## 8. Practical Notes

- Operational compression baseline is `lzma`.
- Outputs are dataset-scoped under `results/<dataset>/data` and `results/<dataset>/plot`.
- Explicit output redirection remains available through `--output` flags.
