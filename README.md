# Maxwell-Demon

Maxwell-Demon is a CLI tool that analyzes text files to distinguish human-written text from LLM-generated text by measuring local entropy dynamics and compression patterns. It computes windowed Shannon entropy, burstiness (variance of surprisal), and LZ-based compression ratios, exporting a clean CSV ready for plots and phase diagrams.

The core idea: human texts tend to show higher local variability (burstiness) and heavier tails in surprisal distributions, while LLM text often appears smoother and more uniform. This tool gives you measurable signals to test that hypothesis.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development (optional)

```bash
pip install -e '.[dev]'
ruff check .
ruff format .
```

## Tests

```bash
python -m pytest
```

## Usage

```bash
# Raw analysis
maxwell-demon --input data/thesis_human.txt --mode raw --window 50 --step 10 --output results_human.csv --label human --log-base 2

# Differential analysis (with reference dictionary)
maxwell-demon --input data/chatgpt_text.txt --mode diff --ref-dict standard_italian.json --output results_ai.csv --label ai --log-base 2

# Batch folder with per-file outputs
maxwell-demon --input data/ --mode raw --output-dir results/ --log-base 2
```

If you prefer not to install the CLI entrypoint:

```bash
python -m maxwell_demon.cli --input data/thesis_human.txt --mode raw
```

## Config File (TOML)

You can place defaults in a config file and override them via CLI flags:

```bash
maxwell-demon --config config.example.toml --input data/thesis_human.txt
```

See `config.example.toml` for all options.

## What You Get

Each window produces:
- `mean_entropy`: Shannon entropy (raw) or mean surprisal (diff)
- `entropy_variance`: burstiness proxy (variance of surprisal)
- `compression_ratio`: LZ-based compression ratio
- `unique_ratio`: lexical diversity in the window

All results are exported to CSV for plotting or statistical analysis.

## Library Usage

```python
from maxwell_demon import analyze_tokens, preprocess_text

tokens = preprocess_text(\"testo di esempio...\")
rows = analyze_tokens(tokens, mode=\"raw\", window_size=50, step=10)
```

## Build Reference Dictionary

```bash
maxwell-demon --build-ref-dict data/corpus.txt --ref-dict-out standard_italian.json
```

## Build Reference Dictionary From Frequency List

```bash
maxwell-demon \\
  --build-ref-dict-from-freq data/frequency_list.txt \\
  --freq-url https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/it/it_50k.txt \\
  --freq-cache data/frequency_list.txt \\
  --ref-dict-out standard_italian.json
```

## Plot Results

```bash
# Plot mean entropy over time, colored by label
maxwell-demon-plot --input results_human.csv --metric mean_entropy --hue label --output plots/entropy.png

# Plot burstiness (entropy variance) from a folder of CSVs
maxwell-demon-plot --input results/ --metric entropy_variance --hue filename --output plots/burstiness.png

# Interactive HTML plot
maxwell-demon-plot-html --input results/ --metric mean_entropy --color label --output plots/entropy.html

# Phase diagram (entropy vs compression)
maxwell-demon-phase --input results/ --x mean_entropy --y compression_ratio --color label --output plots/phase.html
```

## Dataset Workflow (Quick Start)

```bash
# 1) Create a dataset skeleton
python scripts/scripts_dataset.py init --name dataset_it_01 --count 50

# 2) Fill the files under data/dataset_it_01/human and data/dataset_it_01/ai

# 3) Validate dataset consistency
python scripts/scripts_dataset.py check --name dataset_it_01

# 4) Run analysis
maxwell-demon --input data/dataset_it_01/human/ --mode raw --output-dir results/dataset_it_01/human/ --label human --log-base 2
maxwell-demon --input data/dataset_it_01/ai/ --mode raw --output-dir results/dataset_it_01/ai/ --label ai --log-base 2

# 5) Plot
maxwell-demon-plot-html --input results/dataset_it_01/ --metric mean_entropy --color label --output plots/dataset_it_01_entropy.html
maxwell-demon-phase --input results/dataset_it_01/ --x mean_entropy --y compression_ratio --color label --output plots/dataset_it_01_phase.html
```

## Documentation

- `DOC/docs.md` — Detailed CLI parameters and options
- `DOC/guide.md` — Step‑by‑step workflow for dataset creation, validation, analysis, and plotting
- `DOC/theoretical_framework.md` — Fondamenti teorici (entropia, compressione, burstiness)

## Why Italian (short version)

Italian is a stronger test bed than English for entropy‑based analysis because its syntax is more flexible and its morphology richer. Humans naturally exploit this flexibility to create local variability (burstiness), while LLMs tend to converge on the most probable structures. Italian also reveals “translationese” patterns (e.g., overuse of pronouns) and long‑range dependencies in complex sentences, which make human vs AI differences more measurable. See `DOC/guide.md` for the full rationale.

## Fetch Human Articles

```bash
python scripts/scripts_fetch_human.py --dataset dataset_it_01 --urls data/urls_example.json --min-words 800
```

## Notes

- Tokenization is minimal: lowercase + basic punctuation removal.
- Entropy and surprisal use the specified log base (`--log-base`).
- Burstiness is approximated as variance of token-level surprisal within each window.
- `unique_ratio` is the ratio of unique tokens in each window.
- Compression ratio uses `zlib` by default and can be switched to `gzip`, `bz2`, or `lzma`.
