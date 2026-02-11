# Maxwell-Demon — Overview and Step-by-Step Guide

Maxwell-Demon is a CLI tool that analyzes text files to capture local entropy dynamics and compression patterns. It is designed to compare human-written texts versus LLM-generated texts by measuring how “surprising” and “bursty” the text is over time. The output is a CSV ready for plotting or further statistical analysis.

The tool operates in two modes:

- **Raw mode**: looks only at the text itself, measuring Shannon entropy in local windows.
- **Differential mode**: compares tokens to a reference dictionary and measures surprisal relative to a language baseline.

The typical workflow is:

1. Create a dataset structure (human + LLM pairs).
2. Fill the dataset with texts.
3. Validate dataset consistency.
4. Run analysis for human and LLM texts.
5. Plot results (static and/or interactive).

Below is a clear, step‑by‑step guide.

---

## Why Italian? (Information-Theoretic Rationale)

Below is a concise rationale for choosing Italian as the experimental language:

1) **Entropic flexibility of Italian**  
English has a rigid S‑V‑O order. Italian has richer morphology and freer syntax, allowing humans to modulate rhythm via real variations (inversions, dislocations). LLMs tend to converge on the statistically most probable structure, often close to canonical S‑V‑O, reducing local variability. The contrast is more visible in Italian.

2) **Translationese signatures**  
Many LLMs are optimized on English; when generating Italian they often exhibit measurable traces such as:
- **Excess pronouns** (Italian is pro‑drop).  
- **Anglicized adjective placement** or overly standard collocations.  
These patterns increase redundancy and lower local entropy.

3) **Long‑range dependencies (hypotaxis)**  
Italian editorial and essayistic prose often uses long sentences with subordination and parentheticals. Humans manage this complexity with bursts of surprise, while LLMs tend to simplify (parataxis) to reduce coherence risk. The result is higher burstiness peaks in human text.

In short: Italian provides a more sensitive “laboratory” for entropy‑based differentiation between human and LLM text.

---

## 1) Create a Dataset Skeleton

A dataset contains paired files: one human text and one LLM text for the same ID.
To scaffold a dataset with 50 pairs:

```bash
python scripts/scripts_dataset.py init --name dataset_it_01 --count 50
```

This creates:

```
data/dataset_it_01/
  human/
  ai/
  metadata.csv
```

Each `human/` and `ai/` folder contains empty files named:

- `001_human.txt` / `001_ai.txt`
- `002_human.txt` / `002_ai.txt`
- …

These IDs define the pairings.

---

## 2) Fill the Dataset

Put your texts into the files created above:

- Human texts go in `data/dataset_it_01/human/` as `*_human.txt`
- LLM texts go in `data/dataset_it_01/ai/` as `*_ai.txt`

Rules for each file:

- Plain UTF‑8 text
- One document per file
- No metadata or markup

Optionally fill `metadata.csv` with the title and source info for each ID.
This helps track the origin of each sample and makes downstream analysis cleaner.

### Optional: Auto-fill human texts from URLs

Instead of copying human texts manually, you can fetch them from article URLs.

Quick URL list (`.txt`, one URL per line):

```bash
python scripts/scripts_fetch_human.py --dataset dataset_it_01 --urls data/urls_example.txt --min-words 800
```

Or structured JSON with metadata (`id`, `title`, `source_type`):

```bash
python scripts/scripts_fetch_human.py --dataset dataset_it_01 --urls data/urls_example.json --min-words 800
```

The script writes text files into `data/<dataset>/human/` and upserts rows in `metadata.csv`.
When the dataset was scaffolded with `scripts_dataset.py init`, it fills empty
`NNN_human.txt` stubs in order. If no stubs are empty, it uses the next available ID.
To avoid accidental data loss, non-empty IDs are skipped unless you pass
`--overwrite-existing-id`.
For punctual corrections, use `--only-id 012` or `--only-file 012_human.txt`.

### Optional: Auto-generate AI shadow texts

You can populate `data/<dataset>/ai/` automatically from human texts with OpenAI:

```bash
python scripts/generate_shadow_dataset.py --dataset dataset_it_01 --config config.local.toml
```

Rules:

- reads `data/<dataset>/human/*.txt`
- skips empty human files
- writes matching AI files into `data/<dataset>/ai/`
- skips non-empty AI files unless `--overwrite-existing`
- uses filename as title and first 100 chars (default) as incipit/context
- for punctual corrections, supports `--only-id 012` or `--only-file 012_human.txt`
- if the selected model does not support `temperature`, the script retries automatically without it

---

## 3) Validate Consistency

Before analysis, check that every human file has a matching LLM file and (optionally) a metadata row:

```bash
python scripts/scripts_dataset.py check --name dataset_it_01
```

For a global check across all datasets under `data/`:

```bash
python scripts/scripts_dataset.py audit
```

This reports missing pairs or missing metadata IDs.

---

## 4) Run the Analysis

You should run analysis separately for human and LLM folders, so labels are clean.

**Human analysis (raw mode):**
```bash
maxwell-demon \
  --input data/dataset_it_01/human/ \
  --mode raw \
  --output-dir results/dataset_it_01/human/ \
  --label human \
  --log-base 2
```

**LLM analysis (raw mode):**
```bash
maxwell-demon \
  --input data/dataset_it_01/ai/ \
  --mode raw \
  --output-dir results/dataset_it_01/ai/ \
  --label ai \
  --log-base 2
```

This produces one CSV per input file. Each CSV contains local metrics for each window.

### Differential Mode (optional)
If you want to compare texts against a reference language model:

1) Build a reference dictionary from a corpus:
```bash
maxwell-demon --build-ref-dict data/corpus.txt --ref-dict-out standard_italian.json
```

2) Run analysis in `diff` mode:
```bash
maxwell-demon \
  --input data/dataset_it_01/ai/ \
  --mode diff \
  --ref-dict standard_italian.json \
  --output-dir results/dataset_it_01/ai_diff/ \
  --label ai \
  --log-base 2
```

---

## 5) Plot the Results

You can generate both static PNGs and interactive HTML plots.

### Static PNG (quick inspection)
```bash
maxwell-demon-plot \
  --input results/dataset_it_01/ \
  --metric mean_entropy \
  --hue label \
  --output plots/dataset_it_01_entropy.png
```

### Interactive HTML (exploration)
```bash
maxwell-demon-plot-html \
  --input results/dataset_it_01/ \
  --metric mean_entropy \
  --color label \
  --output plots/dataset_it_01_entropy.html
```

### Phase Diagram (entropy vs compression)
```bash
maxwell-demon-phase \
  --input results/dataset_it_01/ \
  --x mean_entropy \
  --y compression_ratio \
  --color label \
  --output plots/dataset_it_01_phase.html
```

---

## What the Metrics Mean

Each window produces these core measurements:

- **mean_entropy**:
  Shannon entropy of tokens inside the window (raw mode) or mean surprisal (diff mode). Higher values mean more unpredictability.

- **entropy_variance**:
  Variance of token-level surprisal inside the window. This is the burstiness proxy. Higher values indicate more local variability.

- **compression_ratio**:
  Compression ratio for the window. Lower ratio means the window is more compressible (more regular patterns).

- **unique_ratio**:
  Number of unique tokens divided by window size. A quick measure of lexical diversity.

---

## Recommended Baseline Settings

For most experiments:
- `--window 50`
- `--step 10`
- `--log-base 2`

This gives a good balance of sensitivity and stability.

---

## Typical End‑to‑End Run (Minimal)

```bash
python scripts/scripts_dataset.py init --name dataset_it_01 --count 50
python scripts/scripts_dataset.py check --name dataset_it_01
maxwell-demon --input data/dataset_it_01/human/ --mode raw --output-dir results/dataset_it_01/human/ --label human --log-base 2
maxwell-demon --input data/dataset_it_01/ai/ --mode raw --output-dir results/dataset_it_01/ai/ --label ai --log-base 2
maxwell-demon-phase --input results/dataset_it_01/ --x mean_entropy --y compression_ratio --color label --output plots/dataset_it_01_phase.html
```

This leaves you with clean CSVs and a phase diagram ready for inspection.

## Local Coverage

```bash
./scripts/coverage.sh
```
