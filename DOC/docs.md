# Maxwell-Demon Parameters

This document describes every CLI parameter in detail, with typical usage and behavior.

## Main CLI (`maxwell-demon`)

### `--input`
Path to a single `.txt` file or a folder containing `.txt` files.
- If a file is provided, it is analyzed directly.
- If a folder is provided, all `.txt` files under it (recursive) are analyzed.
- Required unless you are building a reference dictionary with `--build-ref-dict`.

Example:
```bash
maxwell-demon --input data/thesis_human.txt --mode raw
```

### `--mode`
Selects the analysis mode.
- `raw`: Pure Shannon entropy from the window's own token distribution.
- `diff`: Differential mode using a reference dictionary (surprisal relative to a corpus).

Default: `raw`

Example:
```bash
maxwell-demon --input data/chatgpt_text.txt --mode diff --ref-dict standard_italian.json
```

### `--window`
Size of the sliding window in tokens.
- Larger windows smooth results but reduce local sensitivity.
- Smaller windows highlight local dynamics but can be noisier.

Default: `50`

### `--step`
Step size between windows in tokens.
- A smaller step increases overlap between windows.
- A larger step reduces total windows (faster, less granular).

Default: `10`

### `--output`
Output CSV file path.
- Only used when `--output-dir` is NOT set.

Default: `results.csv`

### `--output-dir`
Output directory to write one CSV per input file.
- When set, a CSV is created for each `.txt` file, named `<file_stem>.csv`.
- When set, `--output` is ignored.

Example:
```bash
maxwell-demon --input data/ --mode raw --output-dir results/
```

### `--label`
Optional label attached to each output row.
- Useful to differentiate sources (e.g., `human` vs `ai`).

Allowed: `human`, `ai`

### `--ref-dict`
Path to a reference dictionary JSON for `diff` mode.
- JSON must be a mapping of token -> probability.
- Required when `--mode diff`.

Example:
```bash
maxwell-demon --input data/chatgpt_text.txt --mode diff --ref-dict standard_italian.json
```

### `--build-ref-dict`
Build a reference dictionary from a corpus file.
- This mode ignores `--input` and produces a JSON mapping of token -> probability.
- Tokenization is identical to analysis: lowercase + punctuation removal.

Example:
```bash
maxwell-demon --build-ref-dict data/corpus.txt --ref-dict-out standard_italian.json
```

### `--build-ref-dict-from-freq`
Build a reference dictionary from a frequency list file (word + frequency).
- Useful for pre-LLM frequency lists (e.g., 2018 Italian word list).
- This mode ignores `--input`.

Example (with download):
```bash
maxwell-demon \\
  --build-ref-dict-from-freq data/frequency_list.txt \\
  --freq-url https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/it/it_50k.txt \\
  --freq-cache data/frequency_list.txt \\
  --ref-dict-out standard_italian.json
```

### `--freq-url`
Optional URL to download a frequency list before building the reference dict.

### `--freq-cache`
Local path to save the downloaded frequency list.

### `--ref-dict-out`
Output JSON path for the reference dictionary built by `--build-ref-dict`.

Default: `ref_dict.json`

### `--log-base`
Base of the logarithm used for entropy and surprisal.
- Use `2` to get values in bits.
- Use `e` (default) for nats.
- Must be > 0 and not equal to 1.

Default: `e` (natural log)

Example:
```bash
maxwell-demon --input data/thesis_human.txt --mode raw --log-base 2
```

### `--compression`
Compression algorithm for the ratio metric.
- Options: `zlib`, `gzip`, `bz2`, `lzma`.

### `--unknown-prob`
Fallback probability for unseen tokens in `diff` mode.

### `--smoothing-k`
Add‑k smoothing used when building a reference dictionary from a corpus.

## Plot CLI (`maxwell-demon-plot`)

### `--input`
Path to a single CSV or a folder of CSVs.
- If a folder is provided, all `.csv` files under it (recursive) are loaded.

### `--output`
Output image path for the plot.

Default: `plot.png`

### `--metric`
Column name to plot on the Y axis.
- Typical values: `mean_entropy`, `entropy_variance`, `compression_ratio`, `unique_ratio`.

Default: `mean_entropy`

### `--hue`
Column name used for grouping by color.
- Typical values: `label`, `mode`, `filename`.

Default: `label`

## Plot CLI (`maxwell-demon-plot-html`)

### `--input`
Path to a single CSV or a folder of CSVs.
- If a folder is provided, all `.csv` files under it (recursive) are loaded.

### `--output`
Output HTML path for the interactive plot.

Default: `plot.html`

### `--metric`
Column name to plot on the Y axis.
- Typical values: `mean_entropy`, `entropy_variance`, `compression_ratio`, `unique_ratio`.

Default: `mean_entropy`

### `--color`
Column name used for grouping by color.
- Typical values: `label`, `mode`, `filename`.

Default: `label`

## Plot CLI (`maxwell-demon-phase`)

### `--input`
Path to a single CSV or a folder of CSVs.
- If a folder is provided, all `.csv` files under it (recursive) are loaded.

### `--output`
Output HTML path for the phase diagram.

Default: `phase.html`

### `--x`
Column name for the X axis.

Default: `mean_entropy`

### `--y`
Column name for the Y axis.

Default: `compression_ratio`

### `--color`
Column name used for grouping by color.
- Typical values: `label`, `mode`, `filename`.

Default: `label`

### `--facet`
Optional column to facet by (split into panels).
- Typical values: `filename`, `mode`.

## Output CSV Columns

- `filename`: Source `.txt` filename.
- `window_id`: Sequential index of the window.
- `mean_entropy`: Mean entropy or mean surprisal (depends on mode).
- `entropy_variance`: Variance of surprisal in the window (burstiness proxy).
- `compression_ratio`: `len(zlib_compressed) / len(raw_bytes)` for the window.
- `unique_ratio`: Unique tokens / window size.
- `mode`: `raw` or `diff`.
- `label`: Optional label from CLI.
- `log_base`: Log base used for entropy/surprisal.
- `compression`: Compression algorithm used.

## Preprocessing Notes

- Text is lowercased.
- Basic punctuation is removed.
- Tokens are split on whitespace after cleaning.

## Common Pitfalls

- `diff` mode requires `--ref-dict`.
- Very short texts may produce a single window.
- If `--output-dir` is set, `--output` is ignored.

## Config File (TOML)

You can configure default parameters in a TOML file and override them via CLI.

Example (`config.example.toml`):

```toml
[analysis]
mode = "raw"
window = 50
step = 10
log_base = 2.0

[compression]
algorithm = "zlib" # zlib, gzip, bz2, lzma

[reference]
path = "standard_italian.json"
smoothing_k = 0.0
unknown_prob = 1e-10
```

Run:

```bash
maxwell-demon --config config.example.toml --input data/thesis_human.txt
```

## Dataset Workflow (Step by Step)

1. Create a dataset skeleton:
```bash
python scripts/scripts_dataset.py init --name dataset_it_01 --count 50
```

2. Fill the files:
- Put human texts in `data/dataset_it_01/human/001_human.txt`, `002_human.txt`, ...
- Put LLM texts in `data/dataset_it_01/ai/001_ai.txt`, `002_ai.txt`, ...
- Optionally fill `data/dataset_it_01/metadata.csv`.

3. Validate dataset consistency:
```bash
python scripts/scripts_dataset.py check --name dataset_it_01
```

4. Run analysis and generate results:
```bash
maxwell-demon --input data/dataset_it_01/human/ --mode raw --output-dir results/dataset_it_01/human/ --label human --log-base 2
maxwell-demon --input data/dataset_it_01/ai/ --mode raw --output-dir results/dataset_it_01/ai/ --label ai --log-base 2
```

5. Plot results (optional):
```bash
maxwell-demon-plot --input results/dataset_it_01/human/ --metric mean_entropy --hue label --output plots/dataset_it_01_entropy.png
maxwell-demon-plot-html --input results/dataset_it_01/ --metric mean_entropy --color label --output plots/dataset_it_01_entropy.html
maxwell-demon-phase --input results/dataset_it_01/ --x mean_entropy --y compression_ratio --color label --output plots/dataset_it_01_phase.html
```

## Global Consistency Check

```bash
python scripts/scripts_dataset.py audit
```

This checks all datasets under `data/` and reports missing pairs or missing metadata rows.

## Fetch Human Articles from URLs

Provide a JSON file with a list of URLs and optional metadata. Example:

```json
[
  {
    "id": "001",
    "url": "https://example.com/article-1",
    "title": "Article 1",
    "source_type": "editoriale"
  }
]
```

Run:

```bash
python scripts/scripts_fetch_human.py --dataset dataset_it_01 --urls data/urls_example.json --min-words 800
```

This downloads and parses each article (text only), stores it in `data/<dataset>/human/`,
and appends a row to `data/<dataset>/metadata.csv`.

### Extra Options

- `--retries`: number of download retries per URL (default: 3).
- `--retry-delay`: seconds to wait between retries (default: 1.0).
- `--fail-log`: path to a failure log file (default: `data/<dataset>/fetch_failures.log`).

Example:

```bash
python scripts/scripts_fetch_human.py --dataset dataset_it_01 --urls data/urls_example.json --min-words 800 --retries 5 --retry-delay 2 --fail-log data/dataset_it_01/fetch_failures.log
```

## Local Coverage

```bash
./scripts/coverage.sh
```
