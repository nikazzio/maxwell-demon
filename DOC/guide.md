# Maxwell-Demon — Overview and Step-by-Step Guide

Maxwell-Demon is a CLI tool that analyzes text files to capture local entropy dynamics and compression patterns. It is designed to compare human-written texts versus LLM-generated texts by measuring how “surprising” and “bursty” the text is over time. The output is a CSV ready for plotting or further statistical analysis.

The tool operates in two modes:

- **Raw mode**: looks only at the text itself, measuring Shannon entropy in local windows.
- **Differential mode**: compares tokens to a reference dictionary and measures surprisal relative to a language baseline.

The typical workflow is:

1. Create a dataset structure (human + AI pairs).
2. Fill the dataset with texts.
3. Validate dataset consistency.
4. Run analysis for human and AI texts.
5. Plot results (static and/or interactive).

Below is a clear, step‑by‑step guide.

---

## Why Italian? (Information-Theoretic Rationale)

Gemini said: **Assolutamente in Italiano.**  
Here is the reasoning from an information‑theoretic perspective, adapted for this project:

1) **Flessibilita entropica dell'italiano**  
L'inglese e rigido in ordine S‑V‑O. L'italiano ha morfologia ricca e sintassi piu libera, quindi l'umano puo modulare il ritmo con variazioni reali (es. inversioni, dislocazioni).  
Gli LLM tendono a convergere sulla forma piu probabile, spesso vicina allo schema S‑V‑O, riducendo la variabilita locale. In italiano questo contrasto e piu visibile.

2) **Translationese (tracce di “traduzione interna”)**

Molti LLM sono ottimizzati su inglese; quando generano in italiano emergono firme statistiche tipiche:  

- **Eccesso di pronomi** (italiano e pro‑drop).  
- **Aggettivazione “anglicizzata”** o collocazioni troppo standard.  
Questi pattern introducono ridondanza e abbassano l'entropia locale.

3) **Dipendenze a lungo raggio (ipotassi)**

L'italiano saggistico e letterario usa frasi lunghe con subordinate e incisi.  
L'umano gestisce questa complessita con burst di sorpresa, mentre l'LLM tende a semplificare (paratassi) per ridurre il rischio di incoerenze.  
Risultato: la **burstiness** negli umani tende a mostrare picchi piu alti.

In sintesi: l'italiano e un “laboratorio” piu sensibile per rilevare la differenza tra testo umano e testo LLM.

---

## 1) Create a Dataset Skeleton

A dataset contains paired files: one human text and one AI text for the same ID.
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
- AI texts go in `data/dataset_it_01/ai/` as `*_ai.txt`

Rules for each file:

- Plain UTF‑8 text
- One document per file
- No metadata or markup

Optionally fill `metadata.csv` with the title and source info for each ID.
This helps track the origin of each sample and makes downstream analysis cleaner.

---

## 3) Validate Consistency

Before analysis, check that every human file has a matching AI file and (optionally) a metadata row:

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

You should run analysis separately for human and AI folders, so labels are clean.

**Human analysis (raw mode):**

```bash
maxwell-demon \
  --input data/dataset_it_01/human/ \
  --mode raw \
  --output-dir results/dataset_it_01/human/ \
  --label human \
  --log-base 2
```

**AI analysis (raw mode):**

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

1) Run analysis in `diff` mode:

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
  Zlib compression ratio for the window. Lower ratio means the window is more compressible (more regular patterns).

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
