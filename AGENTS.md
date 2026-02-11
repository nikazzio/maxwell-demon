# 🤖 Agent Guidelines

**Current Context**: You are working on **Maxwell-Demon** (Python package + CLI + plotting tools). The supported interfaces are the package APIs under `src/maxwell_demon/` and the CLI entry points declared in `pyproject.toml`.

## 🧭 Project Status (11-02-2026)

Il progetto usa un **SRC Layout** con package principale `maxwell_demon` e tool CLI dedicati per analisi e visualizzazione.

### ✅ Current Milestones

- **Architecture**: codice applicativo centralizzato in `src/maxwell_demon/`.
- **Entry Points**: `maxwell-demon`, `maxwell-demon-plot`, `maxwell-demon-plot-html`, `maxwell-demon-phase` definiti in `pyproject.toml`.
- **Dependency Management**: dipendenze runtime/dev in `pyproject.toml`; `requirements.txt` usato come compat layer (`-e .[dev]`).
- **Quality Gates**: workflow CI attivo con test e coverage; semantic release configurata su branch `main`.

## 🏗️ Project Structure & Rules (MANDATORY)

### 📂 Layout

- **Source Code**: tutto il codice libreria/CLI deve vivere in `src/maxwell_demon/`.
- **Tests**: i test vivono in `tests/`.
- **Scripts**: utility operative in `scripts/`.
- **Docs**: documentazione in `DOC/`.
- **Root Policy**: evita nuovi script Python nella root salvo casi eccezionali e motivati.

### 📦 Dependencies & Config

1. **Dependencies**: aggiungi librerie solo in `pyproject.toml` (`[project.dependencies]` o `[project.optional-dependencies]`).
2. **Editable Install**: installazione locale con `pip install -e .[dev]`.
3. **Config Runtime**: usa file TOML (`--config`) e `config.local.toml` per override locali; non hardcodare percorsi sensibili all'ambiente.
4. **Reference Dicts**: per modalità `diff`, gestisci i dizionari con le opzioni CLI (`--ref-dict`, `--ref-dict-out`, `--build-ref-dict*`).

### 🐍 Coding Standards

- **Ruff** è il linter/formatter di riferimento.
- **Typing**: aggiungi type hints alle API pubbliche e alle funzioni non banali.
- **Path Handling**: preferisci `pathlib.Path` per i percorsi filesystem.
- **Function Design**: evita funzioni monolitiche; separa parsing, trasformazioni e I/O.

## 🗺️ Orientation

- **Main CLI Entry Point**: `src/maxwell_demon/cli.py` (`main()`).
- **Core Analysis**: `src/maxwell_demon/analyzer.py`, `src/maxwell_demon/metrics.py`.
- **Config Parsing**: `src/maxwell_demon/config.py`.
- **Plot Tools**:
  - `src/maxwell_demon/tools/plot_results.py`
  - `src/maxwell_demon/tools/plot_results_html.py`
  - `src/maxwell_demon/tools/plot_phase.py`

## 🚀 Build, Run, Test

- **Install (dev)**: `pip install -e .[dev]`
- **Run CLI**: `maxwell-demon --input <file-or-dir> --mode raw --output <file.csv>`
- **Run as module**: `python -m maxwell_demon.cli ...`
- **Run tests**: `python -m pytest`
- **Coverage**: `./scripts/coverage.sh`
- **Lint/format**:
  - `ruff check .`
  - `ruff format .`

## ⚠️ Critical Constraints

1. **Input/Output Safety**: validare path input e preservare comportamento consistente tra input file e directory.
2. **Data Separation**: `data/`, `results/`, `plots/`, `*.csv`, log e artefatti coverage sono output/runtime e non vanno committati.
3. **Mode Semantics**: mantenere chiara separazione tra `raw` e `diff`; modifiche al calcolo metriche richiedono aggiornamento test e documentazione.
4. **CLI Compatibility**: evitare breaking changes non necessari nelle opzioni già documentate in `README.md` e `DOC/docs.md`.

## 🧹 Data & Output Workflow

- Prima di benchmark o test end-to-end, verifica che `results/` e `plots/` non contengano residui fuorvianti.
- Se aggiungi nuovi output runtime, aggiorna `.gitignore` nella stessa PR.
- Se tocchi pipeline dataset (`scripts/scripts_dataset.py` o `scripts/scripts_fetch_human.py`), documenta i prerequisiti nel README/guide.

## 🚀 GitHub & PR Strategy (MANDATORY)

- **Branching**: non lavorare direttamente su `main`; usa branch `feat/*`, `fix/*`, `docs/*`, `chore/*`.
- **Conventional Commits**: usa prefissi `feat:`, `fix:`, `docs:`, `chore:`.
- **PR Readiness Checklist**:
  1. `ruff check .`
  2. `ruff format .`
  3. `python -m pytest`
  4. se rilevante, `./scripts/coverage.sh`
  5. aggiornare docs quando cambiano opzioni CLI/comportamenti

## 📦 Semantic Release

- I commit message guidano il versioning automatico.
- La versione è in `pyproject.toml` (`project.version`) ed è gestita dal workflow di semantic release.

## 🧑‍💻 Coding Expectations

- Prediligi funzioni piccole, pure quando possibile, e side-effect confinati.
- Mantieni backward compatibility per CSV columns e naming, salvo cambi esplicitamente pianificati.
- Aggiungi o aggiorna test per:
  - parsing opzioni CLI
  - casi limite (input vuoto, finestra > lunghezza testo, file mancanti)
  - regressioni su metriche (`mean_entropy`, `entropy_variance`, `compression_ratio`, `unique_ratio`)
- Aggiorna la documentazione tecnica quando introduci nuovi flag o output.

## ✅ Definition of Done

Una modifica è pronta quando:

1. codice e test sono allineati,
2. lint/format passano,
3. la CLI mantiene comportamento prevedibile,
4. documentazione (`README.md`/`DOC/*.md`) riflette le modifiche,
5. la PR è leggibile, atomica e con commit convenzionali.
