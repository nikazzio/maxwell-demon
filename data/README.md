# Dataset Structure

You can keep multiple datasets under `data/`.
Each dataset is a folder with a fixed structure:

```
data/
  dataset_name/
    human/
      001_human.txt
      ...
    ai/
      001_ai.txt
      ...
    metadata.csv
```

## File Format

- Plain UTF-8 `.txt` files.
- One document per file.
- No frontmatter or markup.

## Naming Convention

- Use 3-digit IDs: `001`, `002`, ...
- Human files: `<id>_human.txt`
- LLM files: `<id>_ai.txt`
- The same ID links the human/LLM pair.

## metadata.csv (optional but recommended)

Columns (suggested):
- `id` (e.g., 001)
- `title` (topic or prompt title)
- `human_source` (editorial/thesis/essay)
- `ai_model` (if known)
- `notes`

Example:
```csv
id,title,human_source,ai_model,notes
001,Crisi energetica in Europa,editoriale,,
002,Etica dell'IA nei media,saggio,,
```

## Filling Human Texts from URLs

Use `scripts/scripts_fetch_human.py` to populate `human/` from a URL list:

```bash
python scripts/scripts_fetch_human.py --dataset dataset_name --urls data/urls_example.txt --min-words 800
```

Behavior summary:

- `scripts/scripts_dataset.py init` creates empty `NNN_human.txt` stubs.
- `scripts/scripts_fetch_human.py` fills those empty stubs first.
- If stubs are full, it creates the next `NNN_human.txt`.
- Existing non-empty IDs are skipped unless `--overwrite-existing-id` is used.

## Running the analyzer

```bash
maxwell-demon --input data/dataset_name/human/ --mode raw --output-dir results/dataset_name/human/ --label human --log-base 2
maxwell-demon --input data/dataset_name/ai/ --mode raw --output-dir results/dataset_name/ai/ --label ai --log-base 2
```
