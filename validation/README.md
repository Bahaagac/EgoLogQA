# Validation Pack

This folder contains deterministic tooling for evaluating EgoLogQA against human labels on a fixed Hugging Face dataset revision.

## Files

- `sample_plan.md`: deterministic file-selection protocol.
- `labels_template.csv`: template for human labels.
- `run_validation.py`: batch runner that writes `validation/results/run_index.json`.
- `summarize_results.py`: computes `validation/results/summary.md`, `summary.csv`, and `confusion_matrix.json`.

## Usage

1. Run batch analysis:

```bash
.venv/bin/python validation/run_validation.py \
  --repo-id MicroAGI-Labs/MicroAGI00 \
  --revision main \
  --prefix raw_mcaps/ \
  --limit 50
```

2. Create `validation/labels.csv` from `validation/labels_template.csv` and fill labels.

3. Summarize:

```bash
.venv/bin/python validation/summarize_results.py
```

Generated outputs in `validation/runs/`, `validation/results/`, and `validation/labels.csv` are intentionally git-ignored.
