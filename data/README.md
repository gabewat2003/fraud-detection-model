# Data

This project uses the **IEEE-CIS Fraud Detection** competition data (Vesta Corporation, hosted on Kaggle).
The files are licensed for competition use only, so they are **never committed** (`data/raw/` is gitignored)
and nothing derived at row level is committed either. Only aggregate reports and charts are.

## Get the files

1. Sign in to Kaggle and open <https://www.kaggle.com/competitions/ieee-fraud-detection/data>.
2. Accept the competition rules (required before the download button works).
3. Download `train_transaction.csv` and `train_identity.csv` and put them in `data/raw/`:

```
data/raw/train_transaction.csv   ~590,540 rows, 394 columns, ~650 MB
data/raw/train_identity.csv      ~144,233 rows,  41 columns,  ~25 MB
```

Optional, with the Kaggle CLI and an API token in `~/.kaggle/kaggle.json`:

```bash
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw
```

(Unzip if the CLI saves `.zip` files.) The `test_*` files are not used: they have no labels.

4. Check: `make data-check`

## No Kaggle account?

`make sample` generates a synthetic dataset with the same schema into `data/sample/` and runs the whole
pipeline on it. The numbers it produces are meaningless; it exists so tests and CI never need the real data.

## What gets created

| Path | Contents |
|---|---|
| `data/processed/fraud.duckdb` | raw tables, `base` (LEFT join), `keyed`, feature tables, lookups |
| `data/processed/features.parquet` | one row per transaction, all features + split label |
| `data/processed/scored.parquet` | per-model raw and calibrated scores for every split |
| `models/serving.duckdb` | narrow history + frequency lookups for the scoring API |
