# Data

This project uses the UCI Cleveland Heart Disease dataset.

Run:

```bash
python -m src.data_ingestion
```

Outputs:

- `data/raw/heart_disease.csv`
- `data/processed/heart_disease_clean.csv`

The original target `target` has values from 0 to 4. The binary modelling target is:

- `0`: absence of heart disease
- `1`: presence of heart disease, generated from original values 1 to 4
