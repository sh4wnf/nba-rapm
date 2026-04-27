## Results Data in This Repo

This repository is configured to be demo-friendly while keeping large artifacts out of git.

### Included for demo mode

- `rapm_demo.csv`: lightweight sample RAPM output for Streamlit hosting and portfolio demos.

### Excluded by design

- Full `rapm_outputs.csv` from complete runs
- Large intermediate outputs
- Experiment log history

### Full-scale data

Full results are maintained in external data storage/repository. If `results/rapm_outputs.csv` exists, the app uses it automatically. If not, it falls back to `results/rapm_demo.csv`.
