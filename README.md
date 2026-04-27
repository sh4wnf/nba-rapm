This project contains a Python implementation of an NBA Regularized
Adjusted Plus-Minus (RAPM) pipeline and a Streamlit dashboard for
exploring player impact based on play-by-play data.


### To set up the Python environment:

1. Open your terminal (Command Prompt or PowerShell on Windows,
Terminal on macOS/Linux) and navigate to the project folder.

2. Create and activate a virtual environment
run: python -m venv .venv
then activate it on windows by running: .venv\Scripts\activate
on mac by running: source .venv/bin/activate

3. Install the dependencies
run: pip install -r requirements.txt


### To get the dataset:

1. Go to the public dataset repository:
   https://github.com/sh4wnf/nba-rapm-data

2. Download or clone that repository, then copy the "data" and "results" folders so
that they are directly inside the project folder. The final layout
should look like:

NBA-Player-Impact/
  app.py
  README.md
  requirements.txt
  src/
  logs/
  run_dashboard.sh
  data/
    processed_stints/stints.csv
    matrices/X_off.npz, X_def.npz, y_off.npy, y_def.npy, player_to_col.json
  results/
    rapm_outputs.csv
    experiment_logs/log.json (if included)

### Demo mode (no full dataset required)

If `results/rapm_outputs.csv` is not present, the app automatically falls back to
`results/rapm_demo.csv` so the dashboard can still run for a lightweight demo.

This is useful for portfolio/recruiter sharing and Streamlit hosting when you do not
want to commit large data artifacts.

### To run the Streamlit dashboard:

1. Make sure results/rapm_outputs.csv exists (either from the
downloaded data or after running the pipeline).

2. In the activated virtual environment, from the project folder,
run: streamlit run app.py

### To run the RAPM pipeline:

1. In the activated virtual environment, from the project folder
run: python -m src.run_pipeline --season 2024-25 --skip-ingest

This runs the program assuming the preprocessed data files are already in
data/processed_stints and data/matrices.






