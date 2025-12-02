<!--
CS 4375 Final Project

Name: Shawn Francis
NetID: saf210007
Class: CS 4375.001
Professor: Anurag Nagar
Days Used: 2
Turned In: 12/2/2025

## NBA RAPM Project README

Note: It is enough to use the provided data and run
the dashboard, but the TA can re-run the full pipeline if they 
choose, however fair warning, ingesting the data will take a good 
amount of time.
-->

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

1. Go to the public dataset link provided in the project submission
(the separate GitHub repository that contains the data
files).

2. Download the data and copy the "data" and "results" folders so
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


### To run the RAPM pipeline:

1. In the activated virtual environment, from the project folder
run: python -m src.run_pipeline --season 2024-25 --skip-ingest

This runs the program assuming the preprocessed data files are already in
data/processed_stints and data/matrices.


### To run the Streamlit dashboard:

1. Make sure results/rapm_outputs.csv exists (either from the
downloaded data or after running the pipeline).

2. In the activated virtual environment, from the project folder,
run: streamlit run app.py



