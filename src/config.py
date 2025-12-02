# the central config file for the pipeline
from pathlib import Path
import os

SEASON = os.getenv("RAPM_SEASON", "2024-25")

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PBP_RAW_DIR = DATA_DIR / "pbp_raw"  
PROCESSED_STINTS_DIR = DATA_DIR / "processed_stints"
MATRICES_DIR = DATA_DIR / "matrices" 

RESULTS_DIR = ROOT / "results"
RAPM_OUTPUTS_CSV = RESULTS_DIR / "rapm_outputs.csv"
EXPERIMENT_LOGS_DIR = RESULTS_DIR / "experiment_logs"

PBP_COMBINED = DATA_DIR / "pbp_combined.csv"
ALL_GAMES_CSV = DATA_DIR / "all_games.csv"  

STINTS_FILE = PROCESSED_STINTS_DIR / "stints.csv"

LOGS_DIR = ROOT / "logs"
INGEST_LOG = LOGS_DIR / "ingest.log"

CHECKPOINT = ROOT / "checkpoint.json"

# ridge regularization default
DEFAULT_LAMBDA = float(os.getenv("RAPM_LAMBDA", "200.0"))

# bootstrap parameters
DEFAULT_BOOTSTRAP_ITERATIONS = int(os.getenv("RAPM_BOOTSTRAP_ITERATIONS", "200"))

# stint filtering parameters
DEFAULT_MIN_POSSESSIONS = float(os.getenv("RAPM_MIN_POSSESSIONS", "1.0"))

# rapm now clips targets to ±50
DEFAULT_NET_CAP = float(os.getenv("RAPM_NET_CAP", "50.0"))

