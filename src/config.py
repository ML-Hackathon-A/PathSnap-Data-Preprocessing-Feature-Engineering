"""Central configuration. Every value can be overridden in the project-root .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _get(name, default, cast=str):
    val = os.getenv(name)
    return cast(val) if val not in (None, "") else default


def _path(name, default):
    p = Path(_get(name, str(default)))
    return p if p.is_absolute() else ROOT / p  # relative paths in .env are relative to the project root


DATA_DIR = _path("PATHSNAP_DATA_DIR", ROOT / "data")
EVAL_DATA_DIR = _path("PATHSNAP_EVAL_DIR", DATA_DIR / "student_test")
FEATURE_DIR = ROOT / "features"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MAPS_DIR = RESULTS_DIR / "maps"
for _d in (FEATURE_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR, MAPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = _get("SEED", 42, int)
N_JOBS = _get("N_JOBS", -1, int)
N_FOLDS = _get("N_FOLDS", 5, int)

# Candidate generation
CANDIDATE_RADIUS_M = _get("CANDIDATE_RADIUS_M", 50.0, float)
MAX_CANDIDATES = _get("MAX_CANDIDATES", 12, int)
INDEX_STEP_M = _get("INDEX_STEP_M", 10.0, float)

# Keep every Nth GPS point of TRAINING trips (1 Hz data is highly redundant). Student-test always keeps all.
TRAIN_OBS_STEP = _get("TRAIN_OBS_STEP", 2, int)

# HMM defaults (overwritten by artifacts/hmm_config.json after decoder tuning)
HMM_BETA = _get("HMM_BETA", 5.0, float)
HMM_EMISSION_WEIGHT = _get("HMM_EMISSION_WEIGHT", 1.0, float)
HMM_GAP_SECONDS = _get("HMM_GAP_SECONDS", 10.0, float)
HMM_SEARCH_CUTOFF_M = _get("HMM_SEARCH_CUTOFF_M", 400.0, float)

OPTUNA_TRIALS = _get("OPTUNA_TRIALS", 40, int)


def partition_dir(partition: str) -> Path:
    if partition == "train":
        return DATA_DIR / "train"
    if partition == "student_test":
        return DATA_DIR / "student_test"
    if partition == "eval":
        return EVAL_DATA_DIR
    raise ValueError(f"unknown partition {partition}")


def max_train_rows(model_name: str) -> int:
    """0 = use all rows. Lets slow models (KNN, MLP) train on a random subsample."""
    return _get(f"MAX_TRAIN_ROWS_{model_name.upper()}", 0, int)
