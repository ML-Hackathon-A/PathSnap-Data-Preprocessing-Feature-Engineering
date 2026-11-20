# PathSnap - Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University). Noisy 1 Hz GPS traces are matched to a directed
road network: candidate generation, then a per-candidate classifier (5 models compared, the best tuned
with Optuna), then HMM/Viterbi route decoding, a FastAPI backend (Laptop 1) and a Streamlit UI (Laptop 2).

## Team

The work was split equally across three people. Every stage of the project was divided, so each
member contributed to preprocessing, feature engineering, modelling, optimisation, deployment and
documentation.

| Stage | Akshath Girish Shirolkar | Akshath Patil | Aditya Jaiswal |
|---|---|---|---|
| Data preprocessing | Raw-data validation notebook: missing values, invalid coordinates/ids, duplicates (`notebooks/01_data_preprocessing.ipynb`) | Timestamp and route-connectivity checks, cleaning rules in the pipeline (`src/data_io.py`) | Exploratory data analysis: trip inventory, distributions, trajectory and network plots (`src/eda.py`) |
| Feature engineering | Local metric projection, road-network model and KD-tree spatial index (`src/geo.py`) | Candidate generation and monotonic DP label alignment with the reference route (`src/features.py`) | The 21 per-candidate features and the dataset builder (`src/features.py`, `src/build_dataset.py`) |
| Model training (5 models) | Logistic Regression and KNN | Random Forest and MLP | LightGBM, evaluation metrics and model comparison (`src/metrics.py`, `src/compare_models.py`) |
| Optimisation and decoding | Bayesian hyperparameter optimisation with Optuna (`src/tune.py`) | HMM / Viterbi route reconstruction (`src/hmm.py`) | Decoder parameter tuning and final evaluator-test evaluation (`src/tune_decoder.py`, `src/evaluate_holdout.py`) |
| Deployment | FastAPI backend and inference pipeline (`api/main.py`, `src/pipeline.py`) | Streamlit evaluator UI (`ui/app.py`) | Two-laptop start scripts, packaging and dependency setup (`start_*.ps1`, `tools/`, `requirements/`) |
| Visualisation and documentation | Technical report sections 2-3 (EDA, model comparison) | Technical report sections 4-5 (optimisation, HMM) | Folium maps, report sections 6-7, README and viva guide (`src/make_map.py`, `report/`) |

## Folder layout

| Folder | Contents |
|---|---|
| `notebooks/` | `01_data_preprocessing.ipynb` - raw-data validation (missing values, invalid ids, duplicates, timestamps, route connectivity) |
| `src/` | pipeline: `data_io` (loading + cleaning), `geo` (projection, road network, KD-tree), `features` (candidates, label alignment, 21 features), `build_dataset`, `models` (5 models + search spaces), `train_model`, `compare_models`, `tune` (Optuna), `finalize`, `hmm` (Viterbi), `pipeline` (inference), `tune_decoder`, `make_map` (Folium), `eda` |
| `api/` | FastAPI backend (`main.py`) |
| `ui/` | Streamlit evaluator UI (`app.py`) |
| `report/` | `build_report.py` (generates the PDF from `results/`), the report PDF, screenshots |
| `tools/` | `screenshots.py`, `package_model.py` (single-model zip for a team member) |
| `artifacts/` | `final_model.joblib`, `final_manifest.json`, `hmm_config.json`, `best_params_lgbm.json` |
| `results/` | every result: `result_<model>.json`, `cv_folds_<model>.csv`, `model_comparison.csv`, `table_3_1_model_comparison.csv`, `optuna_trials_lgbm.csv`, `hmm_grid_*.csv`, `eda/`, `plots/`, `maps/`, `logs/` |
| `requirements/` | pinned dependencies: `base.txt`, `model_<name>.txt`, `tune.txt`, `api.txt`, `ui.txt` |

`data/`, `features/`, `envs/` and `.env` are not committed. Put the dataset in `data/train` and `data/student_test`.

## Installation (Windows PowerShell, Python 3.11)

```powershell
Copy-Item .env.example .env          # local settings: data paths, N_JOBS, row caps, API URL
foreach ($n in "logreg","knn","rf","lgbm","mlp") {
  py -3.11 -m venv "envs\$n"; & "envs\$n\Scripts\python.exe" -m pip install -r "requirements\model_$n.txt" }
foreach ($n in "tune","api","ui") {
  py -3.11 -m venv "envs\$n"; & "envs\$n\Scripts\python.exe" -m pip install -r "requirements\$n.txt" }
```

## Execution (reproduces every number in the report)

```powershell
& envs\lgbm\Scripts\python.exe -m src.build_dataset                  # candidates + features + labels
& envs\logreg\Scripts\python.exe -m src.train_model --model logreg   # 5-fold GroupKFold CV + student_test
& envs\knn\Scripts\python.exe    -m src.train_model --model knn
& envs\rf\Scripts\python.exe     -m src.train_model --model rf
& envs\lgbm\Scripts\python.exe   -m src.train_model --model lgbm
& envs\mlp\Scripts\python.exe    -m src.train_model --model mlp
& envs\tune\Scripts\python.exe   -m src.tune --model lgbm --trials 25    # Bayesian optimisation (TPE)
& envs\lgbm\Scripts\python.exe   -m src.compare_models                 # comparison table + plots
& envs\tune\Scripts\python.exe   -m src.finalize --run-id <lgbm|lgbm_tuned>
& envs\api\Scripts\python.exe    -m src.tune_decoder                   # HMM beta / emission weight + HMM vs classifier
& envs\api\Scripts\python.exe    -m src.eda
& envs\api\Scripts\python.exe    -m src.make_map --partition student_test --trip 00000076
& envs\api\Scripts\python.exe    tools\screenshots.py
& envs\api\Scripts\python.exe    -m src.evaluate_holdout --partition eval   # final: frozen models on evaluator_test (data/eval)
& envs\api\Scripts\python.exe    -m src.plot_holdout
& envs\api\Scripts\python.exe    report\build_report.py
& envs\api\Scripts\python.exe    report\build_viva.py                      # viva preparation guide
```

## Deployment (two laptops, same network)

Laptop 1 (model + API, holds the evaluation trips and their reference routes):
```powershell
# .env: PATHSNAP_EVAL_DIR=./data/eval   (folder with the 10 evaluation trip folders)
& envs\api\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# allow the port once (Administrator PowerShell):
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```
Laptop 2 (UI): `.env` with `PATHSNAP_API_URL=http://<Laptop-1-IP>:8000`, then
```powershell
& envs\ui\Scripts\python.exe -m streamlit run ui\app.py --server.port 8501
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Reference routes are used
only on the server, after prediction, to compute accuracy; they are never returned to the UI.

## Integrity

Evaluator-test data is not used for training, tuning, feature selection or thresholds. There are no
hard-coded predictions. AI assistance was used in development (see report section 8).
