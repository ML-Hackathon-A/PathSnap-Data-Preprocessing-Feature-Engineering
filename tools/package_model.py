"""Create a self-contained zip that lets a teammate train ONE model on their own laptop.

    python tools/package_model.py rf mlp
Outputs dist/PathSnap_<model>.zip
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

FAMILY = {"logreg": "Logistic Regression", "knn": "Distance-weighted KNN", "rf": "Random Forest",
          "lgbm": "LightGBM", "mlp": "MLP neural network"}

RUN_PS1 = r'''# PathSnap - train the {name} model ({family})
# Usage (PowerShell, inside the unzipped folder):   powershell -ExecutionPolicy Bypass -File .\run_{model}.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {{ throw "Python launcher 'py' not found. Install Python 3.11: winget install Python.Python.3.11" }}
if (-not (Test-Path "envs\{model}\Scripts\python.exe")) {{
    Write-Host "Creating virtual environment envs\{model} ..."
    py -3.11 -m venv "envs\{model}"
    & "envs\{model}\Scripts\python.exe" -m pip install --upgrade pip
    & "envs\{model}\Scripts\python.exe" -m pip install -r "requirements\model_{model}.txt"
}}
if (-not (Test-Path ".env")) {{ Copy-Item ".env.example" ".env" }}
$py = "envs\{model}\Scripts\python.exe"
New-Item -ItemType Directory -Force results\logs | Out-Null
$ErrorActionPreference = "Continue"   # Python progress output on stderr must not abort the script
$env:PYTHONWARNINGS = "ignore"
Write-Host "Step 1/2: building candidate + feature tables ..."
& $py -m src.build_dataset | Tee-Object results\logs\build_dataset.log
if ($LASTEXITCODE -ne 0) {{ throw "build_dataset failed (exit $LASTEXITCODE)" }}
Write-Host "Step 2/2: training + evaluating {name} (5-fold GroupKFold CV, then student_test) ..."
& $py -m src.train_model --model {model} | Tee-Object results\logs\train_{model}.log
if ($LASTEXITCODE -ne 0) {{ throw "train_model failed (exit $LASTEXITCODE)" }}
Write-Host ""
Write-Host "DONE. Send back these files:"
Write-Host "  results\result_{model}.json, results\cv_folds_{model}.csv, results\plots\confusion_{model}.png, results\logs\train_{model}.log"
'''

README = """# PathSnap - {name} ({family}) training package

Contents: pipeline code (`src/`), the train + student_test data (`data/`), pinned requirements and a
one-click script. Everything runs locally; nothing is uploaded.

## Run (Windows, Python 3.11 installed)

```powershell
cd <unzipped folder>
powershell -ExecutionPolicy Bypass -File .\\run_{model}.ps1
```

The script creates `envs\\{model}` (a virtual environment with the pinned packages), copies
`.env.example` to `.env`, builds the features (~20 s) and trains/evaluates the model.

## Settings (`.env`)

* `N_JOBS` - CPU cores to use (-1 = all).
* `MAX_TRAIN_ROWS_{upper}` - cap on training rows (0 = all). Lower it if the run is slow or out of memory.
* `TRAIN_OBS_STEP` - keep every Nth GPS point of training trips (default 2).

Keep `SEED`, `N_FOLDS`, `CANDIDATE_RADIUS_M`, `MAX_CANDIDATES` unchanged so results stay comparable
with the rest of the team.

## Output

* `results/result_{model}.json` - all Deliverables 3.1 metrics (CV macro F1, Top-1, route continuity,
  student-test macro F1, log loss, inference time, ...)
* `results/cv_folds_{model}.csv` - per-fold results (train vs validation F1 -> over/under-fitting)
* `results/plots/confusion_{model}.png`{extra}
* `artifacts/models/{model}.joblib` - the trained model

This package is for members of OUR team only - the hackathon rules forbid sharing code, models or
results with other teams.
"""

FAM2 = {"logreg": "linear / probabilistic", "knn": "instance-based", "rf": "ensemble / bagging",
        "lgbm": "boosting", "mlp": "neural network"}
EXCLUDE_DIRS = {"__pycache__"}


def add_tree(z, src: Path, arc_prefix: str):
    for p in src.rglob("*"):
        if p.is_file() and not (set(p.parts) & EXCLUDE_DIRS):
            z.write(p, f"{arc_prefix}/{p.relative_to(src).as_posix()}")


def package(model):
    name = FAMILY[model]
    top = f"PathSnap_{model}"
    out = DIST / f"{top}.zip"
    extra = "\n* `results/plots/history_mlp.png` - epoch-wise training loss / validation accuracy" if model == "mlp" else ""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, ROOT / "src", f"{top}/src")
        add_tree(z, ROOT / "data" / "train", f"{top}/data/train")
        add_tree(z, ROOT / "data" / "student_test", f"{top}/data/student_test")
        z.write(ROOT / "requirements" / "base.txt", f"{top}/requirements/base.txt")
        z.write(ROOT / "requirements" / f"model_{model}.txt", f"{top}/requirements/model_{model}.txt")
        z.write(ROOT / ".env.example", f"{top}/.env.example")
        z.writestr(f"{top}/run_{model}.ps1", RUN_PS1.format(model=model, name=name, family=FAM2[model]))
        z.writestr(f"{top}/README.md", README.format(model=model, name=name, family=FAM2[model],
                                                    upper=model.upper(), extra=extra))
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["rf", "mlp"]:
        package(m)
