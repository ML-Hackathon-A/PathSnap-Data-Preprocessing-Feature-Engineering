"""Bayesian hyperparameter optimisation (Optuna TPE) with group-aware CV on the 80 training trips.
Objective: maximise mean CV observation-level Top-1 accuracy on reliably-labelled observations
(the map-matching target; unreliable reference-route stretches would only add noise to the search).
student_test is NOT used during the search; it is only used afterwards to compare tuned vs untuned.

    python -m src.tune --model lgbm --trials 40
"""
import argparse
import json
import time

import optuna

from .config import ARTIFACT_DIR, OPTUNA_TRIALS, RESULTS_DIR, SEED
from .models import FAMILIES, suggest_params
from .train_model import cross_validate, load_features, run

TUNE_FOLDS = 3  # fewer folds than the main comparison to keep the search affordable


def main(name, n_trials):
    tr, _ = load_features()

    def objective(trial):
        params = suggest_params(name, trial)
        folds, _ = cross_validate(name, params, tr, n_folds=TUNE_FOLDS, verbose=False)
        trial.set_user_attr("cv_f1_macro", float(folds["f1_macro"].mean()))
        trial.set_user_attr("cv_log_loss", float(folds["log_loss"].mean()))
        trial.set_user_attr("fit_s_total", float(folds["fit_s"].sum()))
        trial.set_user_attr("cv_top1_all_obs", float(folds["top1_accuracy"].mean()))
        return float(folds["top1_accuracy_labelled"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name=f"pathsnap_{name}")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(RESULTS_DIR / f"optuna_trials_{name}.csv", index=False)

    best_params = suggest_params(name, optuna.trial.FixedTrial(study.best_params))
    best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}
    json.dump(best_params, open(ARTIFACT_DIR / f"best_params_{name}.json", "w"), indent=2)
    json.dump({"model": name, "n_trials": n_trials, "best_cv_top1": study.best_value,
               "best_raw_params": study.best_params, "search_seconds": search_s},
              open(RESULTS_DIR / f"optuna_summary_{name}.json", "w"), indent=2)
    print(f"best CV top1={study.best_value:.4f} in {search_s / 60:.1f} min; params={best_params}")

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.figure.savefig(RESULTS_DIR / "plots" / f"optuna_history_{name}.png", dpi=130, bbox_inches="tight")
    except Exception as e:  # plotting is optional
        print("optuna plot skipped:", e)

    run(name, best_params, tag="tuned")  # refit on all training trips + evaluate on student_test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FAMILIES))
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    a = ap.parse_args()
    main(a.model, a.trials)
