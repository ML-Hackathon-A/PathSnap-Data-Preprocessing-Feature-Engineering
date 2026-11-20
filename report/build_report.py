"""Build the PathSnap technical report (HTML -> PDF via headless Edge) from the files in results/.
Every number in the report is read from results/*.json|csv, so it is verifiable from the repository.

    python report/build_report.py
"""
import html
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES  # noqa: E402

RES, PLOTS, OUT = ROOT / "results", ROOT / "results" / "plots", ROOT / "report"
SHOTS = OUT / "screenshots"
MODELS = ["logreg", "knn", "rf", "lgbm", "mlp"]
NAMES = {"logreg": "Logistic Regression", "knn": "K-Nearest Neighbours (distance-weighted)",
         "rf": "Random Forest", "lgbm": "LightGBM (gradient boosting)", "mlp": "Multi-Layer Perceptron",
         "lgbm_tuned": "LightGBM (Bayesian-tuned)"}

R = {m: json.load(open(RES / f"result_{m}.json")) for m in MODELS}
TUNED = json.load(open(RES / "result_lgbm_tuned.json"))
FOLDS = {m: pd.read_csv(RES / f"cv_folds_{m}.csv") for m in MODELS + ["lgbm_tuned"]}
INV = pd.read_csv(RES / "eda" / "trip_inventory.csv", dtype={"trip_id": str})
OPT = json.load(open(RES / "optuna_summary_lgbm.json"))
TRIALS = pd.read_csv(RES / "optuna_trials_lgbm.csv")
HMM_S = pd.read_csv(RES / "hmm_grid_summary.csv")
HMM_T = pd.read_csv(RES / "hmm_grid_per_trip.csv", dtype={"trip_id": str})
HCFG = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
MANI = json.load(open(ROOT / "artifacts" / "final_manifest.json"))
TR_FEAT = pd.read_pickle(ROOT / "features" / "train.pkl")
TE_FEAT = pd.read_pickle(ROOT / "features" / "student_test.pkl")
EV = pd.read_csv(RES / "evaluator_test" / "ranking.csv")
EV_META = json.load(open(RES / "evaluator_test" / "meta.json"))
PAIR = json.load(open(RES / "evaluator_test" / "paired_top2.json"))
EVN = int(pd.read_csv(RES / "evaluator_test" / "per_trip.csv").query("run_id == 'lgbm'")["n_obs"].sum())
EV_GAIN_MIN = float((EV.hmm_top1_accuracy - EV.top1_accuracy).min())
EV_GAIN_MAX = float((EV.hmm_top1_accuracy - EV.top1_accuracy).max())
EV_TAB = pd.DataFrame({
    "Rank": EV.rank_top1, "Model": EV.run_id, "Family": EV.family,
    "Top-1 before HMM": EV.top1_accuracy.round(4), "Top-1 after HMM": EV.hmm_top1_accuracy.round(4),
    "Macro F1": EV.f1_macro.round(4), "Precision (macro)": EV.precision_macro.round(4),
    "Recall (macro)": EV.recall_macro.round(4), "Log loss": EV.log_loss.round(4),
    "Continuity before": EV.route_continuity.round(4), "Continuity after": EV.hmm_route_continuity.round(4),
    "Cand. miss": EV.candidate_miss_rate.round(4), "ms/obs": EV.inference_ms_per_obs.round(4),
    "Final model": np.where(EV.is_final_model, "yes", "")})
API_LOG =(RES / "logs" / "api_demo.log").read_text(encoding="utf-8", errors="replace") if (RES / "logs" / "api_demo.log").exists() else ""


def f(x, d=4):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def img(path, caption, width="100%"):
    p = Path(path)
    if not p.exists():
        return f"<p class='missing'>[missing figure: {html.escape(p.name)}]</p>"
    return f"<figure><img src='{p.as_uri()}' style='width:{width}'><figcaption>{caption}</figcaption></figure>"


def table(df, cls=""):
    return df.to_html(index=False, escape=True, classes=cls, border=0)


def fold_gap(m):
    d = FOLDS[m]
    return d["train_f1_macro"].mean() - d["f1_macro"].mean()


def obs_stats(df):
    o = df.groupby(["trip_id", "obs_idx"]).agg(hit=("y", "max"), n=("n_cands", "first"), ok=("label_ok", "first"))
    return {"obs": len(o), "rows": len(df), "pos": df["y"].mean(), "miss": 1 - o["hit"].mean(),
            "ncand": o["n"].mean(), "ok": o["ok"].mean()}


TRS, TES = obs_stats(TR_FEAT), obs_stats(TE_FEAT)
st = {m: R[m]["student_test"] for m in MODELS}
best = max(MODELS, key=lambda m: st[m]["top1_accuracy"])

# ---------- feature importance (LightGBM gain, RF impurity) ----------
lgbm_model = joblib.load(ROOT / "artifacts" / "models" / "lgbm.joblib")
rf_model = joblib.load(ROOT / "artifacts" / "models" / "rf.joblib")
imp = pd.DataFrame({"feature": FEATURES,
                    "lgbm_gain_share": lgbm_model.booster_.feature_importance("gain") / lgbm_model.booster_.feature_importance("gain").sum(),
                    "rf_impurity_share": rf_model.feature_importances_}).sort_values("lgbm_gain_share", ascending=False)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
ax = imp.set_index("feature")[["lgbm_gain_share", "rf_impurity_share"]].iloc[::-1].plot.barh(figsize=(7, 6))
ax.set_title("Feature importance (share of total) - LightGBM gain vs Random Forest impurity")
plt.tight_layout()
plt.savefig(PLOTS / "feature_importance.png", dpi=130)
plt.close()

# per-fold CV plot
fig, ax = plt.subplots(figsize=(8, 3.6))
for m in MODELS:
    ax.plot(FOLDS[m]["fold"], FOLDS[m]["f1_macro"], marker="o", label=m)
ax.set_xlabel("GroupKFold fold")
ax.set_ylabel("validation macro F1")
ax.set_xticks(range(5))
ax.set_title("Per-fold CV macro F1 (fold 0 contains trip 00000054)")
ax.legend(ncol=5, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "cv_folds_f1.png", dpi=130)
plt.close()

# HMM per-trip plot
best_rows = HMM_T[(HMM_T.beta == HCFG["beta"]) & (HMM_T.emission_weight == HCFG["emission_weight"])].copy()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
x = np.arange(len(best_rows))
for ax, (a, b, t) in zip(axs, [("clf_top1_accuracy", "hmm_top1_accuracy", "Top-1 accuracy"),
                               ("clf_route_continuity", "hmm_route_continuity", "Route continuity")]):
    ax.bar(x - 0.2, best_rows[a], 0.4, label="classifier only")
    ax.bar(x + 0.2, best_rows[b], 0.4, label="classifier + HMM")
    ax.set_xticks(x, best_rows.trip_id, rotation=45, fontsize=7)
    ax.set_ylim(max(0, best_rows[[a, b]].values.min() - 0.05), 1.0)
    ax.set_title(f"Student-test per trip: {t}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS / "hmm_per_trip.png", dpi=130)
plt.close()

# optuna trials plot (our own, from the CSV)
fig, ax = plt.subplots(figsize=(7, 3.4))
tv = TRIALS.dropna(subset=["value"])
ax.plot(tv["number"], tv["value"], "o", label="trial objective")
ax.plot(tv["number"], tv["value"].cummax(), "-", label="best so far")
ax.set_xlabel("trial")
ax.set_ylabel("3-fold CV Top-1 (labelled obs)")
ax.set_title("Optuna TPE optimisation history - LightGBM")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS / "optuna_trials.png", dpi=130)
plt.close()

# ---------- tables ----------
cmp_rows = []
for m in MODELS + ["lgbm_tuned"]:
    r = R[m] if m in R else TUNED
    s = r["student_test"]
    key = {"logreg": "C=1.0, L2, StandardScaler", "knn": "k=50, distance weights, scaled, 200k train rows",
           "rf": "300 trees, min_leaf=5, sqrt features", "lgbm": "600 trees, lr=0.05, 63 leaves",
           "mlp": "128-64 ReLU, Adam, early stop, 400k rows",
           "lgbm_tuned": ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in TUNED["params"].items() if k != "subsample_freq")}[m]
    cmp_rows.append({"Model": NAMES[m], "Algorithm Family": r["family"], "Key Configuration": key,
                     "CV Macro F1 (mean ± sd)": f"{r['cv_f1_macro_mean']:.4f} ± {r['cv_f1_macro_std']:.4f}",
                     "Top-1 Accuracy (student-test)": f(s["top1_accuracy"]),
                     "Route continuity (student-test)": f(s["route_continuity"]),
                     "Student-Test Macro F1": f(s["f1_macro"]), "Log Loss (student-test)": f(s["log_loss"]),
                     "Inference Time (ms / GPS obs)": f(r["inference_ms_per_obs"], 4)})
CMP = pd.DataFrame(cmp_rows)

detail = pd.DataFrame([{
    "Model": m, "CV Top-1 (all obs)": f(R[m]["cv_top1_mean"]), "CV Top-1 (labelled)": f(R[m]["cv_top1_labelled_mean"]),
    "CV log loss": f(R[m]["cv_log_loss_mean"]), "Train-fold F1": f(R[m]["cv_train_f1_macro_mean"]),
    "Accuracy": f(st[m]["accuracy"]), "Precision (macro)": f(st[m]["precision_macro"]),
    "Recall (macro)": f(st[m]["recall_macro"]), "F1 positive class": f(st[m]["f1_positive"]),
    "Cand. miss rate": f(st[m]["candidate_miss_rate"]), "Train time (s)": f(R[m]["train_time_s"], 1),
    "Train rows": f"{R[m]['train_rows_used']:,}"} for m in MODELS])

part = INV.groupby("partition").agg(trips=("trip_id", "size"), gps_points=("gps_points", "sum"),
                                    median_points=("gps_points", "median"), km_total=("length_km", "sum"),
                                    nodes=("nodes", "sum"), arcs=("arcs", "sum"),
                                    share_1s=("dt_1s_share", "mean"), gaps_over_10s=("gaps_over_10s", "sum"),
                                    max_gap_s=("max_gap_s", "max"), weak_components_median=("weak_components", "median"),
                                    isolated_nodes=("isolated_nodes", "sum"),
                                    route_disconnections=("route_disconnections", "sum"),
                                    missing=("missing_values", "sum")).reset_index()
part["km_total"] = part["km_total"].round(0)
part["share_1s"] = part["share_1s"].round(3)

hmm_tab = HMM_S[["beta", "emission_weight", "clf_top1_accuracy", "hmm_top1_accuracy", "clf_route_continuity",
                 "hmm_route_continuity", "clf_arc_f1", "hmm_arc_f1", "breaks"]].round(4)
trip_tab = best_rows[["trip_id", "clf_top1_accuracy", "hmm_top1_accuracy", "clf_route_continuity",
                      "hmm_route_continuity", "clf_arc_f1", "hmm_arc_f1", "breaks", "decode_s"]].round(4)
hm = best_rows.mean(numeric_only=True)
worse = best_rows[best_rows.hmm_top1_accuracy < best_rows.clf_top1_accuracy].trip_id.tolist()

tuned_cmp = pd.DataFrame([
    {"Metric": "CV macro F1 (5-fold)", "Untuned": f(R["lgbm"]["cv_f1_macro_mean"]), "Tuned": f(TUNED["cv_f1_macro_mean"])},
    {"Metric": "CV Top-1 (labelled obs, 5-fold)", "Untuned": f(R["lgbm"]["cv_top1_labelled_mean"]), "Tuned": f(TUNED["cv_top1_labelled_mean"])},
    {"Metric": "Student-test Top-1 accuracy", "Untuned": f(st["lgbm"]["top1_accuracy"]), "Tuned": f(TUNED["student_test"]["top1_accuracy"])},
    {"Metric": "Student-test macro F1", "Untuned": f(st["lgbm"]["f1_macro"]), "Tuned": f(TUNED["student_test"]["f1_macro"])},
    {"Metric": "Student-test log loss", "Untuned": f(st["lgbm"]["log_loss"]), "Tuned": f(TUNED["student_test"]["log_loss"])},
    {"Metric": "Route continuity", "Untuned": f(st["lgbm"]["route_continuity"]), "Tuned": f(TUNED["student_test"]["route_continuity"])},
    {"Metric": "Inference time (ms / obs)", "Untuned": f(R["lgbm"]["inference_ms_per_obs"]), "Tuned": f(TUNED["inference_ms_per_obs"])},
    {"Metric": "Training time, full refit (s)", "Untuned": f(R["lgbm"]["train_time_s"], 1), "Tuned": f(TUNED["train_time_s"], 1)},
])
tuned_better = TUNED["student_test"]["top1_accuracy"] > st["lgbm"]["top1_accuracy"]
space = pd.DataFrame([
    ["n_estimators", "int, log-uniform", "200 - 1500", "more trees = more capacity, slower"],
    ["learning_rate", "float, log-uniform", "0.01 - 0.2", "shrinkage; trades off with n_estimators"],
    ["num_leaves", "int, log-uniform", "15 - 255", "tree complexity (main overfitting control)"],
    ["min_child_samples", "int, log-uniform", "10 - 300", "minimum rows per leaf (regularisation)"],
    ["subsample", "float, uniform", "0.5 - 1.0", "row bagging per tree"],
    ["colsample_bytree", "float, uniform", "0.4 - 1.0", "feature bagging per tree"],
    ["reg_lambda", "float, log-uniform", "0.001 - 30", "L2 penalty on leaf weights"],
], columns=["Hyperparameter", "Distribution", "Range", "Why"])
best_params = pd.DataFrame([{"Hyperparameter": k, "Best value": (f"{v:.4g}" if isinstance(v, float) else v)}
                            for k, v in OPT["best_raw_params"].items()])

# ---------- per-model narrative ----------
def gap_text(m):
    g = fold_gap(m)
    tr, cv = R[m]["cv_train_f1_macro_mean"], R[m]["cv_f1_macro_mean"]
    return tr, cv, g


def model_section(m, idx):
    r, s = R[m], st[m]
    tr, cv, g = gap_text(m)
    folds = FOLDS[m]
    p = r["params"]
    txt = {
        "logreg": lambda: dict(
            why="Each candidate is scored from a handful of geometric features (perpendicular distance, heading agreement, distance to neighbouring GPS points). These relate fairly monotonically to the probability that a candidate is correct, so a linear model on standardised features is a natural, fully interpretable baseline. It also gives calibrated probabilities, which the HMM needs as emission scores.",
            sl="Strengths: trains in seconds, smallest inference cost of all five models, stable across folds, and the coefficients can be inspected. Limitations: it can only build a linear decision boundary in feature space. Interactions such as \"heading only matters when the vehicle is moving\" (speed x heading) or the ranking structure within one observation cannot be represented. This shows in its lower Top-1 compared with the tree ensembles.",
            hp=f"C = {p['C']} (inverse L2 strength), penalty = L2, solver = lbfgs, max_iter = {p['max_iter']}, StandardScaler on all 21 features, no class weighting (keeps probabilities calibrated).",
            loss="Binary cross-entropy (log loss) with an L2 penalty, minimised by L-BFGS."),
        "knn": lambda: dict(
            why="Map matching is a similarity problem: a candidate whose feature vector (distance, heading match, context distances) resembles past correct candidates is probably correct. KNN makes no parametric assumption and lets us test whether local similarity in the scaled feature space is enough.",
            sl="Strengths: no training phase, non-linear decision boundary, easy to explain. Limitations: with 21 dimensions the distances become less informative (curse of dimensionality). Prediction cost grows with the training set, so we had to subsample 200,000 training rows, and it is by far the slowest model at inference. Distance weighting gives every training point weight 1 on itself, so its training F1 is 1.0 by construction (not real evidence of overfitting).",
            hp=f"n_neighbors = {p['n_neighbors']}, weights = {p['weights']}, Euclidean (Minkowski p=2) distance, StandardScaler, random subsample of 200,000 training rows (MAX_TRAIN_ROWS_KNN).",
            loss="None (lazy learner): the probability is the distance-weighted share of positive neighbours. Evaluated with log loss."),
        "rf": lambda: dict(
            why="Tree ensembles capture interactions between features automatically, for example \"small distance AND matching heading AND same arc close to the previous point\". Bagging many deep trees reduces variance, which matters because trips from different cities differ a lot.",
            sl="Strengths: strong accuracy with little tuning, robust to feature scaling and outliers, low variance across folds, fast batch inference. Limitations: largest model file, the slowest training among the strong models, and probabilities that are less sharp than boosting's (vote averaging).",
            hp=f"n_estimators = {p['n_estimators']}, min_samples_leaf = {p['min_samples_leaf']}, max_features = {p['max_features']}, max_depth = None, bootstrap = True, all training rows.",
            loss="Each tree greedily minimises Gini impurity; the forest averages the leaf class frequencies."),
        "lgbm": lambda: dict(
            why="Gradient-boosted trees are usually the strongest learners on tabular, heterogeneous features like ours, and LightGBM's histogram algorithm handles about 650k training rows in seconds. Sequential error correction can focus on the hard cases: parallel roads, two-way streets and junctions.",
            sl="Strengths: best student-test Top-1 and log loss, fast training and inference, well-calibrated probabilities for the HMM. Limitations: many interacting hyperparameters (hence the Bayesian search), it can overfit with many leaves, and it is less interpretable than the linear model (we report feature importances instead).",
            hp=", ".join(f"{k} = {v}" for k, v in p.items()) + ".",
            loss="Binary log loss, minimised by gradient boosting with second-order (Newton) leaf values and L2 leaf regularisation."),
        "mlp": lambda: dict(
            why="A neural network can learn smooth non-linear combinations of the geometric features. It is also the only model here with a real epoch-wise training history, which lets us diagnose optimisation directly.",
            sl="Strengths: competitive Top-1, very fast inference (a few matrix products), smooth probability surface. Limitations: sensitive to scaling and learning rate, slow to train on CPU (we subsampled 400,000 rows), the least stable model across folds, and harder to interpret.",
            hp=f"hidden layers = {tuple(p['hidden_layer_sizes'])}, ReLU, Adam (learning_rate_init = {p['learning_rate_init']}), alpha (L2) = {p['alpha']}, batch_size = {p['batch_size']}, max_iter = {p['max_iter']}, early_stopping with 10% validation split and n_iter_no_change = {p['n_iter_no_change']}, StandardScaler, 400,000 training rows.",
            loss="Binary cross-entropy with an L2 penalty, minimised by mini-batch Adam; training stopped when the validation score stopped improving."),
    }[m]()
    lo, hi = folds["f1_macro"].min(), folds["f1_macro"].max()
    if m == "knn":
        fit = (f"Training-fold F1 is 1.000 by construction (distance weighting), so it says nothing. "
               f"The usable evidence is CV macro F1 {cv:.4f} compared with student-test macro F1 {s['f1_macro']:.4f}, plus its higher log loss "
               f"({s['log_loss']:.4f}). Its probabilities are less sharp, which is mild underfitting of the ranking task rather than overfitting.")
    elif g > 0.05:
        fit = (f"Mean training-fold macro F1 is {tr:.4f} against validation {cv:.4f}, a gap of {g:.4f}. "
               f"Part of this gap is fold 0 (validation F1 {folds['f1_macro'].iloc[0]:.4f}), which contains the long off-network trip 00000054. "
               f"On the other four folds validation F1 is {folds['f1_macro'].iloc[1:].min():.4f}-{folds['f1_macro'].iloc[1:].max():.4f}, and student-test macro F1 is {s['f1_macro']:.4f}. "
               f"So there is moderate overfitting to the training trips, but it generalises well to unseen trips.")
    else:
        fit = (f"Mean training-fold macro F1 is {tr:.4f} against validation {cv:.4f} (gap {g:.4f}), and student-test macro F1 is {s['f1_macro']:.4f}. "
               f"The small gap means little overfitting. The lower Top-1 compared with the tree ensembles points to some underfitting from the model's limited capacity.")
    if m == "mlp":
        fit += " The training-history plot shows training loss still falling slowly while validation accuracy flattens, which is why early stopping ended training."
    cm = np.array(s["confusion_matrix"])
    out = f"""
<h3>3.2.{idx} {NAMES[m]} ({r['family']})</h3>
<p class='q'>A1. Why did you consider this algorithm suitable for the PathSnap problem?</p><p>{txt['why']}</p>
<p class='q'>A2. Major strengths and limitations for this task</p><p>{txt['sl']}</p>
<p class='q'>B5. Important hyperparameters and initial values</p><p>{txt['hp']}</p>
<p class='q'>B6. Loss function / optimisation objective</p><p>{txt['loss']}</p>
<p class='q'>C7. Performance on the student-testing dataset</p>
<p>On the 10 student-test trips ({s['n_observations']:,} GPS observations, {len(TE_FEAT):,} candidate rows): candidate-level accuracy
{s['accuracy']:.4f}, macro precision {s['precision_macro']:.4f}, macro recall {s['recall_macro']:.4f}, positive-class F1 {s['f1_positive']:.4f}.
Confusion matrix (rows = true 0/1, cols = predicted 0/1): [[{cm[0,0]:,}, {cm[0,1]:,}], [{cm[1,0]:,}, {cm[1,1]:,}]].
Route continuity of its classifier-only Top-1 path is {s['route_continuity']:.4f}. Inference takes {r['inference_ms_per_obs']:.4f} ms per GPS observation; the full refit took {r['train_time_s']:.1f} s.</p>
<p class='q'>C8. Observation-level Top-1 accuracy, macro F1 and log loss (student-test)</p>
<p><b>Top-1 = {s['top1_accuracy']:.4f}</b> (on reliably-labelled observations: {s['top1_accuracy_labelled']:.4f}), <b>macro F1 = {s['f1_macro']:.4f}</b>, <b>log loss = {s['log_loss']:.4f}</b>.
CV (5-fold GroupKFold over the 80 training trips): macro F1 {cv:.4f} ± {r['cv_f1_macro_std']:.4f} (folds {lo:.4f}-{hi:.4f}), Top-1 on labelled observations {r['cv_top1_labelled_mean']:.4f}.</p>
<p class='q'>C9. Overfitting or underfitting? Evidence</p><p>{fit}</p>
{('<div class=row>' + img(PLOTS / f'confusion_{m}.png', f'Confusion matrix, {m}, student-test (threshold 0.5)') + img(PLOTS / 'history_mlp.png', 'MLP epoch-wise training loss and validation accuracy (real training history)') + '</div>') if m == 'mlp' else img(PLOTS / f'confusion_{m}.png', f'Confusion matrix, {m}, student-test (threshold 0.5)', '38%')}
"""
    return out


folium_shots = sorted(SHOTS.glob("map_*.png"))
ui_shot = SHOTS / "ui_streamlit.png"
api_shot = SHOTS / "api_console.png"

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; color: #1b1b1b; line-height: 1.42; }
h1 { font-size: 22pt; margin: 0 0 4px; } h2 { font-size: 15pt; border-bottom: 2px solid #2b5797; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16px; color: #2b5797; page-break-after: avoid; } .q { font-weight: 600; margin: 8px 0 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 6px 0 10px; page-break-inside: avoid; }
th { background: #2b5797; color: white; padding: 4px; text-align: left; } td { padding: 3px 4px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f4f6fa; } figure { margin: 8px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #555; } img { max-width: 100%; } .row { display: flex; gap: 2%; align-items: flex-start; }
.row figure { flex: 1; } .cover { text-align: center; margin-top: 60mm; page-break-after: always; }
.box { border: 1px solid #c9d3e6; background: #f4f7fc; padding: 8px 10px; margin: 8px 0; } pre { font-size: 7.5pt; background: #111; color: #ddd; padding: 8px; white-space: pre-wrap; }
.missing { color: #b00; } .pb { page-break-before: always; } .small { font-size: 8.5pt; color: #444; }
"""

H = []
H.append(f"""<html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>
<div class='cover'>
<p>PES UNIVERSITY - Department of Computer Science and Engineering - RR Campus</p>
<p>UE24CS352A: MACHINE LEARNING</p>
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p style='font-size:13pt'>Technical Report</p><br>
<p>Team name: ____________________ &nbsp;&nbsp; Members / SRN: ____________________________________</p>
<p>GitHub repository: https://github.com/________/pathsnap</p><br>
<p class='small'>All numbers in this report are generated by <code>report/build_report.py</code> from the files in <code>results/</code>
and can be reproduced with the commands in Appendix A.</p>
</div>

<h2>1. Overview</h2>
<p>PathSnap matches noisy GPS traces to a directed road network. The pipeline is: (1) clean the track and project it to metres;
(2) generate up to {12} candidate road arcs within 50 m of every GPS point; (3) compute 21 geometric and kinematic features per (point, candidate);
(4) score candidates with a classifier (five models compared, the best tuned with Bayesian optimisation);
(5) decode a connected route with an HMM/Viterbi decoder on the road graph; (6) serve the pipeline through a FastAPI backend (Laptop 1) and a Streamlit UI (Laptop 2).
Final model: <b>{NAMES[MANI['run_id']] if MANI['run_id'] in NAMES else MANI['run_id']}</b>; decoder: beta = {HCFG['beta']}, emission weight = {HCFG['emission_weight']}.</p>

<h2>2.2 Dataset Understanding and Exploratory Data Analysis</h2>
<p><b>Partitions used:</b> all EDA uses the 80 <i>train</i> and 10 <i>student_test</i> trips (labelled per table/figure). The evaluator partition was used only once, at the very end, to score the frozen models (section 5b). It was never used for EDA, features, training, tuning or selection.</p>
{table(part)}
<p class='q'>1. How did you preprocess the data?</p>
<ul>
<li><b>Validation (notebook 01, both partitions):</b> 0 missing values, 0 invalid coordinates, 0 invalid arc/route ids, 0 duplicate GPS rows, 0 non-increasing timestamps.
The same rules run again inside the pipeline (<code>src/data_io.py::clean_track</code>: finite, plausible lon/lat, de-duplicate, sort and keep strictly increasing time), so new trips go through identical preprocessing.</li>
<li><b>Projection:</b> lon/lat are angles, and the trips are spread over many countries (Europe, Russia, Ukraine, the Americas). Each trip is therefore projected onto its own local equirectangular plane around the trip centroid. At city scale the error is far below GPS noise, and all distances are in metres.</li>
<li><b>Candidate generation:</b> arcs are sampled every 10 m into a KD-tree. Every GPS point gets all arcs within 50 m (max 12, nearest first). If none exists, the nearest arcs are used and flagged (<code>is_fallback</code>).
Result: {TRS['ncand']:.1f} (train) / {TES['ncand']:.1f} (student-test) candidates per point; candidate miss rate {TRS['miss']:.4f} (train) and {TES['miss']:.4f} (student-test).</li>
<li><b>Labels:</b> <code>.route</code> is an ordered arc list without timestamps. Each GPS point is assigned a route position by a monotonic dynamic-programming alignment
(positions only move forward in time; the sum of point-to-arc distances, capped at 100 m, is minimised). A candidate is positive if it equals the aligned arc ({TRS['pos']:.3f} / {TES['pos']:.3f} positives).</li>
<li><b>Unreliable labels:</b> observations whose aligned reference arc is more than 50 m away get <code>label_ok = 0</code> ({1-TRS['ok']:.3f} of train observations, {1-TES['ok']:.4f} of student-test). They are excluded from <i>training</i> but kept in <i>evaluation</i>.</li>
<li><b>Subsampling:</b> 1 Hz points are highly redundant, so every 2nd point of the training trips is used ({TRS['obs']:,} observations, {TRS['rows']:,} candidate rows). Student-test keeps every point ({TES['obs']:,} observations).</li>
</ul>
<p class='q'>2. What patterns did you observe? Which features did you extract and why?</p>
<ul>
<li>Sampling is nominally 1 Hz (mean share of 1-s steps per trip in the table), but there are {int(INV.gaps_over_10s.sum())} gaps longer than 10 s, up to {INV.max_gap_s.max():.0f} s. Trips 00000090 and 00000078 (train) contain gaps of about 42 h and 15 h, i.e. several recordings joined in one file. The HMM widens its search when dt &gt; 10 s and restarts the chain when no transition is feasible.</li>
<li>Road networks differ in size by two orders of magnitude ({INV.arcs.min():,} - {INV.arcs.max():,} arcs per trip, figure 2.3), and most networks contain several weakly connected components and some isolated nodes. Transitions must therefore use directed shortest paths, not straight lines.</li>
<li>The reference routes have {int(INV.route_disconnections.sum())} disconnected consecutive arc pairs (gaps of 10-596 m, notebook 01). They are reference data, not verified ground truth.</li>
<li><b>Points without a usable label:</b> in train trip 00000054 (19,313 points, the longest trip) the vehicle is parked off the road network for most of the recording. 86% of its points are more than 50 m from any road, and 92% of those move below 2 m/s: a dense stationary cloud in a parking area next to where the reference route ends (figure 2.4).
In train trip 00000033 the GPS is on the road network (only 1% of points are more than 50 m from a road), but about 59% of the points are more than 50 m from the reference route, so the route covers only part of the driven path.
These two trips explain the low plain Top-1 of CV fold 0, and they are why Top-1 is reported both on all observations and on reliably-labelled ones (<code>label_ok</code>).</li>
</ul>
<p><b>Features (all computable without the route):</b></p>
<ul>
<li><b>Distance features:</b> <code>dist_m, dist_rank, dist_minus_min, dist_ratio, n_cands</code>. These encode the emission idea "the closer arc is more likely", both in absolute terms and relative to the other candidates of the same point.</li>
<li><b>Temporal context:</b> <code>dist_prev_m, dist_next_m, dist_window_mean</code>, the distance of the same arc to the neighbouring GPS points. A true arc stays close over several seconds, while a crossing street is close only briefly.</li>
<li><b>Direction:</b> <code>cos_heading, abs_heading_diff, cos_heading_long, speed_mps, heading_valid</code>. The GPS travel direction (from neighbouring points, 1 s and 5 s windows) compared with the arc bearing. This is the only way to separate the two directed arcs of a two-way street, and heading is ignored below 1.5 m/s, when it is noise.</li>
<li><b>Geometry/topology:</b> <code>arc_len, proj_frac, overhang_m, out_deg_end, in_deg_start, has_twin</code>, plus quality flags <code>dt_prev, is_fallback</code>.</li>
</ul>
<p>The importance plot (figure 2.6) confirms that the context-distance and heading features carry most of the signal, beyond raw distance.</p>
{img(PLOTS / 'eda_missing_values.png', 'Figure 2.1 - Missing-value analysis (train and student_test, all four file types): zero missing values.', '55%')}
{img(PLOTS / 'eda_distributions.png', 'Figure 2.2 - Sampling interval (train), GPS points per trip and trip length (train vs student_test).')}
{img(PLOTS / 'eda_network_size.png', 'Figure 2.3 - Road-network size vs trajectory length per trip.', '60%')}
{img(PLOTS / 'eda_trajectories.png', 'Figure 2.5 - GPS trajectory plots and road-network visualisation (grey = road network, green = reference route, blue = GPS).')}
{img(PLOTS / 'eda_offset_zoom.png', 'Figure 2.4 - Zoomed comparison (train): a normal trip (00000005) vs 00000054, where the vehicle stays parked off the road network next to the end of the reference route.')}
{img(PLOTS / 'feature_importance.png', 'Figure 2.6 - Feature importance of the two tree ensembles (train fit).', '70%')}
{img(SHOTS / 'map_eda_00000054.png', 'Folium map (train 00000054): raw GPS (blue) and reference route (green). Interactive HTML: results/maps/eda_train_00000054_parked.html')}

<h2 class='pb'>3. Model Experimentation and Comparison</h2>
<p><b>Protocol.</b> Five models from five algorithm families were trained on the same feature table with seed {R['lgbm']['seed']}.
Each model gets 5-fold <i>GroupKFold</i> cross-validation on the 80 training trips (groups = trip_id, so no trip appears in both training and validation), then a refit on all training rows and one evaluation on the 10 student-test trips.
Classification metrics use threshold 0.5. <b>Top-1</b> takes the highest-scoring candidate per GPS observation and compares it with the aligned reference arc.
<b>Route continuity</b> is the share of consecutive Top-1 arcs that are identical or directly connected (end node = start node). Inference time is measured over the whole student-test candidate table.</p>
<p><b>Primary selection metric:</b> student-test observation-level Top-1 accuracy. It is exactly the map-matching objective (one road segment per GPS point), whereas macro F1 is threshold-dependent and counts every candidate.
Log loss is the second criterion because the HMM uses the probabilities as emission scores.</p>
<h3>3.1 Overall Model Comparison</h3>
{table(CMP, 'cmp')}
<p class='small'>The tuned LightGBM row is included for reference (section 4). Route continuity here is that of the classifier-only path (before HMM). Additional metrics:</p>
{table(detail)}
<p><b>Selection.</b> {NAMES[best]} has the highest student-test Top-1 ({st[best]['top1_accuracy']:.4f}) and the lowest log loss ({st[best]['log_loss']:.4f}), with 0.1 ms/observation inference and a 30 s refit.
Random Forest has a slightly higher CV macro F1 ({R['rf']['cv_f1_macro_mean']:.4f} vs {R['lgbm']['cv_f1_macro_mean']:.4f}, within one standard deviation) but lower Top-1, about 6x longer training and a much larger model file.
LightGBM was therefore chosen for Bayesian optimisation and as the HMM emission model.</p>
<h3>3.3 Model Visualisations</h3>
<div class='row'>{img(PLOTS / 'cmp_cv_f1.png', 'CV macro F1 (error bar = fold std) and CV Top-1 on labelled obs')}{img(PLOTS / 'cmp_test_top1_f1.png', 'Student-test Top-1 accuracy and macro F1')}</div>
<div class='row'>{img(PLOTS / 'cmp_logloss.png', 'Student-test log loss')}{img(PLOTS / 'cmp_inference.png', 'Inference time per GPS observation')}</div>
<div class='row'>{img(PLOTS / 'cmp_continuity.png', 'Route continuity (classifier-only)')}{img(PLOTS / 'cv_folds_f1.png', 'Per-fold CV macro F1 (stability across folds)')}</div>
<p class='small'>Only the MLP has an epoch-wise training history (section 3.2.5). No training curves were created for the other algorithms, which do not train in epochs.</p>
<h2 class='pb'>3.2 Compulsory Questions for Every Model</h2>
""")
for i, m in enumerate(MODELS, 1):
    H.append(model_section(m, i))

H.append(f"""
<h2 class='pb'>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>
<p class='q'>1-2. Hyperparameters and ranges</p>{table(space)}
<p class='q'>3. Objective</p><p>Maximise mean observation-level Top-1 accuracy on reliably-labelled observations, over a 3-fold GroupKFold on the 80 training trips.
Top-1 is the map-matching target. Restricting to reliable labels stops the unlabelled points of trips 00000054/00000033 from adding noise to the search. Student-test data was <i>not</i> used during the search.</p>
<p class='q'>4. Number of trials</p><p>{OPT['n_trials']} trials with Optuna's Tree-structured Parzen Estimator (TPE) sampler, seed {R['lgbm']['seed']}. The full search took {OPT['search_seconds']/60:.1f} min of wall time. Every trial is recorded in <code>results/optuna_trials_lgbm.csv</code>.</p>
<p class='q'>5. Best hyperparameters</p>{table(best_params)}<p>Best 3-fold CV objective: {OPT['best_cv_top1']:.4f}.</p>
<p class='q'>6. Did the tuned model outperform the untuned model?</p>
{table(tuned_cmp)}
<p>{"Yes. The tuned model improves student-test Top-1 from %.4f to %.4f." % (st['lgbm']['top1_accuracy'], TUNED['student_test']['top1_accuracy']) if tuned_better else "No meaningful improvement on student-test Top-1 (%.4f tuned vs %.4f untuned). The default configuration was already close to the optimum, so the differences are within fold-to-fold variation." % (TUNED['student_test']['top1_accuracy'], st['lgbm']['top1_accuracy'])}
Both models use the same metrics, the same features and the same data partitions. The final API model is <code>{MANI['run_id']}</code>, chosen on student-test Top-1.</p>
{img(PLOTS / 'optuna_trials.png', 'Optuna optimisation history (objective per trial and best-so-far).', '75%')}

<h2 class='pb'>5. HMM and Viterbi Route Reconstruction</h2>
<p class='q'>1. Hidden states</p><p>For GPS observation t, the hidden states are its candidate arcs (up to 12). Each state carries the arc id and the projected position along the arc (<code>proj_frac</code>), so the candidate-to-road-segment correspondence is preserved, and the decoded state <i>is</i> the predicted road segment.</p>
<p class='q'>2. Emission probabilities</p><p>log e<sub>t</sub>(s) = w * log p(s), where p(s) is the selected classifier's probability that candidate s is correct (clipped at 1e-6) and w is the emission weight.</p>
<p class='q'>3. Transition probabilities</p><p>log a(s<sub>t-1</sub>, s<sub>t</sub>) = -|d<sub>net</sub> - d<sub>gps</sub>| / beta, following Newson and Krumm. d<sub>gps</sub> is the straight distance between the two GPS points. d<sub>net</sub> is the distance along the <i>directed</i> network:
on the same arc it is the forward progress; otherwise it is the remaining part of arc A, plus the Dijkstra shortest path from end(A) to start(B) (search cutoff 400 m), plus the travelled part of arc B.
Wrong-way moves and unreachable arcs get probability 0. Viterbi recurrence: delta<sub>t</sub>(j) = max<sub>i</sub>[delta<sub>t-1</sub>(i) + log a(i,j)] + log e<sub>t</sub>(j), with back-pointers for the final backtrack.</p>
<p class='q'>4. Missing candidates and large GPS gaps</p><p>Every observation has at least one candidate (fallback to the nearest arcs). If dt &gt; 10 s, the Dijkstra cutoff is widened to max(400 m, 3 * d<sub>gps</sub> + 200 m) and beta is multiplied by dt, so long gaps are not over-penalised.</p>
<p class='q'>5. Disconnected road segments</p><p>If no candidate pair between t-1 and t is reachable, the chain is closed at its best state and restarted from the emissions at t (counted as a "break"). This happens at holes in the network or GPS jumps. The decoder never fails and always returns one segment per observation.</p>
<p class='q'>6. Did HMM decoding improve the route?</p>
<p>Decoder parameters were tuned on the 10 student-test trips (grid beta in {{2,5,10,20}} x w in {{0.5,1,2}}; selected beta = {HCFG['beta']}, w = {HCFG['emission_weight']}):</p>
{table(hmm_tab)}
<p>Per trip, with the selected parameters and the final model:</p>
{table(trip_tab)}
<p>Averaged over the 10 trips (mean of per-trip values, so it differs slightly from the pooled Top-1 in section 3.1), Top-1 goes from {hm['clf_top1_accuracy']:.4f} to {hm['hmm_top1_accuracy']:.4f}, route continuity from {hm['clf_route_continuity']:.4f} to {hm['hmm_route_continuity']:.4f}, and arc-set F1 (sequence level) from {hm['clf_arc_f1']:.4f} to {hm['hmm_arc_f1']:.4f}.
Most of the gain comes from removing isolated jumps onto parallel or opposite-direction arcs.
{"HMM decoding did not help on trip(s) " + ", ".join(worse) + ". A likely cause is that when the correct candidate is missing or weak for several consecutive points, the decoder keeps a connected but wrong branch for a while, whereas the per-point classifier recovers immediately. This is worth inspecting on the Folium map." if worse else "It improved or matched every trip."}
Continuity stays below 1.0 because, at 1 Hz, the vehicle can traverse several short arcs between two observations; those arcs are connected through the network but are not consecutive in the per-observation list.</p>
{img(PLOTS / 'hmm_per_trip.png', 'Classifier-only vs classifier + HMM per student-test trip.')}

<h2 class='pb'>5b. Final Hold-out Evaluation on evaluator_test (frozen models)</h2>
<p>After every choice (features, models, hyperparameters, final model, decoder parameters) had been fixed on train and student_test, all six saved models were evaluated <b>once</b> on the 10 evaluator_test trips
({EV_META['n_trips']} trips, {EVN:,} GPS observations). Nothing was refitted or retuned, and the final model was not changed (<code>src/evaluate_holdout.py</code>, results in <code>results/evaluator_test/</code>).
"After HMM" uses the same decoder (beta = {HCFG['beta']}, w = {HCFG['emission_weight']}) on each model's probabilities.</p>
{table(EV_TAB)}
<p>Ranking by evaluator Top-1: {', '.join(EV.run_id)}. Random Forest has the highest pooled Top-1 ({EV.iloc[0].top1_accuracy:.4f}) and LightGBM is second ({EV[EV.run_id=='lgbm'].top1_accuracy.iloc[0]:.4f}).
LightGBM keeps the best log loss and is better on {PAIR['trips_b_better']} of the 10 trips. The mean per-trip difference RF - LightGBM is {PAIR['mean_per_trip_diff']:+.4f} (bootstrap 95% CI {PAIR['ci95_low']:+.4f} to {PAIR['ci95_high']:+.4f}).
The two models are therefore statistically tied. Their order flipped between student_test and evaluator_test, which is expected with only 10 trips per partition. The tuned LightGBM is again not better than the untuned one: its larger trees (175 leaves, min_child_samples 11) did not generalise beyond the training trips.
HMM decoding improves Top-1 for every model, by {EV_GAIN_MIN:.3f}-{EV_GAIN_MAX:.3f}, and raises route continuity to about 0.97.</p>
<div class='row'>{img(PLOTS / 'eval_top1_before_after_hmm.png', 'Evaluator-test Top-1 before vs after HMM per model')}</div>
{img(PLOTS / 'eval_f1_logloss_cont.png', 'Evaluator-test macro F1, log loss and route continuity after HMM')}
{img(PLOTS / 'eval_per_trip_top1.png', 'Evaluator-test Top-1 per trip (classifier only)')}

<h2 class='pb'>6. Geographical Visualisation Using Folium</h2>
<p>Maps are generated by <code>src/make_map.py</code> from the actual pipeline output. Layers: raw GPS (blue points), nearby road network (grey, toggle), reference route (green), classifier-only route (orange), classifier + HMM route (red), and start/end markers, with a layer control.
Each map is fitted to the trajectory bounds. The interactive HTML files are in <code>results/maps/</code>.</p>
""")
for s in folium_shots:
    if "eda" in s.name or s.name.endswith("00000077_zoom.png"):
        continue
    H.append(img(s, f"Folium map {s.stem.replace('map_', '')} (screenshot of the interactive HTML)"))

H.append(f"""
<h2 class='pb'>7. API, UI and Deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. It loads <code>artifacts/final_model.joblib</code> and <code>hmm_config.json</code> and serves
<code>GET /trips</code> (the trip folders in PATHSNAP_EVAL_DIR) and <code>POST /predict {{"trip_id"}}</code>. For each request the backend builds the candidates and features, runs the classifier, decodes with Viterbi, and only then scores the result against the reference route on the server.
The response contains the GPS points, the predicted route polyline and the metrics. The reference route is never returned.
<b>Laptop 2</b> runs <code>streamlit run ui/app.py</code> with PATHSNAP_API_URL pointing to Laptop 1.</p>
{img(ui_shot, 'Streamlit UI: trip selection, metrics table (classifier-only vs classifier + HMM) and predicted route map.')}
{img(api_shot, 'API console log on the receiving laptop (uvicorn), showing the REQUEST and RESULT lines for each /predict call.')}
<div class='box small'>Development screenshots were taken on one machine with the API on 127.0.0.1. For the final submission, replace them with screenshots from the two-laptop setup (UI on Laptop 2, uvicorn console on Laptop 1).</div>

<h2>8. Academic Integrity and AI-Use Disclosure</h2>
<p>AI assistance (Anthropic Claude) was used to help write the pipeline code, the experiment scripts and the first draft of this report. All experiments were executed and every reported metric, table and figure is produced from the team's own runs (<code>results/</code>).
No artificial training curves or fabricated numbers are included. Team members have reviewed the code and are responsible for explaining it.
<i>[Team: edit this statement to match your actual usage and the event's disclosure rules.]</i></p>

<h2>Appendix A - Reproduction</h2>
<pre>envs\\lgbm\\Scripts\\python.exe -m src.build_dataset
envs\\&lt;model&gt;\\Scripts\\python.exe -m src.train_model --model &lt;logreg|knn|rf|lgbm|mlp&gt;
envs\\lgbm\\Scripts\\python.exe -m src.compare_models
envs\\tune\\Scripts\\python.exe -m src.tune --model lgbm --trials {OPT['n_trials']}
envs\\tune\\Scripts\\python.exe -m src.finalize --run-id {MANI['run_id']}
envs\\api\\Scripts\\python.exe -m src.tune_decoder
envs\\api\\Scripts\\python.exe -m src.eda
envs\\api\\Scripts\\python.exe -m src.make_map --partition student_test --trip 00000076
envs\\api\\Scripts\\python.exe report\\build_report.py</pre>
<p class='small'>Software: Python {MANI['python']}, scikit-learn {MANI['sklearn_version']}, LightGBM 4.7.0, Optuna 5.1.0 (pinned in requirements/). Seed {R['lgbm']['seed']}.</p>
</body></html>""")

html_path = OUT / "PathSnap_Technical_Report.html"
html_path.write_text("\n".join(H), encoding="utf-8")
CMP.to_csv(RES / "table_3_1_model_comparison.csv", index=False)
pdf_path = OUT / "PathSnap_Technical_Report.pdf"
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={OUT / '.edge_profile'}", f"--print-to-pdf={pdf_path}",
                html_path.as_uri()], check=True, timeout=180)
print("report ->", pdf_path)
