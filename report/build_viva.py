"""Viva preparation guide (PDF): deliverables explained, pipeline, models, optimisers, evaluator results.
All numbers are read from results/ (evaluator_test results from src/evaluate_holdout.py).

    python report/build_viva.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, PLOTS, OUT, SHOTS = ROOT / "results", ROOT / "results" / "plots", ROOT / "report", ROOT / "report" / "screenshots"
EVD = RES / "evaluator_test"

EV = pd.read_csv(EVD / "ranking.csv")
TRIP = pd.read_csv(EVD / "per_trip.csv", dtype={"trip_id": str})
PAIR = json.load(open(EVD / "paired_top2.json"))
META = json.load(open(EVD / "meta.json"))
R = {m: json.load(open(RES / f"result_{m}.json")) for m in ["logreg", "knn", "rf", "lgbm", "mlp", "lgbm_tuned"]}
OPT = json.load(open(RES / "optuna_summary_lgbm.json"))
HCFG = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
HG = pd.read_csv(RES / "hmm_grid_summary.csv")
INV = pd.read_csv(RES / "eda" / "trip_inventory.csv", dtype={"trip_id": str})
E = EV.set_index("run_id")
NOBS = int(TRIP[TRIP.run_id == "lgbm"].n_obs.sum())
piv = TRIP.pivot(index="trip_id", columns="run_id", values="clf_top1_accuracy")
nobs = TRIP[TRIP.run_id == "lgbm"].set_index("trip_id").n_obs
big = nobs.idxmax()
NAMES = {"logreg": "Logistic Regression", "knn": "KNN (distance-weighted)", "rf": "Random Forest",
         "lgbm": "LightGBM", "mlp": "MLP neural network", "lgbm_tuned": "LightGBM, Bayesian-tuned"}


def f(x, d=4):
    return f"{x:.{d}f}"


def pp(x):
    return f"{100 * x:+.2f} pp"


def img(p, cap, w="100%"):
    p = Path(p)
    return (f"<figure><img src='{p.as_uri()}' style='width:{w}'><figcaption>{cap}</figcaption></figure>"
            if p.exists() else f"<p style='color:#b00'>[missing {p.name}]</p>")


def tbl(rows, head):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def qa(q, a):
    return f"<div class='qa'><p class='q'>Q: {q}</p><p>{a}</p></div>"


# ---------------- tables ----------------
before_after = tbl([[NAMES[r], E.loc[r, "family"], f(E.loc[r, "top1_accuracy"]), f(E.loc[r, "hmm_top1_accuracy"]),
                     pp(E.loc[r, "hmm_top1_accuracy"] - E.loc[r, "top1_accuracy"]),
                     f(E.loc[r, "route_continuity"]), f(E.loc[r, "hmm_route_continuity"]), int(E.loc[r, "rank_top1"])]
                    for r in EV.run_id],
                   ["Model", "Family", "Top-1 BEFORE HMM (classifier only)", "Top-1 AFTER HMM/Viterbi", "Gain",
                    "Continuity before", "Continuity after", "Rank (Top-1 before)"])

bayes = tbl([
    ["CV macro F1 (5-fold, train)", f(R["lgbm"]["cv_f1_macro_mean"]), f(R["lgbm_tuned"]["cv_f1_macro_mean"])],
    ["CV Top-1 on labelled obs (5-fold, train)", f(R["lgbm"]["cv_top1_labelled_mean"]), f(R["lgbm_tuned"]["cv_top1_labelled_mean"])],
    ["<b>Evaluator Top-1, classifier only</b>", f(E.loc["lgbm", "top1_accuracy"]), f(E.loc["lgbm_tuned", "top1_accuracy"])],
    ["<b>Evaluator Top-1, after HMM</b>", f(E.loc["lgbm", "hmm_top1_accuracy"]), f(E.loc["lgbm_tuned", "hmm_top1_accuracy"])],
    ["Evaluator macro F1", f(E.loc["lgbm", "f1_macro"]), f(E.loc["lgbm_tuned", "f1_macro"])],
    ["Evaluator log loss", f(E.loc["lgbm", "log_loss"]), f(E.loc["lgbm_tuned", "log_loss"])],
    ["Inference ms / GPS point (evaluator)", f(E.loc["lgbm", "inference_ms_per_obs"]), f(E.loc["lgbm_tuned", "inference_ms_per_obs"])],
    ["Trees / leaves", "600 / 63", f"{R['lgbm_tuned']['params']['n_estimators']} / {R['lgbm_tuned']['params']['num_leaves']}"],
], ["Metric", "LightGBM untuned (defaults)", "LightGBM after Bayesian optimisation"])

full = tbl([[NAMES[r], f(E.loc[r, "top1_accuracy"]), f(E.loc[r, "f1_macro"]), f(E.loc[r, "precision_macro"]),
             f(E.loc[r, "recall_macro"]), f(E.loc[r, "accuracy"]), f(E.loc[r, "log_loss"]), f(E.loc[r, "candidate_miss_rate"]),
             f(E.loc[r, "inference_ms_per_obs"])] for r in EV.run_id],
           ["Model", "Top-1", "Macro F1", "Precision", "Recall", "Cand. accuracy", "Log loss", "Cand. miss", "ms/obs"])

trip_rows = []
for t in piv.index:
    trip_rows.append([t, f"{nobs[t]:,}", f(piv.loc[t, "rf"]), f(piv.loc[t, "lgbm"]), pp(piv.loc[t, "rf"] - piv.loc[t, "lgbm"]),
                      f(TRIP[(TRIP.run_id == "lgbm") & (TRIP.trip_id == t)].hmm_top1_accuracy.iloc[0])])
trip_tbl = tbl(trip_rows, ["Trip", "GPS points", "RF Top-1", "LightGBM Top-1", "RF - LGBM", "LightGBM + HMM"])

optim_tbl = tbl([
    ["Logistic Regression", "L-BFGS (quasi-Newton)", "Binary cross-entropy + L2 (C=1)", "Convex problem, so L-BFGS finds the single global optimum; uses curvature information for fast convergence."],
    ["KNN", "None (lazy learner)", "None. Prediction = distance-weighted vote of 50 neighbours", "Nothing is fitted; the training rows are stored in a tree index and searched at prediction time."],
    ["Random Forest", "Greedy recursive splitting", "Gini impurity at each split", "Each tree picks the best split at each node from a random subset of features (sqrt(21) = 4); 300 bootstrap trees are averaged."],
    ["LightGBM", "Gradient boosting with Newton steps", "Binary log loss + L2 on leaf weights", "Each new tree fits the gradient (and Hessian) of the loss of the current ensemble; learning rate 0.05 shrinks each step."],
    ["MLP", "Adam (adaptive moment estimation)", "Binary cross-entropy + L2 (alpha=1e-4)", "Mini-batch (1024) gradient descent with per-weight adaptive learning rates; early stopping on a 10% validation split."],
], ["Model", "Training optimiser", "What it minimises", "In one sentence"])

deliv_repo = tbl([
    ["Data processing", "notebooks/01_data_preprocessing.ipynb, src/data_io.py, src/eda.py", "Reads the 4 raw file types, validates them, cleans the track"],
    ["Feature engineering", "src/geo.py, src/features.py, src/build_dataset.py", "Projection, KD-tree candidates, DP label alignment, 21 features"],
    ["Model training", "src/models.py, src/train_model.py", "Defines and trains all 5 models with GroupKFold CV"],
    ["Model evaluation", "src/metrics.py, src/compare_models.py, src/evaluate_holdout.py, src/plot_holdout.py", "All metrics, comparison table, plots, confusion matrices"],
    ["Hyperparameter optimisation", "src/tune.py, results/optuna_trials_lgbm.csv", "Optuna TPE search + full trial log"],
    ["HMM and Viterbi", "src/hmm.py, src/tune_decoder.py", "Route decoding + decoder parameter grid"],
    ["API", "api/main.py, src/pipeline.py", "FastAPI: /health, /trips, /predict"],
    ["UI", "ui/app.py", "Streamlit evaluator interface"],
    ["Deployment", "README.md, requirements/*.txt, .env.example", "Install + run + two-laptop instructions, pinned versions"],
    ["Visualisations", "src/make_map.py, src/eda.py, results/maps/*.html", "Folium maps + all plots"],
    ["Supporting results", "results/*.json|csv, results/evaluator_test/, results/logs/", "Every number in both PDFs"],
], ["Required component (Deliverables PDF)", "Where it is in our repo", "What it does"])

deliv_report = tbl([
    ["2.2 Dataset understanding & EDA", "Preprocessing answers, patterns, features + why, missing-value plot, trajectory plots, road-network plot, Folium map, partitions named", "Done"],
    ["3.1 Overall model comparison", "Table with family, configuration, CV macro F1, Top-1, continuity, student-test F1, log loss, inference time", "Done (+ evaluator table in 5b)"],
    ["3.2 Per-model questions A1-C9", "Five subsections, each answering why, strengths/limits, hyperparameters, loss, results, over/underfitting", "Done"],
    ["3.3 Visualisations", "CV F1, Top-1/F1, log loss, inference, continuity, per-fold plot, confusion matrices, MLP history (only real one)", "Done"],
    ["4 Bayesian optimisation", "Search space, objective, 25 trials, best params, tuned vs untuned", "Done"],
    ["5 HMM / Viterbi", "Hidden states, emission, transition, gaps, disconnections, classifier vs HMM", "Done"],
    ["6 Folium", "GPS, network, reference, predicted routes, legend, layer control, screenshots", "Done"],
    ["7 API / UI / deployment", "UI screenshot, API log screenshot", "Replace with real two-laptop screenshots"],
    ["Cover page", "Team name, SRNs, GitHub link", "<b>You must fill in</b>"],
], ["Report section", "What it must contain / what we put there", "Status"])

demo = tbl([
    ["1. API running on Laptop 1", "envs\\api\\Scripts\\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 (.env: PATHSNAP_EVAL_DIR=./data/eval)"],
    ["2. UI running on Laptop 2", "envs\\ui\\Scripts\\python.exe -m streamlit run ui\\app.py; Laptop 2 .env has PATHSNAP_API_URL=http://&lt;Laptop-1-IP&gt;:8000"],
    ["3-4. Evaluator opens UI, sees 10 trips", "UI calls GET /trips, which lists the trip folders in data/eval"],
    ["5-6. Evaluator picks a trip; UI sends trip id", "POST /predict {\"trip_id\": ...}"],
    ["7. Full pipeline runs on the API", "clean, then candidates + features, then LightGBM probabilities, then Viterbi (src/pipeline.py::predict_trip). The route file is NOT read at this stage."],
    ["8. API computes accuracy server-side", "evaluate_prediction(): aligns the .route to the GPS points and compares (classifier-only and HMM)"],
    ["9. UI shows route + results", "Map (GPS blue, predicted route red) + metrics table. The reference route is never sent to the UI."],
    ["Show API logs", "uvicorn console prints REQUEST /predict trip_id=... and RESULT trip=... top1(hmm)=... lines"],
], ["Demo-day step (Main Guidelines / Models PDF)", "How our system does it"])

integ = tbl([
    ["Evaluator data untouched during development", "Used only once, at the end, on frozen models (src/evaluate_holdout.py). No training, tuning, feature or model choice used it."],
    ["No hard-coded predictions", "Every prediction comes from model.predict_proba + Viterbi at request time"],
    ["Ground truth never in prediction path / UI", "predict_trip() sets route=None; scoring happens afterwards; /predict returns no reference route"],
    ["Fit preprocessing on training data only", "StandardScalers live inside the model pipelines, fitted on the train folds"],
    ["No sharing between teams", "The RF/MLP zips are for our own team members only"],
    ["Disclose AI assistance", "Section 8 of the technical report (edit to match your usage)"],
    ["Every member can explain the work", "This document"],
], ["Rule", "How we comply"])

gain_min = float((EV.hmm_top1_accuracy - EV.top1_accuracy).min())
gain_max = float((EV.hmm_top1_accuracy - EV.top1_accuracy).max())
best_hmm = EV.sort_values("hmm_top1_accuracy", ascending=False).run_id.iloc[0]

model_cards = ""
cards = {
    "rf": f"""<b>Idea:</b> 300 decision trees, each trained on a bootstrap sample of the rows and choosing among 4 random features at every split; the probability is the average of the trees' votes.
<br><b>Why it performed this way:</b> it ranked #1 on evaluator Top-1 ({f(E.loc['rf','top1_accuracy'])}). Trees capture the <i>interactions</i> that decide map matching, e.g. "small distance AND same direction AND also close to the previous GPS point". Averaging many de-correlated trees gives low variance, so it was robust on the hardest, longest trip {big} ({nobs[big]:,} points, {100*nobs[big]/NOBS:.0f}% of all evaluator points), where it beat LightGBM by {pp(piv.loc[big,'rf']-piv.loc[big,'lgbm'])}. That single trip creates its whole pooled lead: on the other trips LightGBM was better {PAIR['trips_b_better']} times out of 10. Downsides: largest model file, and 6x slower to train than LightGBM ({R['rf']['train_time_s']:.0f} s vs {R['lgbm']['train_time_s']:.0f} s for the full refit).""",
    "lgbm": f"""<b>Idea:</b> 600 small trees added one after another; each new tree corrects the remaining errors of the previous ones (gradient boosting).
<br><b>Why it performed this way:</b> #2 on pooled evaluator Top-1 ({f(E.loc['lgbm','top1_accuracy'])}) but the <b>best log loss</b> ({f(E.loc['lgbm','log_loss'])}) and the best model on {PAIR['trips_b_better']} of 10 trips. Boosting focuses on hard cases (parallel roads, two-way streets), and its probabilities are sharp and well calibrated, which is what the HMM needs. That is why it is almost the best <i>after</i> HMM too ({f(E.loc['lgbm','hmm_top1_accuracy'])} vs RF {f(E.loc['rf','hmm_top1_accuracy'])}). It was our final model because it was best on student_test (Top-1 0.8987 vs RF 0.8897) and much faster to train.""",
    "logreg": f"""<b>Idea:</b> a weighted sum of the 21 standardised features passed through a sigmoid.
<br><b>Why it performed this way:</b> surprisingly competitive ({f(E.loc['logreg','top1_accuracy'])}) because our features are already "pre-digested" geometry (distance, heading cosine, context distance) that relate almost monotonically to correctness. It cannot model interactions (e.g. heading only matters when the car moves), so it is below the tree ensembles, and its log loss ({f(E.loc['logreg','log_loss'])}) is clearly worse. Fastest model by far ({f(E.loc['logreg','inference_ms_per_obs'])} ms/point).""",
    "knn": f"""<b>Idea:</b> look at the 50 most similar training candidates in feature space and take their distance-weighted vote.
<br><b>Why it performed this way:</b> second-worst Top-1 ({f(E.loc['knn','top1_accuracy'])}) and the smallest HMM gain ({pp(E.loc['knn','hmm_top1_accuracy']-E.loc['knn','top1_accuracy'])}). With 21 dimensions, "nearest" neighbours are not very near (curse of dimensionality), we could only keep 200,000 training rows, and its probabilities are coarse vote shares, so the HMM receives weaker emission information. It is also the slowest at prediction ({f(E.loc['knn','inference_ms_per_obs'])} ms/point) because it searches the stored rows every time.""",
    "mlp": f"""<b>Idea:</b> two hidden layers (128 and 64 ReLU units) trained with Adam on cross-entropy.
<br><b>Why it performed this way:</b> lowest classifier-only Top-1 ({f(E.loc['mlp','top1_accuracy'])}) but a decent log loss ({f(E.loc['mlp','log_loss'])}) and a large HMM gain ({pp(E.loc['mlp','hmm_top1_accuracy']-E.loc['mlp','top1_accuracy'])}). Neural nets need lots of data and tuning to beat trees on tabular features; we trained on a 400,000-row subsample on CPU and did not tune it. It was also the least stable across CV folds (std {f(R['mlp']['cv_f1_macro_std'])}). Its training curve is real: loss keeps falling while validation accuracy flattens, so early stopping ended it.""",
    "lgbm_tuned": f"""<b>Idea:</b> the same LightGBM with Optuna's best hyperparameters (1133 trees, 175 leaves, min_child_samples 11).
<br><b>Why it performed this way:</b> Top-1 {f(E.loc['lgbm_tuned','top1_accuracy'])}, slightly <i>below</i> untuned. The search maximised 3-fold CV on the training trips; in the 5-fold CV the tuned model is only {(R['lgbm_tuned']['cv_top1_labelled_mean']-R['lgbm']['cv_top1_labelled_mean'])*100:+.2f} pp better than untuned; the chosen trees are much larger, which fits the training cities better but does not transfer to new cities (mild overfitting to the training trips). Lesson for the viva: tuning only helps if the CV objective reflects generalisation, and here the default settings were already near the optimum.""",
}
for r in EV.run_id:
    model_cards += f"<h3>{int(E.loc[r,'rank_top1'])}. {NAMES[r]} <span class='fam'>({E.loc[r,'family']})</span></h3><p>{cards[r]}</p>"

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1d1d1d; }
h1 { font-size: 22pt; margin: 0; } h2 { font-size: 15pt; color: #fff; background: #2b5797; padding: 4px 8px; margin-top: 20px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2b5797; margin: 12px 0 4px; page-break-after: avoid; } .fam { color: #777; font-weight: normal; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 6px 0 10px; page-break-inside: avoid; }
th { background: #e3e9f5; text-align: left; padding: 4px; border-bottom: 2px solid #2b5797; } td { padding: 3px 4px; border-bottom: 1px solid #ddd; vertical-align: top; }
.box { border-left: 4px solid #2b5797; background: #f3f6fb; padding: 6px 10px; margin: 8px 0; }
.warn { border-left: 4px solid #d08000; background: #fff6e6; padding: 6px 10px; margin: 8px 0; }
.qa { margin: 6px 0 8px; page-break-inside: avoid; } .q { font-weight: 600; margin: 0 0 2px; color: #2b5797; }
figure { margin: 6px 0; text-align: center; page-break-inside: avoid; } figcaption { font-size: 8.5pt; color: #555; } img { max-width: 100%; }
.pb { page-break-before: always; } code { background: #f1f1f1; padding: 0 3px; font-size: 9pt; }
.cover { text-align: center; margin-top: 55mm; page-break-after: always; }
"""

lg, lgt, rf = E.loc["lgbm"], E.loc["lgbm_tuned"], E.loc["rf"]
H = f"""<html><head><meta charset='utf-8'><title>PathSnap Viva Guide</title><style>{CSS}</style></head><body>
<div class='cover'><p>UE24CS352A Machine Learning - PathSnap hackathon</p><h1>Viva Preparation Guide</h1>
<p style='font-size:13pt'>Deliverables, models, optimisers and evaluator-test results, explained</p><br>
<p>Final model: <b>LightGBM + HMM/Viterbi</b> &nbsp;|&nbsp; Evaluator-test Top-1: <b>{f(lg['top1_accuracy'])} before HMM, {f(lg['hmm_top1_accuracy'])} after HMM</b></p>
<p style='font-size:9pt;color:#555'>Every number comes from results/ (evaluator numbers from src/evaluate_holdout.py, run once on the frozen models).</p></div>

<h2>1. The 60-second answer</h2>
<div class='box'>"Our system matches noisy 1 Hz GPS points to a directed road network. For every GPS point we take the road arcs within 50 m as candidates and describe each candidate with 21 features: how far it is, whether its direction matches the driving direction, and whether it stays close over neighbouring seconds.
We trained five classifiers from five families to score the candidates, with group-aware cross-validation over the 80 training trips. We picked LightGBM on the student-test trips because it had the best Top-1 accuracy and log loss, and ran Bayesian optimisation with Optuna on it.
Then an HMM with Viterbi decoding combines those probabilities with road connectivity, so the route cannot jump between unconnected roads. On the 10 evaluator trips, LightGBM alone gets {f(lg['top1_accuracy'])} Top-1 and {f(lg['hmm_top1_accuracy'])} after the HMM, with route continuity going from {f(lg['route_continuity'])} to {f(lg['hmm_route_continuity'])}.
Everything runs behind a FastAPI server on one laptop and a Streamlit UI on the other."</div>

<h2>2. The deliverables explained</h2>
<p>The Deliverables PDF asks for exactly <b>two things</b>: (1) a public <b>GitHub repository</b> with all the code, and (2) one <b>technical report PDF</b>. Separately, the Main/Models guidelines define a <b>live two-laptop demo</b> and integrity rules. Here is each part and where we satisfy it.</p>
<h3>2.1 Deliverable 1 - GitHub repository (all components required)</h3>{deliv_repo}
<p>Also required: a README with installation/execution/deployment (README.md), clearly named folders, saved model weights (artifacts/final_model.joblib) and <b>no secrets</b> (.env is git-ignored).</p>
<h3>2.2 Deliverable 2 - Technical report (report/PathSnap_Technical_Report.pdf)</h3>{deliv_report}
<h3>2.3 The live demo (what the evaluator will do)</h3>{demo}
<h3>2.4 Integrity rules and how we comply</h3>{integ}
<div class='warn'><b>Still on your to-do list:</b> fill in team names/SRNs/GitHub link on the report cover; push the repo and make it public; take real two-laptop screenshots (UI on Laptop 2, uvicorn console on Laptop 1) for report section 7; edit the AI-disclosure paragraph to describe your actual use.</div>

<h2 class='pb'>3. Data and preprocessing (what they will ask first)</h2>
<p><b>Files per trip:</b> <code>.track</code> = lon, lat, time (1 Hz GPS); <code>.nodes</code> = lon, lat of road nodes (row number = node id); <code>.arcs</code> = from-node, to-node (row number = arc id, <i>directed</i>, so a two-way street is two arcs); <code>.route</code> = the ordered ground-truth arc ids, <b>without timestamps</b>.
A GPS row does not correspond to a route row; we have to work out which arc each point belongs to.</p>
<p><b>Partitions:</b> train 80 trips (fitting, cross-validation, tuning), student_test 10 (model comparison, decoder tuning), evaluator_test 10 (final scoring only).</p>
<p><b>Preprocessing.</b> The notebook found 0 missing values, 0 invalid ids, 0 duplicates, 0 non-increasing timestamps and 14 disconnected pairs in the reference routes. The pipeline repeats the cleaning rules for every trip.
Each trip is projected to metres with a local equirectangular projection, because lon/lat are angles and the trips are spread across many countries.</p>
<p><b>Candidates.</b> Every arc is sampled every 10 m into a KD-tree. Each GPS point gets all arcs within 50 m (at most 12). Candidate miss rate (true arc not among the candidates): 0.0021 on student_test and {f(E.loc['lgbm','candidate_miss_rate'])} on evaluator.</p>
<p><b>Labels.</b> A monotonic dynamic-programming alignment assigns each GPS point a position on the route (positions only move forward), minimising total distance. Points whose aligned arc is more than 50 m away are unreliable (<code>label_ok=0</code>): trip 00000054 has the car parked off-road, and 00000033 has a route covering only part of the drive. We do not train on them, but we do evaluate on them.</p>
<p><b>21 features:</b> distance group (dist, rank, difference/ratio to nearest), temporal context (distance of the same arc to the previous, next and +/-2 points), direction (cosine of GPS heading vs arc bearing at 1 s and 5 s; only valid above 1.5 m/s), and geometry/topology (arc length, projection position, overhang, node degrees, two-way flag, time gap, fallback flag).
Why: distance alone cannot separate the two directions of a street or a crossing street; heading and context can.</p>

<h2 class='pb'>4. What "optimizers" we used, at three levels</h2>
<h3>4.1 Training optimisers inside each model (fit the model's weights)</h3>{optim_tbl}
<h3>4.2 Bayesian hyperparameter optimisation (Optuna TPE), applied to LightGBM</h3>
<p><b>What it is:</b> grid search tries every combination; random search tries random ones. <b>TPE (Tree-structured Parzen Estimator)</b> is Bayesian: after each trial it builds two probability models, one of the hyperparameters that gave good scores and one of those that gave bad scores, and proposes the next trial where the ratio good/bad is highest.
It learns where the good region is, so it needs far fewer trials.</p>
<p><b>Our setup:</b> 7 hyperparameters (n_estimators 200-1500, learning_rate 0.01-0.2, num_leaves 15-255, min_child_samples 10-300, subsample 0.5-1, colsample_bytree 0.4-1, reg_lambda 0.001-30). Objective: mean Top-1 on reliably-labelled points over a <b>3-fold GroupKFold on the training trips only</b>. {OPT['n_trials']} trials, {OPT['search_seconds']/60:.0f} min. Best: 1133 trees, lr 0.051, 175 leaves, min_child_samples 11.</p>
<p><b>Accuracy before and after Bayesian optimisation (LightGBM):</b></p>{bayes}
<p><b>Interpretation:</b> on the training CV the tuned model is marginally better (+{(R['lgbm_tuned']['cv_top1_labelled_mean']-R['lgbm']['cv_top1_labelled_mean'])*100:.2f} pp). On the unseen evaluator trips it is {pp(lgt['top1_accuracy']-lg['top1_accuracy'])} worse and has a higher log loss. Its trees are much bigger, which fits the training cities more closely but transfers less well, so this is mild overfitting.
We therefore kept the <b>untuned</b> LightGBM as the final model; that decision was made on student_test, before the evaluator data was used.</p>
<h3>4.3 Sequence-level optimisation: HMM + Viterbi (applied on top of every model)</h3>
<p>The classifier scores each GPS point <i>independently</i>, so consecutive points can land on unconnected roads. The HMM treats the candidates as hidden states. <b>Emission</b> = w * log(classifier probability). <b>Transition</b> = -|road-network distance - straight GPS distance| / beta, with the network distance from Dijkstra on the <i>directed</i> graph, so wrong-way and unconnected moves are impossible.
<b>Viterbi</b> is dynamic programming: for each time step and state it keeps the best score of any path ending there, then backtracks. It finds the single most probable <i>whole route</i>, not the best point-by-point choice.
Decoder parameters (beta = {HCFG['beta']}, w = {HCFG['emission_weight']}) were chosen by a 4x3 grid search on student_test.</p>
<p><b>Accuracy of each model before and after HMM optimisation, evaluator_test ({NOBS:,} GPS points, 10 trips):</b></p>{before_after}
{img(PLOTS / 'eval_top1_before_after_hmm.png', 'Evaluator-test Top-1 per model, before vs after HMM/Viterbi', '92%')}
<p>The HMM improves <b>every</b> model, by {gain_min*100:.1f}-{gain_max*100:.1f} percentage points, and lifts route continuity from about 0.94 to about 0.97. The gain is smallest for KNN, whose coarse vote-share probabilities give the decoder less to work with.
After HMM the best is {NAMES[best_hmm]} ({f(E.loc[best_hmm,'hmm_top1_accuracy'])}), with LightGBM essentially level ({f(lg['hmm_top1_accuracy'])}).</p>

<h2 class='pb'>5. Evaluator-test results - all models, ranked by Top-1 accuracy</h2>
<p>All six saved models were evaluated <b>once</b>, frozen (no refit, no retuning), on the 10 evaluator trips. Ranking by observation-level Top-1 accuracy (classifier only):</p>{full}
{img(PLOTS / 'eval_f1_logloss_cont.png', 'Evaluator-test macro F1, log loss and route continuity after HMM')}
<h3>Is Random Forest really better than LightGBM?</h3>
<p>Pooled over all points RF leads by {pp(rf['top1_accuracy']-lg['top1_accuracy'])}, but per trip:</p>{trip_tbl}
<p>LightGBM is better on <b>{PAIR['trips_b_better']} of 10 trips</b>. RF's pooled lead comes from trip {big} alone, which holds {100*nobs[big]/NOBS:.0f}% of all points and is the hardest trip.
The mean per-trip difference is {PAIR['mean_per_trip_diff']*100:+.2f} pp with a bootstrap 95% CI of {PAIR['ci95_low']*100:+.2f} to {PAIR['ci95_high']*100:+.2f} pp, which includes 0. <b>They are statistically tied.</b></p>
<div class='warn'><b>Why we did NOT switch the final model to RF after seeing these numbers:</b> the rules say evaluator results must not be used to choose algorithms or hyperparameters.
The choice was made on student_test (LightGBM 0.8987 vs RF 0.8897 Top-1) and is backed by better log loss, a 6x faster training time and a much smaller model. Switching now would be selecting on the test set, which is data leakage.
If asked: "The two are within noise; we report the evaluator ranking honestly, but selection used only the permitted partitions."</div>
{img(PLOTS / 'eval_per_trip_top1.png', 'Evaluator-test Top-1 per trip for every model: the differences between trips dominate the differences between models')}

<h2 class='pb'>6. Why the models performed the way they did</h2>
<div class='box'><b>Big picture:</b> all models land within about 1 pp of each other (0.887-0.898). The features carry most of the information, so the algorithm matters less than the feature design. The differences between <i>trips</i> (0.78 to 0.99) are far larger than the differences between models, and the hardest trips are long, dense urban ones with parallel roads.
Models that capture feature interactions (trees) are on top; models limited by assumptions (linear), data size (MLP subsample, KNN subsample) or dimensionality (KNN) are slightly lower.</div>
{model_cards}

<h2 class='pb'>7. Likely viva questions, with answers</h2>
{qa("Why Top-1 accuracy as the main metric and not macro F1?", "Top-1 is the actual task: one road segment per GPS point. Macro F1 is computed over all candidate rows at a 0.5 threshold, so it depends on the threshold and on how many wrong candidates there are. We report both, plus log loss because the HMM uses the probabilities.")}
{qa("Why GroupKFold?", "Consecutive GPS points of one trip are almost identical. A random split would put the same trip in training and validation and inflate the scores. Grouping by trip_id makes every validation fold a set of unseen trips, like the real test.")}
{qa("How do you get labels when .route has no timestamps?", "Monotonic dynamic-programming alignment: each GPS point gets a route position that can only move forward in time, minimising the sum of point-to-arc distances (capped at 100 m). The candidate equal to the aligned arc is the positive.")}
{qa("Why is CV fold 0 so low?", "It contains trip 00000054, where the car is parked off the road network for most of the recording (86% of points are more than 50 m from any road). Those points have no usable label. We therefore also report Top-1 on reliably-labelled points.")}
{qa("Why didn't Bayesian optimisation improve the model?", f"The default LightGBM was already close to the optimum. The best trial gained under 0.1 pp in CV and chose much larger trees (175 leaves), which overfit to the training cities: evaluator Top-1 {f(lgt['top1_accuracy'])} vs {f(lg['top1_accuracy'])}. We kept the untuned model, a decision made on student_test.")}
{qa("What is the difference between TPE and grid search?", "Grid search evaluates a fixed grid blindly. TPE models which regions of the search space gave good results and samples new trials there, so it reaches good settings in far fewer trials.")}
{qa("What are the hidden states, emissions and transitions of your HMM?", "States: the candidate arcs of each GPS point. Emission: w times the log of the classifier probability. Transition: minus the absolute difference between the directed road-network distance (Dijkstra) and the straight GPS distance, divided by beta. Viterbi finds the most probable sequence.")}
{qa("What happens at GPS gaps or disconnected roads?", "If dt > 10 s we widen the Dijkstra search and scale beta by dt. If no transition at all is feasible, the chain is closed at its best state and restarted from the emissions (a 'break'), so decoding never fails.")}
{qa("Why can route continuity be below 1 even after HMM?", "At 1 Hz the car can cross several short arcs between two points. Our decoded arcs are connected through the network but are not always directly adjacent in the per-point list.")}
{qa("Why does KNN have a training F1 of 1.0?", "With distance weighting, each training point is its own nearest neighbour at distance 0, so it predicts itself perfectly. That is not evidence of overfitting; the CV and test numbers are what count.")}
{qa("How do you make sure ground truth never reaches the model or UI?", "predict_trip() runs with route=None. Only after prediction does evaluate_prediction() load the route on the server to compute the metrics. The /predict response contains GPS points, the predicted polyline and the metrics, never the reference route.")}
{qa("Why project coordinates?", "Lon/lat are angles; one degree of longitude shrinks with cos(latitude). We project each trip onto a local plane in metres around its centroid, so distances and angles are physical.")}
{qa("Why not just pick the nearest road?", "Nearest-road matching fails on parallel streets, at junctions and on two-way roads, where the opposite direction is at the same distance. Our heading and temporal-context features plus the HMM fix exactly those cases.")}
{qa("Which model would you deploy and why?", f"LightGBM + HMM: best student-test Top-1 and log loss, about 0.1 ms per point, 30 s training, a small model file, and a statistical tie with RF on evaluator_test ({f(lg['hmm_top1_accuracy'])} vs {f(rf['hmm_top1_accuracy'])} after HMM).")}

<h2>8. Numbers cheat-sheet</h2>
{tbl([["Train / student / evaluator trips", "80 / 10 / 10"], ["Evaluator GPS points", f"{NOBS:,}"],
      ["Candidates per point (max / radius)", "12 / 50 m"], ["Features", "21"],
      ["Final model", "LightGBM, 600 trees, lr 0.05, 63 leaves"], ["HMM", f"beta {HCFG['beta']}, emission weight {HCFG['emission_weight']}"],
      ["Evaluator Top-1 LightGBM before / after HMM", f"{f(lg['top1_accuracy'])} / {f(lg['hmm_top1_accuracy'])}"],
      ["Evaluator Top-1 RF before / after HMM", f"{f(rf['top1_accuracy'])} / {f(rf['hmm_top1_accuracy'])}"],
      ["Evaluator continuity LightGBM before / after HMM", f"{f(lg['route_continuity'])} / {f(lg['hmm_route_continuity'])}"],
      ["Evaluator macro F1 / log loss (LightGBM)", f"{f(lg['f1_macro'])} / {f(lg['log_loss'])}"],
      ["Optuna", f"{OPT['n_trials']} TPE trials, {OPT['search_seconds']/60:.0f} min, no gain on test"],
      ["Student-test Top-1 LightGBM (selection)", f(R['lgbm']['student_test']['top1_accuracy'])]], ["Item", "Value"])}
</body></html>"""

html_path = OUT / "PathSnap_Viva_Guide.html"
html_path.write_text(H, encoding="utf-8")
pdf = OUT / "PathSnap_Viva_Guide.pdf"
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={OUT / '.edge_profile' / 'viva'}",
                f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, timeout=180)
print("viva ->", pdf)
