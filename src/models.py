"""The five candidate classifiers (one per algorithm family) + their Bayesian search spaces."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_JOBS, SEED

FAMILIES = {
    "logreg": "Linear / probabilistic",
    "knn": "Instance-based",
    "rf": "Ensemble (bagging)",
    "lgbm": "Boosting",
    "mlp": "Neural network",
}

DEFAULT_PARAMS = {
    "logreg": {"C": 1.0, "max_iter": 2000},
    "knn": {"n_neighbors": 50, "weights": "distance"},
    "rf": {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt", "max_depth": None},
    "lgbm": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "mlp": {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3,
            "batch_size": 1024, "max_iter": 200, "early_stopping": True, "n_iter_no_change": 10},
}


def make_model(name, params=None):
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **p))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=N_JOBS, **p))
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **p)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=-1, **p)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED, **p))
    raise ValueError(name)


def suggest_params(name, trial):
    """Optuna (TPE = Bayesian) search spaces. Ranges are centred on the defaults above."""
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 6, 40),
        }
    if name == "mlp":
        width = trial.suggest_categorical("width", [64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        return {
            "hidden_layer_sizes": tuple([width] * depth),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 200, log=True),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    raise ValueError(name)
