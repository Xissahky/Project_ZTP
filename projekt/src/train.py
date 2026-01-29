from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import build_preprocess_pipeline
from .config import paths

@dataclass(frozen=True)
class TrainedModel:
    name: str
    pipeline: Pipeline

def train_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
    pre = build_preprocess_pipeline(X_train)
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    return TrainedModel(name="logreg", pipeline=pipe)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModel:
    pre = build_preprocess_pipeline(X_train)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    return TrainedModel(name="rf", pipeline=pipe)


def save_model(model: TrainedModel, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.name}.joblib"
    joblib.dump(model.pipeline, out_path)
    return out_path
