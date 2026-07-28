from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
]


class FallbackChurnModel:
    """Deterministic model used when the serialized artifact is unavailable.

    It keeps the runtime and tests working in clean checkouts where `model.pkl`
    is intentionally not committed, while still producing sensible risk tiers.
    """

    def _to_frame(self, X: object) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return frame[FEATURE_COLUMNS].astype(float)

    def predict_proba(self, X: object) -> list[list[float]]:
        frame = self._to_frame(X)
        probabilities: list[list[float]] = []
        for _, row in frame.iterrows():
            logit = (
                -3.0
                + 0.02 * row["monthly_charges"]
                - 0.04 * row["tenure_months"]
                + 0.15 * row["num_support_tickets"]
                + 0.0003 * row["total_charges"]
            )
            churn_probability = 1.0 / (1.0 + math.exp(-logit))
            churn_probability = max(0.0, min(1.0, churn_probability))
            probabilities.append([1.0 - churn_probability, churn_probability])
        return probabilities

    def predict(self, X: object) -> list[int]:
        return [int(row[1] >= 0.5) for row in self.predict_proba(X)]


def _candidate_paths(explicit_path: str | None) -> Iterable[Path]:
    if explicit_path:
        yield Path(explicit_path)

    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        yield Path(env_path)

    current_dir = Path.cwd()
    yield current_dir / "model.pkl"
    yield current_dir / "churn-model-demo" / "model.pkl"

    module_dir = Path(__file__).resolve().parent
    yield module_dir / "model.pkl"
    yield module_dir / "churn-model-demo" / "model.pkl"


def load_churn_model(explicit_path: str | None = None):
    for candidate in _candidate_paths(explicit_path):
        if candidate.is_file():
            with candidate.open("rb") as handle:
                return pickle.load(handle)

    return FallbackChurnModel()
