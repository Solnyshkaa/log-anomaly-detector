"""Fit and score Isolation Forest feature vectors."""

import pandas as pd
from sklearn.ensemble import IsolationForest


def fit(features):
    if len(features) < 20:
        raise ValueError("Provide at least 20 normal training observations")
    return IsolationForest(random_state=42, n_estimators=100).fit(features)


def score(model, features):
    return pd.Series(-model.score_samples(features), index=features.index)
