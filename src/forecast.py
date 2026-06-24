"""Per-oblast air-raid risk forecast + honest time-based backtest.

Model: HistGradientBoostingClassifier (handles the categorical region natively),
probabilities calibrated with isotonic regression on a held-out validation slice.
Evaluated against two baselines on a strictly later test period:
  - climatology:  P(alert next H | region, hour-of-day) learned on train
  - persistence:  "currently active" used as the probability

Metrics: ROC-AUC, PR-AUC (average precision), Brier score, plus a reliability
(calibration) curve. Results are written to data/processed/model_metrics.json and
near-term per-oblast risk to data/processed/risk_now.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from . import config
from .features import build_features, feature_names
from .transform import load_series


def _time_splits(feat: pd.DataFrame, test_frac: float = 0.2, valid_frac: float = 0.15):
    """Split by absolute time: train_fit < valid < test (no shuffling)."""
    labelled = feat[feat["y"].notna()]
    tmin, tmax = labelled["ts"].min(), labelled["ts"].max()
    span = tmax - tmin
    test_start = tmin + span * (1 - test_frac)
    valid_start = tmin + span * (1 - test_frac - valid_frac)

    train_fit = labelled[labelled["ts"] < valid_start]
    valid = labelled[(labelled["ts"] >= valid_start) & (labelled["ts"] < test_start)]
    test = labelled[labelled["ts"] >= test_start]
    return train_fit, valid, test, test_start


def _fit_model(train_fit: pd.DataFrame, valid: pd.DataFrame):
    """Fit HGB on train_fit, calibrate on valid. Returns (model, isotonic)."""
    feats = feature_names()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    model.fit(train_fit[feats], train_fit["y"].astype(int))

    raw_valid = model.predict_proba(valid[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_valid, valid["y"].astype(int))
    return model, iso


def _predict(model, iso, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X[feature_names()])[:, 1]
    return iso.transform(raw)


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def _climatology_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rate = train.groupby(["region", "hour"], observed=True)["y"].mean()
    overall = train["y"].mean()
    idx = list(zip(test["region"].astype(str), test["hour"]))
    return np.array([rate.get(k, overall) for k in idx])


def backtest() -> dict:
    print("[forecast] building features ...")
    feat = build_features(load_series())
    train_fit, valid, test, test_start = _time_splits(feat)
    train_all = pd.concat([train_fit, valid])
    print(f"[forecast] train={len(train_all):,} valid={len(valid):,} test={len(test):,}")
    print(f"[forecast] test period starts {test_start}")

    model, iso = _fit_model(train_fit, valid)
    p_model = _predict(model, iso, test)
    p_raw = model.predict_proba(test[feature_names()])[:, 1]
    y_test = test["y"].astype(int).to_numpy()

    p_clim = _climatology_baseline(train_all, test)
    p_pers = test["active_now"].to_numpy().astype(float)

    m_model = _metrics(y_test, p_model)
    m_model["brier_uncalibrated"] = float(brier_score_loss(y_test, p_raw))

    # reliability curve (calibrated model)
    frac_pos, mean_pred = calibration_curve(y_test, p_model, n_bins=10, strategy="quantile")
    # downsampled ROC curve
    fpr, tpr, _ = roc_curve(y_test, p_model)
    step = max(1, len(fpr) // 200)

    results = {
        "horizon_h": config.FORECAST_HORIZON_HOURS,
        "n_train": int(len(train_all)),
        "n_test": int(len(test)),
        "test_start": str(test_start),
        "test_end": str(test["ts"].max()),
        "positive_rate_test": float(y_test.mean()),
        "model": m_model,
        "baseline_climatology": _metrics(y_test, p_clim),
        "baseline_persistence": _metrics(y_test, p_pers),
        "reliability": {
            "mean_predicted": [float(x) for x in mean_pred],
            "fraction_positive": [float(x) for x in frac_pos],
        },
        "roc_curve": {
            "fpr": [float(x) for x in fpr[::step]],
            "tpr": [float(x) for x in tpr[::step]],
        },
    }

    lift = m_model["roc_auc"] - results["baseline_climatology"]["roc_auc"]
    print(f"[forecast] model ROC-AUC={m_model['roc_auc']:.3f}  PR-AUC={m_model['pr_auc']:.3f}  "
          f"Brier={m_model['brier']:.3f}")
    print(f"[forecast] climatology ROC-AUC={results['baseline_climatology']['roc_auc']:.3f}  "
          f"persistence ROC-AUC={results['baseline_persistence']['roc_auc']:.3f}")
    print(f"[forecast] lift over climatology baseline: {lift:+.3f} ROC-AUC")

    with open(config.MODEL_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[forecast] saved {config.MODEL_METRICS_JSON}")
    return results


def predict_current_risk() -> pd.DataFrame:
    """Train on all history, forecast next-H risk for the latest hour per region."""
    feat = build_features(load_series())
    train_fit, valid, _test, _ = _time_splits(feat, test_frac=0.0, valid_frac=0.15)
    model, iso = _fit_model(train_fit, valid)

    latest = feat.sort_values("ts").groupby("region", observed=True).tail(1).copy()
    latest["risk"] = _predict(model, iso, latest)
    out = latest[["region", "ts", "risk", "active_now"]].sort_values("risk", ascending=False)
    out.to_parquet(config.SERIES_PARQUET.parent / "risk_now.parquet", index=False)
    print(f"[forecast] saved risk_now.parquet (as of {out['ts'].max()})")
    print(out.head(8).to_string(index=False))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Air-raid risk forecast")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args(argv)
    if not (args.backtest or args.predict):
        args.backtest = args.predict = True  # default: both

    if args.backtest:
        backtest()
    if args.predict:
        predict_current_risk()
    return 0


if __name__ == "__main__":
    sys.exit(main())
