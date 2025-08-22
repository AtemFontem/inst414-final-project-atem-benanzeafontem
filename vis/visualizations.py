"""
vis/visualizations.py

Generates visuals for the regression results:
  1) Scatter: Actual vs Predicted values (model performance)
  2) Horizontal bar chart: Standardized coefficients (feature importance)
  3) Histogram: Residuals (error distribution)

Inputs (from analysis/model.py outputs):
  - data/outputs/metrics_regression.csv
  - data/outputs/coefficients_regression.csv
  - data/outputs/predictions_regression.csv

Outputs:
  - data/outputs/viz_actual_vs_predicted.png
  - data/outputs/viz_coefficients.png
  - data/outputs/viz_residuals.png
"""
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Paths
ANALYSIS_OUT = os.path.join("analysis", "outputs")
OUT_DIR      = os.path.join("vis", "outputs")

COEFS_CSV = os.path.join(ANALYSIS_OUT, "coefficients_regression.csv")
PREDS_CSV = os.path.join(ANALYSIS_OUT, "predictions_regression.csv")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_coefficients(coefs: pd.DataFrame, out_path: str, top_k: int = 20) -> None:
    if coefs.empty or "feature" not in coefs or "coef_standardized" not in coefs:
        print("[VIZ][WARN] Coefficients dataframe empty or missing columns; skipping coefficients plot.")
        return

    c = coefs.copy()
    c["abs_coef"] = c["coef_standardized"].abs()
    c = c.sort_values("abs_coef", ascending=True).tail(top_k)

    plt.figure(figsize=(9, 7))
    plt.barh(c["feature"], c["coef_standardized"])
    plt.title("Standardized Coefficients (Top by |coef|)")
    plt.xlabel("Coefficient (standardized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(preds: pd.DataFrame, out_path: str) -> None:
    if preds.empty:
        print("[VIZ][WARN] Predictions dataframe is empty; skipping residuals plot.")
        return
    if "y_true" not in preds or "y_pred" not in preds:
        print("[VIZ][WARN] Missing y_true/y_pred; skipping residuals plot.")
        return

    # Compute residuals if missing (or overwrite just to be safe/consistent)
    preds = preds.copy()
    preds["residual"] = preds["y_true"] - preds["y_pred"]

    plt.figure(figsize=(8, 6))
    plt.hist(preds["residual"], bins=20)
    plt.title("Residuals (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_actual_vs_pred(preds: pd.DataFrame, out_path: str) -> None:
    if preds.empty:
        print("[VIZ][WARN] Predictions dataframe is empty; skipping actual-vs-predicted plot.")
        return
    if "y_true" not in preds or "y_pred" not in preds:
        print("[VIZ][WARN] Missing y_true/y_pred; skipping actual-vs-predicted plot.")
        return

    x = preds["y_true"].to_numpy()
    y = preds["y_pred"].to_numpy()
    lim_min = float(min(np.min(x), np.min(y)))
    lim_max = float(max(np.max(x), np.max(y)))
    if np.isnan(lim_min) or np.isnan(lim_max) or lim_min == lim_max:
        print("[VIZ][WARN] Degenerate limits for actual-vs-predicted; skipping plot.")
        return

    plt.figure(figsize=(7.5, 7))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual (y_true)")
    plt.ylabel("Predicted (y_pred)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    _ensure_dir(OUT_DIR)

    if not os.path.exists(COEFS_CSV):
        raise FileNotFoundError(f"Missing {COEFS_CSV}. Run analysis/model.py first.")
    if not os.path.exists(PREDS_CSV):
        raise FileNotFoundError(f"Missing {PREDS_CSV}. Run analysis/model.py first.")

    coefs = pd.read_csv(COEFS_CSV)
    preds = pd.read_csv(PREDS_CSV)

    p1 = os.path.join(OUT_DIR, "viz_coefficients.png")
    p2 = os.path.join(OUT_DIR, "viz_residuals.png")
    p3 = os.path.join(OUT_DIR, "viz_actual_vs_predicted.png")

    plot_coefficients(coefs, p1)
    plot_residuals(preds, p2)
    plot_actual_vs_pred(preds, p3)

    print("[VIZ] Saved visuals to vis/outputs (where generated)")
    for p in [p1, p2, p3]:
        if os.path.exists(p):
            print(f"  - {os.path.relpath(p)}")


if __name__ == "__main__":
    main()