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

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "data" / "outputs"

METRICS_IN = OUTPUTS / "metrics_regression.csv"
COEFS_IN   = OUTPUTS / "coefficients_regression.csv"
PRED_IN    = OUTPUTS / "predictions_regression.csv"

SCATTER_OUT = OUTPUTS / "viz_actual_vs_predicted.png"
COEF_OUT    = OUTPUTS / "viz_coefficients.png"
RESID_OUT   = OUTPUTS / "viz_residuals.png"


def load_inputs():
    """
    Load the three CSVs produced by analysis/model.py.
    We only need coefficients and predictions for the plots; metrics are for titles/annotations.
    """
    if not PRED_IN.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_IN}")
    if not COEFS_IN.exists():
        raise FileNotFoundError(f"Missing coefficients file: {COEFS_IN}")

    preds = pd.read_csv(PRED_IN)
    coefs = pd.read_csv(COEFS_IN)

    # Metrics are optional (used for annotating the scatter title if available)
    metrics = None
    if METRICS_IN.exists():
        metrics = pd.read_csv(METRICS_IN)

    # Expect these columns from model.py
    for col in ["y_true_tons_out", "y_pred_tons_out"]:
        if col not in preds.columns:
            raise ValueError(f"Column '{col}' not found in {PRED_IN.name}")

    if "feature" not in coefs.columns or "coef_standardized" not in coefs.columns:
        raise ValueError(f"Columns 'feature' and 'coef_standardized' not found in {COEFS_IN.name}")

    return preds, coefs, metrics


def make_scatter_actual_vs_predicted(preds: pd.DataFrame, metrics: pd.DataFrame | None, out_path: Path):
    """
    Scatter of actual vs predicted with a y=x reference line.
    One figure only; no custom colors (class constraint).
    """
    y_true = preds["y_true_tons_out"].values
    y_pred = preds["y_pred_tons_out"].values

    # For a square plot and nice 45° line, set symmetric limits
    m = np.nanmin([y_true.min(), y_pred.min()])
    M = np.nanmax([y_true.max(), y_pred.max()])
    # Pad a bit
    pad = 0.02 * (M - m) if M > m else 1.0
    lo, hi = m - pad, M + pad

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    # 45° reference line
    plt.plot([lo, hi], [lo, hi])

    title = "Actual vs Predicted (Freight Value: tons_out model)"
    if metrics is not None and not metrics.empty:
        # Safely format metrics for title
        r2 = metrics.iloc[0].get("R2", None)
        rmse = metrics.iloc[0].get("RMSE", None)
        mae = metrics.iloc[0].get("MAE", None)
        bits = []
        if isinstance(r2, (int, float)) and not math.isnan(r2):
            bits.append(f"R²={r2:.3f}")
        if isinstance(rmse, (int, float)) and not math.isnan(rmse):
            bits.append(f"RMSE={rmse:,.0f}")
        if isinstance(mae, (int, float)) and not math.isnan(mae):
            bits.append(f"MAE={mae:,.0f}")
        if bits:
            title += " | " + " · ".join(bits)

    plt.title(title)
    plt.xlabel("Actual value_out (test)")
    plt.ylabel("Predicted value_out (test)")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_barh_coefficients(coefs: pd.DataFrame, out_path: Path):
    """
    Horizontal bar chart of standardized coefficients.
    Sort by absolute value so the most important features appear at the top.
    No custom colors per course instruction.
    """
    c = coefs.copy()
    c["abs_coef"] = c["coef_standardized"].abs()
    c = c.sort_values("abs_coef", ascending=True)  # ascending for a bottom-up barh

    plt.figure(figsize=(8, max(4, 0.4 * len(c))))
    plt.barh(c["feature"], c["coef_standardized"])
    plt.title("Standardized Coefficients (Linear Regression)")
    plt.xlabel("Coefficient (standardized units)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_hist_residuals(preds: pd.DataFrame, out_path: Path):
    """
    Histogram of residuals (actual - predicted).
    Useful to see bias, skew, or heavy tails in errors.
    """
    resid = preds["y_true_tons_out"] - preds["y_pred_tons_out"]

    plt.figure(figsize=(8, 6))
    plt.hist(resid.dropna().values, bins=20)
    plt.title("Residuals (Actual − Predicted)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    preds, coefs, metrics = load_inputs()

    make_scatter_actual_vs_predicted(preds, metrics, SCATTER_OUT)
    make_barh_coefficients(coefs, COEF_OUT)
    make_hist_residuals(preds, RESID_OUT)

    print("[VIS] Saved:")
    print(f"  - {SCATTER_OUT}")
    print(f"  - {COEF_OUT}")
    print(f"  - {RESID_OUT}")


if __name__ == "__main__":
    main()
