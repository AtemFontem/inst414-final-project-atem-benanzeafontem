"""
analysis/model.py

Train a baseline regression model to predict outbound freight volume (tons_out)
from state-year weather features.

Outputs (written to analysis/outputs/):
  - metrics_regression.csv          Overall test metrics (R2, RMSE, MAE)
  - coefficients_regression.csv     Standardized coefficients for interpretability
  - predictions_regression.csv      Per-row predictions for the test set
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

# scikit-learn: pipeline for clean preprocessing + modeling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# Paths
INPUT_CSV = os.path.join("data", "processed", "model_ready_state_year.csv")
OUT_DIR   = os.path.join("analysis", "outputs")

# Train/test split (same as before)
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]
TEST_YEAR   = 2023

# Target (same as before in your runs)
TARGET_COL = "value_out"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _feature_list(df: pd.DataFrame) -> list[str]:
    """
    Use the same schema/column names you had previously.
    Exclude IDs, flags, target, and derived text columns.
    """
    drop_cols = {
        "state_fips", "state_abbr", "year",
        "missing_weather", TARGET_COL,
        "state_abbr_x", "state_abbr_y"  # tolerate either if present
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in drop_cols]


def train(df: pd.DataFrame) -> dict:
    """
    Fit standardized linear regression and return artifacts:
    metrics, coefficients, predictions, features.
    """
    # keep the years used earlier
    df_use = df[df["year"].isin(TRAIN_YEARS + [TEST_YEAR])].copy()
    df_use = df_use[~df_use[TARGET_COL].isna()].copy()

    # keys to keep in predictions
    keys = [c for c in ["state_fips", "state_abbr", "year"] if c in df_use.columns]

    features = _feature_list(df_use)
    if not features:
        raise ValueError("No features found")

    df_train = df_use[df_use["year"].isin(TRAIN_YEARS)].copy()
    df_test  = df_use[df_use["year"] == TEST_YEAR].copy()
    if df_train.empty or df_test.empty:
        raise ValueError("Train/test split is empty; check years in processed CSV.")

    X_train, y_train = df_train[features], df_train[TARGET_COL]
    X_test,  y_test  = df_test[features],  df_test[TARGET_COL]

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # metrics
    r2   = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = mean_absolute_error(y_test, y_pred)

    # standardized coefficients
    model = pipe.named_steps["model"]
    coefs = pd.DataFrame({
        "feature": features,
        "coef_standardized": model.coef_
    }).sort_values(by="coef_standardized", key=np.abs, ascending=False)

    preds = df_test[keys].copy()
    preds["y_true"] = y_test.values
    preds["y_pred"] = y_pred
    preds["residual"] = preds["y_true"] - preds["y_pred"]

    return {
        "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
        "coefficients": coefs,
        "predictions": preds,
        "features": features,
    }


def save_outputs(artifacts: dict) -> None:
    """Write outputs where Part 4 expects them."""
    _ensure_dir(OUT_DIR)

    # metrics
    pd.DataFrame([artifacts["metrics"]]).to_csv(
        os.path.join(OUT_DIR, "metrics_regression.csv"),
        index=False
    )

    # coefficients
    artifacts["coefficients"].to_csv(
        os.path.join(OUT_DIR, "coefficients_regression.csv"),
        index=False
    )

    # predictions
    artifacts["predictions"].to_csv(
        os.path.join(OUT_DIR, "predictions_regression.csv"),
        index=False
    )


    m = artifacts["metrics"]
    print("[MODEL] Saved artifacts to analysis/outputs")
    print(f"  R2={m['R2']:.3f} | RMSE={m['RMSE']:,.2f} | MAE={m['MAE']:,.2f}")


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run ETL first.")
    df = pd.read_csv(INPUT_CSV)
    artifacts = train(df)
    save_outputs(artifacts)


if __name__ == "__main__":
    main()