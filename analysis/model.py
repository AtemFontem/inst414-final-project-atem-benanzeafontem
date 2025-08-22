"""
analysis/model.py

Train a baseline regression model to predict outbound freight volume (tons_out)
from state-year weather features.

Outputs (written to data/outputs/):
  - metrics_regression.csv          Overall test metrics (R2, RMSE, MAE)
  - coefficients_regression.csv     Standardized coefficients for interpretability
  - predictions_regression.csv      Per-row predictions for the test set
"""

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

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "data" / "outputs"

IN_CSV = PROCESSED / "model_ready_state_year.csv"
METRICS_OUT = OUTPUTS / "metrics_regression.csv"
COEFS_OUT = OUTPUTS / "coefficients_regression.csv"
PRED_OUT = OUTPUTS / "predictions_regression.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the merged state-year dataset produced by etl/transform.py
    and perform minimal cleaning for modeling.
    """
    df = pd.read_csv(path)

    # Prefer a single readable state_abbr column if both exist
    if "state_abbr" not in df.columns:
        if "state_abbr_faf" in df.columns or "state_abbr_wx" in df.columns:
            df["state_abbr"] = df.get("state_abbr_faf", pd.Series([np.nan]*len(df))).fillna(
                df.get("state_abbr_wx", pd.Series([np.nan]*len(df)))
            )

    # Drop rows explicitly missing weather (e.g., 2024 if your weather file stops in 2023)
    if "missing_weather" in df.columns:
        df = df[df["missing_weather"] == 0].copy()

    # Ensure expected columns exist
    required = {"state_fips", "year", "tons_out"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Select the target and feature set.

    Target:
      - tons_out (continuous)

    Predictors (weather-based, engineered in transform.py):
      - precipitation stats
      - temperature stats
      - extreme-day counts
      - n_days (coverage)

    Returns:
      X (features DataFrame), y (Series target), feature_names (list of column names)
    """
    target_col = "tons_out"

    candidate_features = [
        "prcp_total_mm",
        "prcp_mean_mm",
        "prcp95_mm",
        "tavg_mean_c",
        "tavg_std_c",
        "hot_days",
        "cold_days",
        "heavy_rain_days",
        "very_heavy_rain_days",
        "n_days",
    ]

    # Keep only features that actually exist in the file
    features = [c for c in candidate_features if c in df.columns]
    if not features:
        raise ValueError("No weather feature columns found; check transform outputs.")

    X = df[features].copy()
    y = df[target_col].copy()

    return X, y, features


def train_test_split_time_aware(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """
    Time-aware split: Train on 2018-2022, Test on 2023 (if available).
    Fallback to random 80/20 if 2023 is not present.
    """
    if 2023 in df["year"].unique():
        train_mask = df["year"].between(2018, 2022)
        test_mask = df["year"] == 2023

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # If for any reason this yields empty sets (e.g., data coverage quirks),
        # we fallback below.
        if len(X_train) > 0 and len(X_test) > 0:
            return X_train, X_test, y_train, y_test, "time_2018_2022_train__2023_test"

    # Fallback: simple random split (stratification not relevant for regression)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, "random_80_20"


def build_pipeline(numeric_features: list) -> Pipeline:
    """
    Build a modeling pipeline:
      - Impute missing numeric values with median
      - Scale numeric features (StandardScaler)
      - LinearRegression model
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preproc", preproc),
        ("model", model),
    ])
    return pipe


def evaluate_and_save(pipe: Pipeline,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      feature_names: list,
                      df_test_keys: pd.DataFrame):
    """
    Evaluate the model on the test set and save metrics, coefficients, and predictions.

    - Metrics: R2, RMSE, MAE
    - Coefficients: standardized coefficients (since inputs were scaled)
    - Predictions: merged with state_fips, state_abbr, year for easy diagnostics
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Predictions
    y_pred = pipe.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)   # returns MSE
    rmse = float(np.sqrt(mse))                 # take sqrt to get RMSE
    mae = mean_absolute_error(y_test, y_pred)

    metrics_df = pd.DataFrame([{"R2": r2, "RMSE": rmse, "MAE": mae}])
    metrics_df.to_csv(METRICS_OUT, index=False)

    # Coefficients (standardized inputs -> coefficients are comparable)
    # Access underlying LinearRegression coef_ via named steps
    coefs = pipe.named_steps["model"].coef_
    coefs_df = pd.DataFrame({
        "feature": feature_names,
        "coef_standardized": coefs
    }).sort_values("coef_standardized", ascending=False)
    coefs_df.to_csv(COEFS_OUT, index=False)

    # Prediction rows with keys for inspection
    pred_df = df_test_keys.copy()
    pred_df["y_true_tons_out"] = y_test.values
    pred_df["y_pred_tons_out"] = y_pred
    pred_df["abs_error"] = (pred_df["y_true_tons_out"] - pred_df["y_pred_tons_out"]).abs()
    pred_df.to_csv(PRED_OUT, index=False)

    print("[MODEL] Saved:")
    print(f"  - {METRICS_OUT.name}: R2={r2:.3f}, RMSE={rmse:,.2f}, MAE={mae:,.2f}")
    print(f"  - {COEFS_OUT.name}: standardized coefficients")
    print(f"  - {PRED_OUT.name}: per-row predictions")


def main():
    # 1) Load & basic clean
    df = load_dataset(IN_CSV)

    # 2) Select features/target
    X, y, features = select_features(df)

    # 3) Time-aware split (or fallback)
    X_train, X_test, y_train, y_test, split_label = train_test_split_time_aware(df, X, y)
    print(f"[SPLIT] Strategy: {split_label} | train={len(X_train)} rows, test={len(X_test)} rows")

    # Keep identifying columns for the test output
    id_cols = [c for c in ["state_fips", "state_abbr", "year"] if c in df.columns]
    df_test_keys = df.loc[X_test.index, id_cols].copy()

    # 4) Build pipeline & fit
    pipe = build_pipeline(features)
    pipe.fit(X_train, y_train)

    # 5) Evaluate & save artifacts
    evaluate_and_save(pipe, X_test, y_test, features, df_test_keys)


if __name__ == "__main__":
    main()
