import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def load_model_ready():
    df = pd.read_csv(PROCESSED / "model_ready_state_year.csv")
    return df
