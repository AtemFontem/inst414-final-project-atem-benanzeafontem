from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "extracted"

def verify_inputs():
    needed = [
        EXTRACTED / "weather_us_county_2018_2023.csv",
        EXTRACTED / "FAF5.7.1_State.csv",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    print("[OK] inputs present.")

if __name__ == "__main__":
    verify_inputs()
