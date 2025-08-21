"""
Main entrypoint for INST414 Final Project — Part 2.
"""

from etl.extract import verify_inputs
from etl.transform import run as transform_run
from etl.load import load_model_ready

def main():
    verify_inputs()
    transform_run()
    df = load_model_ready()
    print("[OK] Loaded processed table:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()