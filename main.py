"""
Main entrypoint for INST414 Final Project — Part 2.
"""
#importing functions from etl modules for execution
from etl.extract import verify_inputs
from etl.transform import run as transform_run
from etl.load import load_model_ready

#imports for analysis + visualizations
from analysis.model import main as model_run
from vis.visualizations import main as viz_run

def main():
    verify_inputs()
    
    #run transforms (ETL)
    transform_run()
    
    #load
    df = load_model_ready()
    print("[OK] Loaded processed table:", df.shape)
    
    #run analysis
    model_run()
    
    #run visualizations
    viz_run()
    
    print("Done! Outputs in /analysis/outputs and /vis/outputs.")

if __name__ == "__main__":
    main()