# Weather and Freight Flows: INST414 Final Project

## Project Overview
This project investigates how weather patterns influence freight flows across U.S. states. Using data from the Freight Analysis Framework (FAF) and NOAA weather summaries, we construct a regression model to evaluate whether climate and precipitation trends can explain variation in freight value.

The goal is to:

- Build a data pipeline (ETL → modeling → visualization)

- Apply regression analysis with proper evaluation metrics

- Interpret the role of weather features in explaining freight movement

## Structure
```
├── data/
│   ├── raw/           # Original CSVs (FAF, NOAA weather)
│   ├── processed/     # Cleaned/model-ready datasets
│   └── outputs/       # Final metrics/coefficients/predictions
│
├── etl/
│   ├── extract.py     # Data extraction
│   ├── transform.py   # Data cleaning, merging, and processing
│   └── load.py        # Loading processed data
│
├── analysis/
│   └── model.py       # Regression modeling & evaluation
│
├── vis/
│   └── visualizations.py  # Generates plots
│
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```
## Setup Instructions

#### Create venv (Python 3.12 recommended)
```
python -m venv .venv
```


#### Activate on Windows (PowerShell)
```
.venv\Scripts\activate
```


#### Activate on macOS/Linux
```
source .venv/bin/activate
```


#### Install dependencies
```
pip install -r requirements.txt
```
#### Run Pipeline

###### Step 1 - Transform Data
```
python etl/transform.py
```
- This cleans and merges freight + weather data.
- Produces: 
    `data/processed/model_ready_state_year.csv`
    (raw data originates from `FAF5.7.1_State.csv` & 
    `weather_us_county_2018_2023.csv` [weather dataset webscraped from NOAA website])

###### Step 2 – Run Model
```
python analysis/model.py
```
- Trains a regression model.

- Outputs:
    - `metrics_regression.csv`
    - `coefficients_regression.csv`        
    - `predictions_regression.csv`

###### Step 3 – Generate Visualizations
```
python vis/visualizations.py
```
- Trains a regression model.
    - Produces plots in `data/outputs/`:
        - Coefficient Plot
        - Residuals Plot
        - Actual vs Predicted

## Example Outputs
### Metrics (`metrics_regression.csv`)
```
R2, RMSE, MAE
0.116, 450893.40, 273031.42
```

## Visualization Samples
- `viz_coefficients.png`→ shows standardized coefficients

- `viz_residuals.png` → residual errors across predictions

- `viz_actual_vs_predicted.png` → model fit comparison

- 
## Notes
- Python 3.12 was used to ensure compatibility with numpy/matplotlib.

- Data sources:
    - Freight Analysis Framework (FAF)
    - NOAA Weather Data

- This pipeline is modular; ETL, modeling, and visualization can be run independently

## Author
Atem Benanzea-Fontem

INST414 – University of Maryland

Summer II 2025