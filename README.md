# Weather and Freight Flows

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
To make sense of the regression outputs, I generated three plots. Below is what each visualization shows and how it connects back to the business problem of analyzing freight movement and its relationship with climate/weather patterns.

- `viz_coefficients.png`→ shows standardized coefficients
<img width="1600" height="800" alt="viz_coefficients" src="https://github.com/user-attachments/assets/7b868602-969f-4ec5-a48e-1988a6f6e76a" />

- This bar chart shows the standardized regression coefficients for each predictor variable.  
- The height and sign (positive/negative) of each bar tell us how strongly that variable contributes to predicting freight activity (measured in tons/value). Positive coefficients mean that as the predictor increases, freight also tends to increase. Negative coefficients mean the opposite.
- This helps identify which factors, such as precipitation totals, average temperature, or extreme weather events (hot days, heavy rain days), have the strongest relationship with freight outcomes. Even if the overall R² is modest, this gives a directional signal about which climate factors might matter most for planning.

- `viz_residuals.png` → residual errors across predictions
<img width="1600" height="1200" alt="viz_residuals" src="https://github.com/user-attachments/assets/4d586d0a-26c3-4d33-a34d-6eeb2bb6d315" />

- This scatter plot shows residuals (errors) vs. predicted values. Each point represents a state–year observation.  
- The vertical position shows how far off the model’s prediction was from the actual outcome. Ideally, residuals should be scattered randomly around zero; if patterns appear, it means the model is missing some structure in the data.  
- This helps evaluate whether the regression model is systematically underpredicting/overpredicting freight activity for certain ranges. It tells us about the limitations of the model and whether additional features or different modeling strategies might be needed.


- `viz_actual_vs_predicted.png` → model fit comparison
<img width="1600" height="1200" alt="viz_actual_vs_predicted" src="https://github.com/user-attachments/assets/9e177454-367c-486a-a14a-8f40c0490fda" />

- This plot compares the **true freight values** (on the x-axis) with the model’s predictions (on the y-axis). The 45 degree diagonal line represents “perfect predictions.” Points close to the line indicate accurate predictions. Points far from the line show where the model struggled.  
- This visualization directly shows the predictive accuracy of the model. It could give stakeholders an intuitive way to see how well the model aligns with reality; whether predictions are generally on target, or whether there are large discrepancies for certain states or years.

Even with modest performance (R² ~ 0.12), these visuals demonstrate how data science can start to quantify the relationship between freight activity and environmental conditions, and they point toward areas for future refinement.

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
