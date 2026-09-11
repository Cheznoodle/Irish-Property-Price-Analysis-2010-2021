# Irish Property Price Analysis & Prediction (2010–2021)

Analysis and machine learning prediction of residential property prices in 
Ireland using the official Property Price Register (PSRA) dataset, covering 
~476,000 transactions from 2010 to 2021.

## Overview

This project explores how residential property prices in Ireland have evolved 
over the past decade, how trends differ across counties, and whether a 
regression model can estimate property prices based on location, sale date, 
and available property characteristics.

## Project Files

The full set of scripts and datasets used for this project are provided in a 
single zip file (`Code.zip`), which includes:

- `describe.py`, `clean_dataset.py`, `feature_engineer.py`, `predict.py` — pipeline scripts
- `Property_Price_Register_Ireland-28-05-2021.csv` — raw dataset
- `Property_Price_Register_Ireland-28-05-2021_cleaned.csv` — cleaned dataset
- `Property_Price_Register_Ireland-28-05-2021_features.csv` — feature-engineered dataset
- `gadm41_IRL_1.json` — Ireland county boundary GeoJSON (used for choropleth maps)

Extract the zip into the repo root before running the pipeline so the scripts 
can find the CSV and JSON files alongside them.

## Getting Started

### 1. Install Python

This project requires **Python 3.9 or later**.

- **Windows / macOS:** Download the installer from [python.org/downloads](https://www.python.org/downloads/) and run it.
  - On Windows, make sure to tick **"Add Python to PATH"** during setup.
- **macOS (alternative):** `brew install python`
- **Linux:** `sudo apt install python3 python3-pip` (Debian/Ubuntu)

Verify the install:
```bash
python --version
# or on macOS/Linux:
python3 --version
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/irish-property-price-analysis.git
cd irish-property-price-analysis
```

### 3. (Recommended) Create a virtual environment

Keeps project dependencies isolated from your system Python.

```bash
python -m venv venv

# Activate it:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn geopandas folium mplcursors statsmodels prophet
```

Or

```bash
pip install -r requirements.txt
```

> **Note on `geopandas`:** it depends on GEOS/GDAL/PROJ under the hood, which can be tricky to install via plain `pip` on Windows. If `pip install geopandas` fails, use conda instead:
> ```bash
> conda install -c conda-forge geopandas
> ```

> **Note on `prophet`:** installation can be slow as it compiles a Stan backend. If it fails, try:
> ```bash
> pip install prophet --no-cache-dir
> ```
> or install via conda: `conda install -c conda-forge prophet`

### 5. Run the pipeline

Scripts are run in order, each producing the input for the next:

```bash
python describe.py Property_Price_Register_Ireland-28-05-2021.csv
python clean_dataset.py Property_Price_Register_Ireland-28-05-2021.csv
python feature_engineer.py Property_Price_Register_Ireland-28-05-2021_cleaned.csv
python predict.py Property_Price_Register_Ireland-28-05-2021_features.csv
```

> Adjust filenames/arguments as needed — check each script's `argparse` help with `python <script>.py --help`.

## Pipeline

1. **`describe.py`** — Diagnostic exploration of the raw dataset (missing 
   values, data types, formatting issues)
2. **`clean_dataset.py`** — Cleaning and standardisation: column normalisation, 
   Unicode/Gaelic text correction, date parsing, numeric coercion, deduplication
3. **`feature_engineer.py`** — Feature creation (year/month/quarter sold, 
   new-build flags, normalised property type descriptors)
4. **`predict.py`** — ML dataset preparation, multi-model training and 
   comparison, forecasting to 2030, and geospatial visualisation

## Models Evaluated

1) Random Forest
2) Gradient Boosting Regressor (tuned) — **best performer**
3) Linear Regression
4) Polynomial Regression, ARIMA, and Prophet (forecasting only)

**Final model:** Tuned Gradient Boosting Regressor — MAE €18,500, RMSE €27,300, R² 0.29

## Key Findings

- Dublin, Wicklow, and Kildare consistently show the highest average sale prices
- Prices declined 2010–2013 (post-crisis) before recovering steadily from 2014 onward
- County, year sold, and property type are the strongest predictors of price
- Model explanatory power is limited by the dataset's lack of structural 
  features (property size, bedrooms, condition)

## Data Source

- [Property Price Register Ireland (Kaggle)](https://www.kaggle.com/datasets/erinkhoo/property-price-register-ireland)
- Original data from the [Property Services Regulatory Authority (PSRA)](https://www.propertypriceregister.ie)

## Tech Stack

Python · pandas · numpy · scikit-learn · matplotlib · geopandas · folium · statsmodels · Prophet
