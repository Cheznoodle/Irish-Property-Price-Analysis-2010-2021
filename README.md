# Irish Property Price Analysis & Prediction (2010–2021)

Analysis and machine learning prediction of residential property prices in 
Ireland using the official Property Price Register (PSRA) dataset, covering 
~476,000 transactions from 2010 to 2021.

## Overview

This project explores how residential property prices in Ireland have evolved 
over the past decade, how trends differ across counties, and whether a 
regression model can estimate property prices based on location, sale date, 
and available property characteristics.

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

- Random Forest
- Gradient Boosting Regressor (tuned) — **best performer**
- Linear Regression
- Polynomial Regression, ARIMA, and Prophet (forecasting only)

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
