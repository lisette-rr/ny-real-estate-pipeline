# New York Real Estate Predictive Pipeline

A Python machine learning workflow for cleaning residential property transaction data, engineering tabular features, and training predictive regression models using Scikit-learn.

## Project Workflow
1. **Data Cleaning & Validation (`data_cleaning.py`):**
   - Handled missing values, categorical encoding, and numeric type conversions on property records.
   - Filtered outliers and resolved distribution anomalies across tabular features.
2. **Exploratory Analysis (`detailed_data_cleaning.ipynb`):**
   - Evaluated correlation between property attributes and sale prices.
3. **Model Training & Evaluation (`train.py`):**
   - Trained and compared Ridge Regression and Random Forest Regressor models.
   - Evaluated performance using 5-fold cross-validation, reporting Mean Absolute Error (MAE) and R-squared (R²) metrics.

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
