# data_processing.py
"""
Handles all data loading, validation, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import config

def load_data(filepath):
    """Loads the raw CSV data."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None

def data_quality_checks(df):
    """Performs and reports on basic data quality checks."""
    print("--- Running Data Quality Checks ---")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values per column:\n", missing_values[missing_values > 0])
    
    # Check for duplicates
    duplicates = df.duplicated(subset=[config.POLICY_ID_COL]).sum()
    print(f"\nFound {duplicates} duplicate Policy_IDs.")
    
    # Outlier handling (simple example for 'Annual_Premium')
    # In production, this would be more robust (e.g., clipping or logging)
    q_low = df['Annual_Premium'].quantile(0.01)
    q_hi  = df['Annual_Premium'].quantile(0.99)
    
    outliers = df[(df['Annual_Premium'] < q_low) | (df['Annual_Premium'] > q_hi)].shape[0]
    print(f"\nFound {outliers} potential outliers in 'Annual_Premium' (outside 1st-99th percentile).")
    
    print("--- Data Quality Checks Complete ---")
    return df.drop_duplicates(subset=[config.POLICY_ID_COL])


def engineer_features(df):
    """
    Creates the engineered features our model was trained on.
    This function is used in both training and prediction.
    """
    df_eng = df.copy()
    
    # 1. Handle Dates
    for col in config.DATE_COLS:
        df_eng[col] = pd.to_datetime(df_eng[col], errors='coerce')
    
    # Use the most recent date as the 'snapshot date' for imputation
    snapshot_date = df_eng[config.DATE_COLS].max().max()
    if pd.isna(snapshot_date):
        snapshot_date = pd.to_datetime('today') # Fallback for new data

    df_eng['Days_Since_Last_Payment'] = (snapshot_date - df_eng['Last_Payment_Date']).dt.days
    df_eng['Days_Since_Last_Contact'] = (snapshot_date - df_eng['Last_Contact_Date']).dt.days
    df_eng['Days_Since_Last_Claim'] = (snapshot_date - df_eng['Last_Claim_Date']).dt.days
    
    # 2. Fill Missing Values from Engineering
    # Fill NaNs in 'Days_Since_Last_Claim' (for no claims) with a large number
    # (e.g., tenure in days + a buffer)
    max_tenure_days = (df_eng['Policy_Tenure_Years'].max() * 365) + 365
    df_eng['Days_Since_Last_Claim'] = df_eng['Days_Since_Last_Claim'].fillna(max_tenure_days)
    
    # Fill NaNs in 'Avg_Time_Between_Claims' (for 0 or 1 claim) with 0
    df_eng['Avg_Time_Between_Claims'] = df_eng['Avg_Time_Between_Claims'].fillna(0)
    
    # Handle potential NaNs in other date-engineered cols (if original date was missing)
    for col in config.ENGINEERED_DATE_COLS:
        if df_eng[col].isnull().any():
            df_eng[col] = df_eng[col].fillna(df_eng[col].median())

    # 3. Create Ratios
    total_payments = df_eng['On_Time_Payments'] + df_eng['Missed_Payments']
    df_eng['Payment_Ratio'] = np.where(total_payments > 0, df_eng['On_Time_Payments'] / total_payments, 0)
    
    df_eng['Claim_Rate'] = np.where(df_eng['Policy_Tenure_Years'] > 0, df_eng['Number_of_Claims'] / df_eng['Policy_Tenure_Years'], 0)
    
    return df_eng

def get_preprocessor():
    """
    Returns the scikit-learn ColumnTransformer for preprocessing.
    """
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERIC_COLS + config.ENGINEERED_DATE_COLS + config.ENGINEERED_RATIO_COLS),
            ('cat', categorical_transformer, config.CATEGORICAL_COLS)
        ],
        remainder='drop' # Drop any columns not explicitly defined
    )
    
    return preprocessor