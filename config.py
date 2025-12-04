# config.py
"""
Central configuration file for the customer renewal project.
"""

# --- File Paths ---
DATA_FILE = "synthetic_insurance_renewal_data.csv"
MODEL_PATH = "models/"
MODEL_FILE = MODEL_PATH + "renewal_model_catboost.joblib"
EXPLAINER_FILE = MODEL_PATH + "shap_explainer.joblib"
GLOBAL_SHAP_PLOT = MODEL_PATH + "global_shap_summary.png"

# --- Feature Definitions ---
# These are based on your notebook's feature engineering
TARGET_VARIABLE = "Renewed"

# Original columns
POLICY_ID_COL = "Policy_ID"
DATE_COLS = ['Last_Claim_Date', 'Last_Payment_Date', 'Last_Contact_Date']
CATEGORICAL_COLS = ['Gender', 'Region', 'Policy_Type', 'Payment_Frequency', 'Interaction_Channel']
NUMERIC_COLS = [
    'Age', 'Policy_Tenure_Years', 'Number_of_Claims', 'Total_Claim_Amount', 
    'Avg_Time_Between_Claims', 'Annual_Premium', 'On_Time_Payments', 'Missed_Payments', 
    'Total_Paid', 'Credit_Score', 'Support_Tickets', 'Satisfaction_Score', 
    'Customer_Lifetime_Value', 'Has_Previous_Accident'
]

# Engineered features to be created
ENGINEERED_DATE_COLS = ['Days_Since_Last_Payment', 'Days_Since_Last_Contact', 'Days_Since_Last_Claim']
ENGINEERED_RATIO_COLS = ['Payment_Ratio', 'Claim_Rate']

# --- Model & Training ---
# Features to use for training (after engineering)
# We drop original date cols and identifiers
FEATURES_TO_USE = CATEGORICAL_COLS + NUMERIC_COLS + ENGINEERED_DATE_COLS + ENGINEERED_RATIO_COLS

# For Optuna tuning
OPTUNA_TRIALS = 50
OPTUNA_METRIC = 'roc_auc'