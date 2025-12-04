import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Renewal Dashboard",
    page_icon="📊",
    layout="wide"
)
warnings.filterwarnings('ignore')

# --- Feature Engineering Function (THE FIX) ---
# This function replicates the steps from our notebook
def engineer_features(df):
    """
    Creates the engineered features our model was trained on.
    """
    # 1. Handle Dates
    date_cols = ['Last_Claim_Date', 'Last_Payment_Date', 'Last_Contact_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Use the most recent date as the 'snapshot date'
    snapshot_date = df[date_cols].max().max()
    
    df['Days_Since_Last_Payment'] = (snapshot_date - df['Last_Payment_Date']).dt.days
    df['Days_Since_Last_Contact'] = (snapshot_date - df['Last_Contact_Date']).dt.days
    df['Days_Since_Last_Claim'] = (snapshot_date - df['Last_Claim_Date']).dt.days
    
    # 2. Fill Missing Values from Engineering
    # Fill NaNs in 'Days_Since_Last_Claim' (for no claims)
    df['Days_Since_Last_Claim'] = df['Days_Since_Last_Claim'].fillna(df['Policy_Tenure_Years'] * 365 + 365)
    
    # Fill NaNs in 'Avg_Time_Between_Claims' (for 0 or 1 claim)
    df['Avg_Time_Between_Claims'] = df['Avg_Time_Between_Claims'].fillna(0)

    # 3. Create Ratios
    # Create Payment_Ratio (On-Time / Total)
    total_payments = df['On_Time_Payments'] + df['Missed_Payments']
    df['Payment_Ratio'] = np.where(total_payments > 0, df['On_Time_Payments'] / total_payments, 0)
    
    # Create Claim_Rate (Claims / Tenure)
    df['Claim_Rate'] = np.where(df['Policy_Tenure_Years'] > 0, df['Number_of_Claims'] / df['Policy_Tenure_Years'], 0)
    
    return df

# --- Caching ---
@st.cache_data
def load_assets():
    """
    Loads model, raw data, engineers features, and generates predictions.
    """
    # Load the trained model pipeline
    model_pipeline = joblib.load('renewal_model_v2_balanced.joblib')
    
    # Load the raw data
    df_raw = pd.read_csv('synthetic_insurance_renewal_data.csv')
    
    # *** NEW STEP: Apply feature engineering ***
    df_engineered = engineer_features(df_raw.copy())
    
    # Use the pipeline to get probabilities on the *engineered* data
    probabilities = model_pipeline.predict_proba(df_engineered)
    
    # Add the probabilities to our engineered dataframe
    df_engineered['Churn_Probability'] = probabilities[:, 0] # Prob of "Did Not Renew"
    df_engineered['Renewal_Probability'] = probabilities[:, 1] # Prob of "Renewed"
    
    # Return the model and the *fully processed* dataframe
    return model_pipeline, df_engineered

# --- Helper Function ---
def get_feature_impact(model_pipeline, single_customer_data):
    """
    Explains *why* a customer got their score by showing the impact of each feature.
    """
    preprocessor = model_pipeline.named_steps['preprocessor']
    model = model_pipeline.named_steps['model']
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Transform the single customer's data
    processed_data = preprocessor.transform(single_customer_data)
    
    # --- THIS IS THE FIX ---
    # The output is a dense array, so we just select the first row [0]
    # We remove .toarray()
    impacts = processed_data[0] * coefficients
    # --- END FIX ---
    
    impact_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': impacts
    })
    
    impact_df['Absolute_Impact'] = impact_df['Impact'].abs()
    impact_df = impact_df.sort_values('Absolute_Impact', ascending=False).drop(columns=['Absolute_Impact'])
    
    return impact_df

# --- Main Application ---
st.title("📊 Customer Renewal Prediction Dashboard")
st.markdown("This dashboard uses our trained model to predict customer renewal probability and identify at-risk customers.")

# Load data
try:
    model, df = load_assets()
    
    # --- Section 1: High-Risk Customer List ---
    st.header("High-Risk Customer List")
    
    risk_threshold = st.slider(
        "Show customers with a 'Did Not Renew' probability greater than:",
        min_value=0.5,
        max_value=1.0,
        value=0.70,
        step=0.01
    )
    
    high_risk_df = df[df['Churn_Probability'] > risk_threshold]
    high_risk_df_display = high_risk_df[[
        'Policy_ID', 
        'Churn_Probability', 
        'Satisfaction_Score', 
        'Missed_Payments', 
        'Support_Tickets', 
        'Policy_Tenure_Years',
        'Days_Since_Last_Contact' # Changed from Last_Contact_Date
    ]].sort_values('Churn_Probability', ascending=False)
    
    st.dataframe(high_risk_df_display, use_container_width=True)
    st.markdown(f"Found **{len(high_risk_df)} customers** at or above the risk threshold.")

    # --- Section 2: Individual Customer Deep-Dive ---
    st.header("Individual Customer Analysis")
    st.markdown("Select a customer (from the high-risk list or any other) to see their detailed profile.")
    
    all_customers = df['Policy_ID'].unique()
    selected_customer_id = st.selectbox("Select a Customer Policy_ID:", all_customers)
    
    if selected_customer_id:
        customer_data = df[df['Policy_ID'] == selected_customer_id]
        churn_prob = customer_data['Churn_Probability'].iloc[0]
        renew_prob = customer_data['Renewal_Probability'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="**Renewal Probability**",
                value=f"{renew_prob:.1%}",
                delta=f"{(renew_prob - 0.5):.1%}",
                delta_color="normal"
            )
        with col2:
            st.metric(
                label="**Churn Probability**",
                value=f"{churn_prob:.1%}",
                delta=f"{(churn_prob - 0.5):.1%}",
                delta_color="inverse"
            )
        
        st.subheader(f"Key Factors for {selected_customer_id}")
        
        feature_impact_df = get_feature_impact(model, customer_data)
        
        positive_factors = feature_impact_df[feature_impact_df['Impact'] > 0].head(5)
        negative_factors = feature_impact_df[feature_impact_df['Impact'] < 0].head(5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔴 **Top 5 Factors Driving Churn**")
            st.dataframe(negative_factors, use_container_width=True)
        
        with col2:
            st.markdown("🟢 **Top 5 Factors Driving Renewal**")
            st.dataframe(positive_factors, use_container_width=True)

except FileNotFoundError:
    st.error(f"Error: The files 'renewal_model_v2_balanced.joblib' or 'synthetic_insurance_renewal_data.csv' were not found.")
    st.markdown("Please make sure both files are in the same directory as this `dashboard.py` script.")
except Exception as e:
    st.error(f"An error occurred: {e}")