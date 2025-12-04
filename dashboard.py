# dashboard.py (Corrected for Altair SchemaValidationError)
"""
Enhanced Streamlit dashboard for Customer Renewal Prediction.
"""
import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# Import matplotlib after pandas to avoid conflicts
import matplotlib.pyplot as plt

# Import from our custom modules
import config
from data_processing import load_data, engineer_features

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Renewal Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Caching ---
@st.cache_resource
def load_assets():
    """
    Loads the model, SHAP explainer, and raw data for cohort analysis.
    """
    try:
        model_pipeline = joblib.load(config.MODEL_FILE)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {config.MODEL_FILE}")
        return None, None, None, None
        
    try:
        shap_explainer = joblib.load(config.EXPLAINER_FILE)
    except FileNotFoundError:
        st.error(f"Error: SHAP explainer not found at {config.EXPLAINER_FILE}")
        return model_pipeline, None, None, None

    df_raw = load_data(config.DATA_FILE)
    if df_raw is not None:
        df_full_processed = engineer_features(df_raw)
        # Ensure we're working with DataFrames throughout
        X = df_full_processed[config.FEATURES_TO_USE].copy()
        
        # Check if predictions are already cached in session state
        if 'predictions' not in st.session_state:
            print("Calculating predictions for all data...")
            probs = model_pipeline.predict_proba(X)
            st.session_state.predictions = probs
        else:
            print("Using cached predictions.")
            probs = st.session_state.predictions
            
        # Ensure we're adding columns to a DataFrame, not a numpy array
        # Use pd.concat to ensure we maintain DataFrame properties
        prob_df = pd.DataFrame({
            'Churn_Probability': probs[:, 0],
            'Renewal_Probability': probs[:, 1]
        })
        df_full_processed = pd.concat([df_full_processed, prob_df], axis=1)
    else:
        df_full_processed = None
        
    try:
        global_shap_plot = plt.imread(config.GLOBAL_SHAP_PLOT)
    except FileNotFoundError:
        global_shap_plot = None

    return model_pipeline, shap_explainer, df_full_processed, global_shap_plot


@st.cache_data
def get_shap_explanation(_shap_explainer, customer_data_processed):
    """
    Generates SHAP values for a single customer.
    We cache this computation.
    """
    shap_values = _shap_explainer(customer_data_processed)
    return shap_values

def get_retention_recommendations(shap_explanation_sample, feature_names):
    """
    Generates personalized retention recommendations based on top negative features.
    """
    # Convert feature_names to a list if it's not already
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
        
    # Create DataFrame with explicit parameter names to avoid confusion
    data = list(zip(feature_names, shap_explanation_sample.values))
    # Use a different approach to create the DataFrame to avoid conflicts
    feature_shap_df = pd.DataFrame(data)
    feature_shap_df.columns = ['feature', 'shap_value']
    
    # We are looking for the most NEGATIVE SHAP values,
    # as these are the biggest drivers *against* renewal (i.e., for churn).
    feature_shap_df = feature_shap_df.sort_values(by='shap_value', ascending=True)
    
    top_churn_drivers = feature_shap_df.head(3)
    recommendations = []

    for _, row in top_churn_drivers.iterrows():
        feature = row['feature']
        if row['shap_value'] < 0: # Only list features that are actively driving churn
            if feature == 'Days_Since_Last_Contact':
                recommendations.append(
                    "**Proactive Outreach:** This customer hasn't been contacted recently. "
                    "Schedule a personalized check-in call or email to discuss their policy."
                )
            elif feature == 'Missed_Payments' or feature == 'Payment_Ratio':
                recommendations.append(
                    "**Payment Flexibility:** Payment issues are a key driver. "
                    "Offer a flexible payment plan, a one-time grace period, or autopay setup assistance."
                )
            elif feature == 'Satisfaction_Score':
                recommendations.append(
                    "**Service Recovery:** Low satisfaction is a major risk. "
                    "Escalate to a senior support agent to resolve any outstanding issues immediately."
                )
            elif feature == 'Support_Tickets':
                recommendations.append(
                    "**Issue Resolution:** High support ticket volume suggests friction. "
                    "Review their support history and confirm all issues are fully resolved."
                )
            elif feature == 'Claim_Rate':
                recommendations.append(
                    "**Policy Review:** A high claim rate might be driving up their premium. "
                    "Offer a policy review to ensure their coverage matches their needs and explore potential discounts."
                )

    if not recommendations:
        recommendations.append("This customer's churn risk is not driven by the usual key factors. Review profile manually for custom outreach.")
        
    return recommendations

# --- Main Application ---
st.title("📊 Customer Renewal Prediction Dashboard")
st.markdown("Using machine learning (CatBoost + SHAP) to identify at-risk customers and guide retention efforts.")

model, explainer, df_all, global_shap_plot = load_assets()

# Ensure df_all is a DataFrame
if df_all is not None and not isinstance(df_all, pd.DataFrame):
    df_all = pd.DataFrame(df_all)

if model is None or df_all is None:
    st.warning("Could not load all assets. Please run the model_training.py script first.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 At-Risk Customer Hunter", 
        "🔍 Individual Deep-Dive", 
        "📈 Global Insights",
        "🚀 Predict on New Data"
    ])

    # --- Tab 1: At-Risk Customer Hunter ---
    with tab1:
        st.header("🎯 High-Risk Customer List")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_threshold = st.slider(
                "Show customers with a 'Did Not Renew' probability greater than:",
                min_value=0.5,
                max_value=1.0,
                value=0.70,
                step=0.01
            )
            
            segment_by = st.selectbox(
                "Segment by:",
                ('None', 'Policy_Type', 'Region', 'Satisfaction_Score')
            )

        # Ensure df_all is a DataFrame before using it
        if not isinstance(df_all, pd.DataFrame):
            df_all = pd.DataFrame(df_all)
            
        high_risk_df = df_all[df_all['Churn_Probability'] > risk_threshold]
        
        with col2:
            st.metric("Total At-Risk Customers", len(high_risk_df))
            st.markdown(f"Found **{len(high_risk_df)} customers** at or above the {risk_threshold*100:.0f}% risk threshold.")

        display_cols = [
            'Policy_ID', 'Churn_Probability', 'Satisfaction_Score', 'Missed_Payments', 
            'Support_Tickets', 'Policy_Tenure_Years', 'Days_Since_Last_Contact'
        ]
        # Ensure high_risk_df is a DataFrame before using it
        if not isinstance(high_risk_df, pd.DataFrame):
            high_risk_df = pd.DataFrame(high_risk_df)
            
        # Use a different approach to sort the DataFrame to avoid conflicts
        sorted_df = high_risk_df[display_cols].copy()
        # Convert to DataFrame to ensure it's a DataFrame
        sorted_df = pd.DataFrame(sorted_df)
        # Use a different approach to sort
        sorted_df = sorted_df.sort_values(by=['Churn_Probability'], ascending=[False])
        st.dataframe(
            sorted_df,
            use_container_width=True
        )
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(high_risk_df[display_cols])
        st.download_button(
            label="📥 Download At-Risk List (CSV)",
            data=csv,
            file_name="at_risk_customers.csv",
            mime="text/csv",
        )
        
        # ##################################################################
        # #####               THIS BLOCK IS UPDATED                  #####
        # ##################################################################
        if segment_by != 'None':
            st.subheader(f"At-Risk Customers by {segment_by}")
            # Ensure high_risk_df is a DataFrame before using it
            if not isinstance(high_risk_df, pd.DataFrame):
                high_risk_df = pd.DataFrame(high_risk_df)
                
            if segment_by == 'Satisfaction_Score':
                # Binning for a continuous variable
                bins = pd.cut(high_risk_df[segment_by], bins=range(0, 11))
                chart_data = high_risk_df.groupby(bins, observed=True).size()
                
                # --- THIS IS THE FIX ---
                # st.bar_chart (Altair) cannot plot pandas.Interval objects.
                # We must convert the index (the bins) to strings first.
                chart_data.index = chart_data.index.astype(str)
                # --- END FIX ---
            else:
                chart_data = high_risk_df[segment_by].value_counts()
            
            st.bar_chart(chart_data)
        # ##################################################################


    # --- Tab 2: Individual Customer Deep-Dive ---
    with tab2:
        st.header("🔍 Individual Customer Analysis")
        st.markdown("Select any customer to see their renewal prediction and *why* the model scored them that way.")
        
        # Ensure df_all is a DataFrame before using it
        if not isinstance(df_all, pd.DataFrame):
            df_all = pd.DataFrame(df_all)
            
        all_customers = df_all[config.POLICY_ID_COL].unique()
        selected_customer_id = st.selectbox("Select a Customer Policy_ID:", all_customers)
        
        if selected_customer_id and explainer:
            customer_data = df_all[df_all[config.POLICY_ID_COL] == selected_customer_id]
            # Ensure customer_data is a DataFrame before using it
            if not isinstance(customer_data, pd.DataFrame):
                customer_data = pd.DataFrame(customer_data)
                
            customer_processed = customer_data[config.FEATURES_TO_USE]
            
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
                    label="**Churn Probability (Risk)**",
                    value=f"{churn_prob:.1%}",
                    delta=f"{(churn_prob - 0.5):.1%}",
                    delta_color="inverse"
                )
            
            st.subheader(f"Explanation for {selected_customer_id}")
            st.markdown("This plot shows the factors pushing the prediction towards 'Renewed' (positive, blue) or 'Did Not Renew' (negative, red).")
            
            with st.spinner("Generating explanation..."):
                shap_values_obj = get_shap_explanation(explainer, customer_processed)
                shap_explanation_sample = shap_values_obj[0]
                
                fig, ax = plt.subplots()
                # Fix for SHAP API change: Pass the Explanation object directly
                shap.plots.waterfall(shap_explanation_sample, max_display=10, show=False)
                st.pyplot(fig)
            
            st.subheader("💡 Personalized Retention Recommendations")
            recommendations = get_retention_recommendations(shap_explanation_sample, config.FEATURES_TO_USE)
            for rec in recommendations:
                st.markdown(f"- {rec}")
                
            st.subheader("Raw Customer Data")
            st.dataframe(customer_data)

    # --- Tab 3: Global Insights ---
    with tab3:
        st.header("📈 Global Business & Model Insights")
        
        if global_shap_plot is not None:
            st.subheader("Top Drivers of Renewal (Model-Wide)")
            st.image(global_shap_plot, use_container_width=True, 
                     caption="This SHAP plot shows the average impact of each feature on the model's output (for Class 1: 'Renewed').")
            
            st.subheader("Key Business Takeaways")
            st.markdown("""
            * **Positive drivers (top)** push customers to **RENEW**.
            * **Negative drivers (bottom)** push customers to **CHURN**.
            * **Engagement is Critical:** **'Days_Since_Last_Contact'** being *low* is a top driver for renewal. Customers we talk to stay.
            * **Payment Friction is a Killer:** **'Payment_Ratio'** and **'Missed_Payments'** are top churn drivers (appearing as strong negative bars).
            * **Service Matters:** **'Satisfaction_Score'** is a top driver for renewal.
            """)
        else:
            st.warning("Global SHAP plot not found. Please run `model_training.py`.")

    # --- Tab 4: On-the-Fly Prediction ---
    with tab4:
        st.header("🚀 Predict on New Data")
        st.markdown("Upload a CSV file with new customer data to get instant renewal predictions.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                new_data_raw = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(new_data_raw)} records.")
                
                with st.spinner("Processing features and making predictions..."):
                    new_data_processed = engineer_features(new_data_raw)
                    new_data_features = new_data_processed[config.FEATURES_TO_USE]
                    
                    new_probs = model.predict_proba(new_data_features)
                    new_preds = model.predict(new_data_features)
                    
                    new_data_raw['Churn_Probability'] = new_probs[:, 0]
                    new_data_raw['Renewal_Probability'] = new_probs[:, 1]
                    new_data_raw['Prediction'] = np.where(new_preds == 0, 'Did Not Renew', 'Renewed')
                
                st.subheader("Prediction Results")
                st.dataframe(new_data_raw[[config.POLICY_ID_COL, 'Prediction', 'Churn_Probability', 'Renewal_Probability']])
                
                @st.cache_data
                def convert_df_to_csv_results(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_results = convert_df_to_csv_results(new_data_raw)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv_results,
                    file_name="new_customer_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your CSV has the required columns: " + ", ".join(config.CATEGORICAL_COLS + config.NUMERIC_COLS + config.DATE_COLS))