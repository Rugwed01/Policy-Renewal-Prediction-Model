# 📊 Customer Renewal Prediction System  
A full-stack Machine Learning system for predicting customer policy renewal, identifying churn risk, and generating actionable retention insights using CatBoost + SHAP + Streamlit.

<p align="center">
  <img src="https://img.shields.io/badge/Machine_Learning-CatBoost-orange?style=for-the-badge&logo=catboost" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-00B3E6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=yellow" />
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/github/stars/Rugwed01/Policy-Renewal-Prediction-Model?style=social" /></a>
  <a href="#"><img src="https://img.shields.io/github/forks/Rugwed01/Policy-Renewal-Prediction-Model?style=social" /></a>
</p>

---

## 🚀 Quick Links

<p align="left">
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Launch_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="model_training.py">
    <img src="https://img.shields.io/badge/Run_Training_Pipeline-1E90FF?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <a href="models/global_shap_summary.png" target="_blank">
    <img src="https://img.shields.io/badge/SHAP_Insights-32CD32?style=for-the-badge&logo=chart-bar&logoColor=white" />
  </a>
</p>

<p align="left">
  <a href="models/renewal_model_catboost.joblib" download>
    <img src="https://img.shields.io/badge/Download_Model-8000FF?style=for-the-badge&logo=download&logoColor=white" />
  </a>
  <a href="synthetic_insurance_renewal_data.csv" download>
    <img src="https://img.shields.io/badge/Download_Dataset-0096FF?style=for-the-badge&logo=database&logoColor=white" />
  </a>
  <a href="https://github.com/your/repo/issues" target="_blank">
    <img src="https://img.shields.io/badge/Report_Issue-red?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

# 📘 Overview  
This project delivers an **end-to-end prediction system** designed for insurance renewal forecasting. It leverages:

- **CatBoost** for high-accuracy prediction  
- **SHAP explainability** for transparent decisions  
- **Streamlit Dashboard** for interactive exploration  
- **Automated feature engineering**  
- **At-risk customer segmentation**  
- **Personalized retention recommendations**

---

# 🧠 Core Components

## 🔧 1. `config.py` — Central Configuration  
Defines file paths, feature groups, target variable, and model metadata.

---

## 📦 2. `data_processing.py` — ETL + Feature Engineering  
Handles:  
- Loading raw CSVs  
- Running data quality checks  
- Transforming date features  
- Creating ratios  
- Building scikit-learn preprocessors  

---

## 🤖 3. `model_training.py` — Model Training Pipeline  
Features include:

- CatBoost final model  
- Class balancing with SMOTE  
- Optuna search for LGBM  
- SHAP global importance plot generation  
- Saves model + SHAP explainer  

---

## 🖥️ 4. `dashboard.py` — Streamlit Dashboard  
Features:  
- At-risk customer list with segmentation  
- SHAP waterfall charts  
- Global feature importance  
- CSV export  
- Upload & predict on new data  

---

## 📁 5. `requirements.txt`  
All dependencies for ML + dashboard.

---

# 📊 System Architecture Diagram

            ┌──────────────────────────┐
            │  synthetic_insurance_*.csv│
            └─────────────┬────────────┘
                          ▼
                ┌──────────────────┐
                │ data_processing.py│
                │  - cleaning       │
                │  - feature engg   │
                └─────────┬────────┘
                          ▼
            ┌──────────────────────────┐
            │    model_training.py      │
            │  - CatBoost model         │
            │  - SHAP explainer         │
            └─────────┬────────────────┘
                      ▼
     ┌────────────────────────────────────┐
     │            models/                 │
     │  - renewal_model_catboost.joblib   │
     │  - shap_explainer.joblib           │
     │  - global_shap_summary.png         │
     └────────────────┬───────────────────┘
                      ▼
       ┌──────────────────────────────┐
       │        dashboard.py          │
       │  At-Risk Finder              │
       │  Deep-Dive SHAP              │
       │  Insights + Upload Predict   │
       └──────────────────────────────┘

---

# 🚀 How to Run

### 1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the model
```bash
python model_training.py
```
### 3️⃣ Launch Streamlit Dashboard
```bash
streamlit run dashboard.py
```

# 📈 Key Features

## 🎯 At-Risk Customer Identification
Find customers most likely to churn (based on probability threshold).

---

## 🔍 Individual Deep-Dive
- SHAP waterfall plots  
- Personalized retention recommendations  

---

## 📈 Global Insights
- SHAP summary bar plot  
- Business interpretations  

---

## 📤 Upload New Data
Automatically computes:
- Predictions  
- Renewal/Churn probabilities  
- Exportable results  

---

## 📥 Downloads
- ✔ Model File  
- ✔ Dataset  
- ✔ SHAP Plot  
- ✔ Predictions CSV  

---

# 🛠️ Tech Stack
- **Python 3.10+**  
- **CatBoost**  
- **LightGBM (Optuna tuning)**  
- **SHAP Explainability**  
- **SMOTE (Imbalanced-learn)**  
- **Streamlit Dashboard**  
- **Scikit-Learn Pipeline**  

---

# 🐞 Issues & Contributions
If you find bugs or want new features:

👉 Submit here:  
https://github.com/Rugwed01/Policy-Renewal-Prediction-Model/issues

---

# 📄 License
MIT License © 2025
