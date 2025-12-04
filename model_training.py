# model_training.py (Corrected for SHAP 2D Array)
"""
Handles model training, hyperparameter tuning, and saving of
the final model and SHAP explainer.
"""
import pandas as pd
import joblib
import os
import shutil
import shap
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Import from our custom modules
import config
import data_processing

# ... (train_lgbm_with_optuna and train_final_model functions are unchanged) ...
# ... (Copy them from the previous response) ...

def train_lgbm_with_optuna(X_train, y_train):
    """
    Uses Optuna to find the best hyperparameters for LGBM.
    """
    print("\n--- Starting Optuna Hyperparameter Tuning for LGBM ---")
    
    # Get the standard preprocessor
    preprocessor = data_processing.get_preprocessor()

    # Create the full pipeline *for tuning* (preprocessor + model)
    lgbm_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
    ])

    def objective(trial):
        # Define the search space for hyperparameters
        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
            'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 50),
            'model__max_depth': trial.suggest_int('model__max_depth', 5, 15),
            'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.0, 1.0),
            'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.0, 1.0),
        }
        
        lgbm_pipeline.set_params(**params)
        
        # Evaluate the model using cross-validation
        score = cross_val_score(lgbm_pipeline, X_train, y_train, n_jobs=-1, cv=3, scoring=config.OPTUNA_METRIC)
        return score.mean()

    # Run the Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS)

    print(f"Best trial {config.OPTUNA_METRIC}: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def train_final_model(X_train, X_test, y_train, y_test):
    """
    Trains the final, production-ready model (CatBoost).
    """
    print("\n--- Training Final CatBoost Model ---")
    
    y_counts = y_train.value_counts()
    scale_pos_weight = y_counts[0] / y_counts[1]

    final_model_simple = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        scale_pos_weight=scale_pos_weight, 
        cat_features=config.CATEGORICAL_COLS
    )
    
    final_pipeline = Pipeline(steps=[('model', final_model_simple)])
    
    final_pipeline.fit(X_train[config.FEATURES_TO_USE], y_train, 
                       model__eval_set=(X_test[config.FEATURES_TO_USE], y_test))
    
    return final_pipeline

# ##################################################################
# #####               THIS FUNCTION IS UPDATED                 #####
# ##################################################################
def save_model_and_explainer(pipeline, X_train, model_filepath, explainer_filepath):
    """
    Saves the trained model pipeline and a SHAP explainer.
    """
    print("\n--- Saving Model and SHAP Explainer ---")
    
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
        
    # 1. Save the model pipeline
    joblib.dump(pipeline, model_filepath)
    print(f"Model saved to {model_filepath}")

    # 2. Create and save the SHAP Explainer
    model = pipeline.named_steps['model']
    X_train_processed = X_train[config.FEATURES_TO_USE]

    print("Initializing SHAP TreeExplainer with feature_perturbation='tree_path_dependent'...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    
    # 3. Save the explainer
    joblib.dump(explainer, explainer_filepath)
    print(f"SHAP explainer saved to {explainer_filepath}")
    
    # 4. Create and save a global SHAP summary plot
    print("Generating global feature importance plot...")
    
    # We pass the data here to get the actual SHAP values
    X_sample = X_train_processed.sample(500, random_state=42)
    shap_values = explainer(X_sample)
    
    # --- THIS IS THE FIX ---
    # The error confirmed shap_values.values is 2D (samples, features)
    # We no longer need to index for class 1 [:,:,1]
    
    plt.figure()
    shap.summary_plot(shap_values.values, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP) - for 'Renewed' (Class 1)")
    plt.tight_layout()
    plt.savefig(config.GLOBAL_SHAP_PLOT)
    plt.close()
    # --- END FIX ---
    
    print(f"Global SHAP plot saved to {config.GLOBAL_SHAP_PLOT}")


def main():
    """Main training script."""
    
    if os.path.exists(config.MODEL_PATH):
        print(f"Removing old model directory: {config.MODEL_PATH}")
        shutil.rmtree(config.MODEL_PATH)
        
    # 1. Load and Process Data
    df_raw = data_processing.load_data(config.DATA_FILE)
    df_clean = data_processing.data_quality_checks(df_raw)
    df_features = data_processing.engineer_features(df_clean)
    
    # 2. Define Features (X) and Target (y)
    X = df_features[config.FEATURES_TO_USE]
    y = df_features[config.TARGET_VARIABLE]
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train Final Model (CatBoost)
    final_model_pipeline = train_final_model(X_train, X_test, y_train, y_test)
    
    # 6. Evaluate Final Model
    print("\n--- Final Model Evaluation (on Test Set) ---")
    preds = final_model_pipeline.predict(X_test[config.FEATURES_TO_USE])
    probs = final_model_pipeline.predict_proba(X_test[config.FEATURES_TO_USE])[:, 1]
    
    print(classification_report(y_test, preds, target_names=["Did Not Renew (0)", "Renewed (1)"]))
    print(f"Test Set ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    
    # 7. Save Model and Explainer
    save_model_and_explainer(
        final_model_pipeline, 
        X_train, 
        config.MODEL_FILE, 
        config.EXPLAINER_FILE
    )
    
    print("\n--- Training Pipeline Complete ---")

if __name__ == "__main__":
    main()