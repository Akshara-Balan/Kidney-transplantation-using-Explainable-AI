import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('synthetic_kidney_transplant_data.csv')

# Feature selection and preprocessing
features = dataset[['Donor_Age', 'Donor_Blood_Type', 'Donor_HLA_Typing', 'Donor_GFR',
                    'Donor_Medical_History', 'Recipient_Age', 'Recipient_Blood_Type',
                    'Recipient_HLA_Typing', 'Recipient_Comorbidities', 'Recipient_Previous_Transplant_History']]
target_graft = dataset['Graft_Survival_Rate']
target_complication = dataset['Complication_Risk']

categorical_cols = ['Donor_Blood_Type', 'Donor_HLA_Typing', 'Donor_Medical_History',
                    'Recipient_Blood_Type', 'Recipient_HLA_Typing', 'Recipient_Comorbidities',
                    'Recipient_Previous_Transplant_History']

# Encoding categorical variables for modeling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Donor_Age', 'Donor_GFR', 'Recipient_Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X = preprocessor.fit_transform(features)

# Splitting the data into training and testing sets
X_train_graft, X_test_graft, y_train_graft, y_test_graft = train_test_split(X, target_graft, test_size=0.2, random_state=42)
X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(X, target_complication, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(kernel='rbf')
}

# Evaluate metrics without SHAP
results_no_shap = {}

for model_name, model in models.items():
    print(f"\nTraining and evaluating {model_name} for graft survival prediction")
   
    model.fit(X_train_graft, y_train_graft)
    y_pred_graft = model.predict(X_test_graft)
   
    mse_graft = mean_squared_error(y_test_graft, y_pred_graft)
    r2_graft = r2_score(y_test_graft, y_pred_graft)
   
    print(f"{model_name} Graft Survival Model Evaluation (No SHAP)")
    print("Mean Squared Error:", mse_graft)
    print("R^2 Score:", r2_graft)
   
    results_no_shap[model_name] = {'MSE_Graft': mse_graft, 'R2_Graft': r2_graft}
   
    print(f"\nTraining and evaluating {model_name} for complication risk prediction")
   
    model.fit(X_train_comp, y_train_comp)
    y_pred_comp = model.predict(X_test_comp)
   
    mse_comp = mean_squared_error(y_test_comp, y_pred_comp)
    r2_comp = r2_score(y_test_comp, y_pred_comp)
   
    print(f"{model_name} Complication Risk Model Evaluation (No SHAP)")
    print("Mean Squared Error:", mse_comp)
    print("R^2 Score:", r2_comp)
   
    results_no_shap[model_name].update({'MSE_Comp': mse_comp, 'R2_Comp': r2_comp})

# SHAP analysis for tree-based models
results_with_shap = results_no_shap.copy()  # Start with results without SHAP

for model_name, model in models.items():
    if model_name in ['Random Forest', 'Gradient Boosting']:
        print(f"\nEvaluating {model_name} with SHAP for graft survival prediction")
       
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_graft = explainer.shap_values(X_test_graft)
            shap.summary_plot(shap_values_graft, X_test_graft, feature_names=preprocessor.get_feature_names_out(), show=False)
            plt.title(f"{model_name} - Graft Survival SHAP Summary Plot")
            plt.show()

            shap_values_comp = explainer.shap_values(X_test_comp)
            shap.summary_plot(shap_values_comp, X_test_comp, feature_names=preprocessor.get_feature_names_out(), show=False)
            plt.title(f"{model_name} - Complication Risk SHAP Summary Plot")
            plt.show()
        except Exception as e:
            print(f"SHAP analysis failed for {model_name}: {e}")

# Compare results before and after adding SHAP
results_df_no_shap = pd.DataFrame(results_no_shap).T
results_df_with_shap = pd.DataFrame(results_with_shap).T

print("\nModel Comparison (No SHAP):")
print(results_df_no_shap)

print("\nModel Comparison (With SHAP):")
print(results_df_with_shap)

# Determine the best model
best_model_graft_no_shap = results_df_no_shap['R2_Graft'].idxmax()
best_model_comp_no_shap = results_df_no_shap['R2_Comp'].idxmax()
best_model_graft_with_shap = results_df_with_shap['R2_Graft'].idxmax()
best_model_comp_with_shap = results_df_with_shap['R2_Comp'].idxmax()

print("\nBest Model for Graft Survival Prediction (No SHAP):", best_model_graft_no_shap)
print("Best Model for Complication Risk Prediction (No SHAP):", best_model_comp_no_shap)

print("\nBest Model for Graft Survival Prediction (With SHAP):", best_model_graft_with_shap)
print("Best Model for Complication Risk Prediction (With SHAP):", best_model_comp_with_shap)
