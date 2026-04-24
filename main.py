"""
main.py
Main entry point — runs the entire Loan Approval Analytics pipeline.
"""

import os
import sys

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_loan_dataset
from preprocessing import load_and_preprocess
from models import train_all_models, get_feature_importance
from visualization import generate_all

print("=" * 65)
print("   LOAN APPROVAL PREDICTION — DATA ANALYTICS SYSTEM")
print("=" * 65)

# Step 1: Generate dataset
print("\n[STEP 1] Generating Dataset...")
df_raw = generate_loan_dataset(n_samples=500)
df_raw.to_csv('loan_dataset.csv', index=False)
print(f"         Saved 'loan_dataset.csv' with {len(df_raw)} records.")

# Step 2: Preprocess
print("\n[STEP 2] Preprocessing Data...")
df, X_train, X_test, y_train, y_test, scaler, feature_cols, encoders = load_and_preprocess()

# Step 3: Train & evaluate models
print("\n[STEP 3] Training Models...")
trained, results, best_name = train_all_models(X_train, X_test, y_train, y_test)

# Step 4: Feature importance
importance = get_feature_importance(trained[best_name], feature_cols)

# Step 5: Visualizations
print("\n[STEP 4] Generating Visualizations...")
chart_paths = generate_all(df, results, best_name, trained[best_name], y_test, importance)

print("\n" + "=" * 65)
print(f"  BEST MODEL  : {best_name}")
print(f"  Accuracy    : {results[best_name]['Accuracy']}%")
print(f"  Precision   : {results[best_name]['Precision']}%")
print(f"  Recall      : {results[best_name]['Recall']}%")
print(f"  F1-Score    : {results[best_name]['F1-Score']}%")
print(f"  ROC-AUC     : {results[best_name]['ROC-AUC']}%")
print("=" * 65)
print(f"\nAll charts saved in: output_charts/")
print("Pipeline complete!\n")
