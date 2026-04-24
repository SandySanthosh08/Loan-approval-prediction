"""
models.py
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
Evaluates them and picks the best one.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        metrics = {
            'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            'Recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            'F1-Score':  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            'ROC-AUC':   round(roc_auc_score(y_test, y_prob) * 100, 2),
            'CM':        confusion_matrix(y_test, y_pred),
            'y_pred':    y_pred,
            'y_prob':    y_prob,
        }
        results[name] = metrics
        trained[name] = model
        print(f"[{name}] Acc={metrics['Accuracy']}% | F1={metrics['F1-Score']}% | AUC={metrics['ROC-AUC']}%")

    # Best model by F1
    best_name = max(results, key=lambda k: results[k]['F1-Score'])
    print(f"\n[BEST MODEL] {best_name}")
    return trained, results, best_name


def get_feature_importance(model, feature_cols):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0])
    else:
        return None
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)


if __name__ == '__main__':
    from data_generator import generate_loan_dataset
    from preprocessing import load_and_preprocess
    df = generate_loan_dataset()
    df.to_csv('loan_dataset.csv', index=False)
    df, X_train, X_test, y_train, y_test, scaler, feature_cols, encoders = load_and_preprocess()
    trained, results, best_name = train_all_models(X_train, X_test, y_train, y_test)
