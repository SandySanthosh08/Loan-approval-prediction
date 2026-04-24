"""
data_generator.py
Generates a synthetic but realistic loan approval dataset with 500 samples.
"""

import numpy as np
import pandas as pd

def generate_loan_dataset(n_samples=500, random_state=42):
    np.random.seed(random_state)

    # --- Applicant demographics ---
    ages = np.random.randint(21, 65, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n_samples, p=[0.25, 0.45, 0.22, 0.08]
    )
    employment = np.random.choice(
        ['Salaried', 'Self-Employed', 'Business', 'Unemployed'],
        n_samples, p=[0.50, 0.25, 0.18, 0.07]
    )

    # --- Financial features ---
    annual_income = np.where(
        employment == 'Unemployed',
        np.random.randint(0, 80000, n_samples),
        np.random.randint(180000, 1500000, n_samples)
    )
    credit_score = np.clip(np.random.normal(680, 80, n_samples).astype(int), 300, 900)
    loan_amount = np.random.randint(50000, 5000000, n_samples)
    loan_term = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240], n_samples)
    existing_loans = np.random.randint(0, 5, n_samples)
    monthly_expenses = (annual_income / 12 * np.random.uniform(0.3, 0.8, n_samples)).astype(int)
    assets_value = np.random.randint(0, 10000000, n_samples)
    dependents = np.random.randint(0, 6, n_samples)

    # --- Derived / engineered features ---
    debt_to_income = np.round(
        (loan_amount / 12) / np.where(annual_income == 0, 1, annual_income / 12), 3
    )
    loan_to_asset = np.round(
        loan_amount / np.where(assets_value == 0, 1, assets_value), 3
    )
    savings = np.maximum(0, (annual_income - monthly_expenses * 12) *
                         np.random.uniform(0.5, 3.0, n_samples)).astype(int)

    # --- Loan approval logic (realistic rule-based) ---
    score = np.zeros(n_samples)
    score += np.where(credit_score >= 750, 3,
             np.where(credit_score >= 650, 2,
             np.where(credit_score >= 550, 1, -1)))
    score += np.where(annual_income >= 600000, 2,
             np.where(annual_income >= 300000, 1, -1))
    score += np.where(debt_to_income <= 0.3, 2,
             np.where(debt_to_income <= 0.5, 1, -2))
    score += np.where(employment == 'Salaried', 1,
             np.where(employment == 'Business', 1, 0))
    score += np.where(existing_loans == 0, 1,
             np.where(existing_loans <= 2, 0, -1))
    score += np.where(loan_to_asset <= 0.5, 1, 0)
    score += np.where(education.isin(['Master', 'PhD']) if hasattr(education, 'isin')
                      else np.isin(education, ['Master', 'PhD']), 1, 0)

    # Add noise
    score += np.random.uniform(-1, 1, n_samples)
    loan_approved = (score >= 3).astype(int)

    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Education': education,
        'Employment_Type': employment,
        'Annual_Income': annual_income,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Loan_Term_Months': loan_term,
        'Existing_Loans': existing_loans,
        'Monthly_Expenses': monthly_expenses,
        'Assets_Value': assets_value,
        'Dependents': dependents,
        'Savings': savings,
        'Debt_to_Income_Ratio': debt_to_income,
        'Loan_to_Asset_Ratio': loan_to_asset,
        'Loan_Approved': loan_approved
    })

    return df


if __name__ == '__main__':
    df = generate_loan_dataset()
    df.to_csv('loan_dataset.csv', index=False)
    print(f"Dataset generated: {df.shape}")
    print(df['Loan_Approved'].value_counts())
    print(df.head())
