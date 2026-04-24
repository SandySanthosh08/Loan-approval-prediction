from flask import Flask, render_template, request
import numpy as np
import os, shutil

from data_generator import generate_loan_dataset
from preprocessing import load_and_preprocess
from models import train_all_models, get_feature_importance
from visualization import generate_all

app = Flask(__name__)

# ===============================
# STEP 1: RUN PIPELINE
# ===============================
df = generate_loan_dataset()
df.to_csv('loan_dataset.csv', index=False)

df, X_train, X_test, y_train, y_test, scaler, feature_cols, encoders = load_and_preprocess()
trained, results, best_model_name = train_all_models(X_train, X_test, y_train, y_test)

best_model = trained[best_model_name]

# ===============================
# STEP 2: GENERATE CHARTS
# ===============================
generate_all(df, results, best_model_name, best_model, y_test, None)

# Copy charts to static
if not os.path.exists('static/charts'):
    os.makedirs('static/charts')

for file in os.listdir('output_charts'):
    shutil.copy(os.path.join('output_charts', file), 'static/charts')

# ===============================
# ROUTES
# ===============================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', results=results, best=best_model_name)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
        age = float(request.form['age'])
        income = float(request.form['income'])
        credit = float(request.form['credit'])
        loan = float(request.form['loan'])
        term = float(request.form['term'])
        dependents = float(request.form['dependents'])

        # Encode simple categorical inputs
        gender = 1 if request.form['gender'] == 'Male' else 0
        education = 1 if request.form['education'] == 'Graduate' else 0
        employment = 1 if request.form['employment'] == 'Salaried' else 0

        # Derived features (same as preprocessing)
        emi = loan / term
        emi_income = emi / (income / 12 + 1)
        savings = income * 0.2
        savings_loan = savings / (loan + 1)
        credit_income = credit * (income / 100000)

        sample = np.array([[age, gender, education, employment,
                            income, credit, loan, term, 0, 20000,
                            100000, dependents, savings, 0.2, 0.3,
                            income/(dependents+1), emi,
                            emi_income, savings_loan, credit_income]])

        sample_scaled = scaler.transform(sample)

        pred = best_model.predict(sample_scaled)[0]
        result = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)


# ===============================
# RUN
# ===============================
if __name__ == '__main__':
    app.run(debug=True)