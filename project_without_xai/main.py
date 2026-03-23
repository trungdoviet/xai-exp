"""
============================================================
  PROJECT 1: Loan Prediction WITHOUT XAI (Black Box)
============================================================
This script trains a Random Forest model on synthetic loan data
and outputs predictions with NO explanation whatsoever.

The user experience is: "Computer says NO." — No reason given.
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# 1. Generate Synthetic Loan Data
# ──────────────────────────────────────────────

def generate_loan_data(n_samples=1000, random_state=42):
    """
    Generate synthetic loan application data with realistic features.
    
    Features:
        - income: Annual income ($20K - $200K)
        - credit_score: Credit score (300 - 850)
        - debt_ratio: Debt-to-income ratio (0.0 - 1.0)
        - employment_years: Years at current job (0 - 30)
        - loan_amount: Requested loan amount ($1K - $100K)
    
    Target:
        - approved: 1 = Approved, 0 = Denied
    """
    np.random.seed(random_state)
    
    income = np.random.uniform(20000, 200000, n_samples)
    credit_score = np.random.randint(300, 851, n_samples)
    debt_ratio = np.random.uniform(0.0, 1.0, n_samples)
    employment_years = np.random.randint(0, 31, n_samples)
    loan_amount = np.random.uniform(1000, 100000, n_samples)
    
    # Create realistic approval logic
    approval_score = (
        (credit_score - 300) / 550 * 0.35 +           # Credit score (35%)
        (1 - debt_ratio) * 0.30 +                       # Low debt ratio (30%)
        (income / 200000) * 0.20 +                      # High income (20%)
        (employment_years / 30) * 0.10 +                # Job stability (10%)
        (1 - loan_amount / 100000) * 0.05 +             # Low loan amount (5%)
        np.random.normal(0, 0.1, n_samples)             # Random noise
    )
    
    approved = (approval_score > 0.45).astype(int)
    
    data = pd.DataFrame({
        'income': np.round(income, 2),
        'credit_score': credit_score,
        'debt_ratio': np.round(debt_ratio, 3),
        'employment_years': employment_years,
        'loan_amount': np.round(loan_amount, 2),
        'approved': approved
    })
    
    return data


# ──────────────────────────────────────────────
# 2. Train the Black-Box Model
# ──────────────────────────────────────────────

def train_model(data):
    """Train a Random Forest classifier — the classic 'black box'."""
    
    features = ['income', 'credit_score', 'debt_ratio', 'employment_years', 'loan_amount']
    X = data[features]
    y = data['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_test, y_test, accuracy


# ──────────────────────────────────────────────
# 3. Make Predictions — NO EXPLANATIONS
# ──────────────────────────────────────────────

def predict_without_explanation(model, X_test, y_test):
    """
    Make predictions with ZERO explanation.
    This is the black-box experience.
    """
    
    predictions = model.predict(X_test)
    
    print("=" * 60)
    print("  LOAN PREDICTION RESULTS — BLACK BOX MODEL (No XAI)")
    print("=" * 60)
    print()
    print("  ⚠️  No explanations provided. Just raw decisions.")
    print()
    print("-" * 60)
    
    # Show first 10 applicants
    for i in range(min(10, len(X_test))):
        row = X_test.iloc[i]
        prediction = predictions[i]
        actual = y_test.iloc[i]
        result = "✅ APPROVED" if prediction == 1 else "❌ DENIED"
        
        print(f"\n  Applicant #{i+1}:")
        print(f"    Income:           ${row['income']:>12,.2f}")
        print(f"    Credit Score:     {row['credit_score']:>12.0f}")
        print(f"    Debt Ratio:       {row['debt_ratio']:>12.3f}")
        print(f"    Employment Years: {row['employment_years']:>12.0f}")
        print(f"    Loan Amount:      ${row['loan_amount']:>12,.2f}")
        print(f"    ──────────────────────────────────")
        print(f"    Decision:         {result}")
        print(f"    Reason:           ??? (NO EXPLANATION AVAILABLE)")
        print(f"    Actual:           {'APPROVED' if actual == 1 else 'DENIED'}")
    
    print("\n" + "=" * 60)
    print("  MODEL PERFORMANCE")
    print("=" * 60)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n  Accuracy: {accuracy:.2%}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, predictions, target_names=['Denied', 'Approved'])
    for line in report.split('\n'):
        print(f"    {line}")
    
    print("\n" + "=" * 60)
    print("  ❓ WHY were these decisions made?")
    print("  ❓ WHAT factors were most important?")
    print("  ❓ HOW can applicants improve their chances?")
    print("  ❓ IS the model fair and unbiased?")
    print()
    print("  ➡️  We DON'T KNOW. That's the black-box problem.")
    print("  ➡️  See 'project_with_xai/' for the solution!")
    print("=" * 60)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔄 Generating synthetic loan data...")
    data = generate_loan_data(n_samples=1000)
    print(f"   Generated {len(data)} loan applications")
    print(f"   Approval rate: {data['approved'].mean():.1%}")
    
    print("\n🔄 Training Random Forest model (black box)...")
    model, X_test, y_test, accuracy = train_model(data)
    print(f"   Model accuracy: {accuracy:.2%}")
    
    print("\n🔄 Making predictions...\n")
    predict_without_explanation(model, X_test, y_test)
