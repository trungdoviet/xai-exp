"""
============================================================
  PROJECT 2: Loan Prediction WITH XAI (Explainable AI)
============================================================
Same model as Project 1, but NOW with full explanations using:
  1. Feature Importance (built-in from Random Forest)
  2. SHAP (SHapley Additive exPlanations)
  3. LIME (Local Interpretable Model-agnostic Explanations)

The user experience: Every decision comes with a clear reason.
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore')

# Output directory for charts
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Generate Synthetic Loan Data (SAME as Project 1)
# ──────────────────────────────────────────────

def generate_loan_data(n_samples=1000, random_state=42):
    """
    Generate synthetic loan application data with realistic features.
    IDENTICAL to Project 1 to ensure fair comparison.
    """
    np.random.seed(random_state)

    income = np.random.uniform(20000, 200000, n_samples)
    credit_score = np.random.randint(300, 851, n_samples)
    debt_ratio = np.random.uniform(0.0, 1.0, n_samples)
    employment_years = np.random.randint(0, 31, n_samples)
    loan_amount = np.random.uniform(1000, 100000, n_samples)

    # Same approval logic as Project 1
    approval_score = (
        (credit_score - 300) / 550 * 0.35 +
        (1 - debt_ratio) * 0.30 +
        (income / 200000) * 0.20 +
        (employment_years / 30) * 0.10 +
        (1 - loan_amount / 100000) * 0.05 +
        np.random.normal(0, 0.1, n_samples)
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
# 2. Train the Model (SAME as Project 1)
# ──────────────────────────────────────────────

def train_model(data):
    """Train the SAME Random Forest model as Project 1."""

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

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_train, X_test, y_test, accuracy, features


# ──────────────────────────────────────────────
# 3. XAI Technique #1: Feature Importance
# ──────────────────────────────────────────────

def explain_feature_importance(model, features):
    """
    Extract and visualize the built-in feature importance
    from the Random Forest model.

    This is a GLOBAL explanation — shows what features
    matter most across ALL predictions.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n" + "=" * 60)
    print("  📊 XAI TECHNIQUE #1: FEATURE IMPORTANCE (Global)")
    print("=" * 60)
    print("\n  How important is each feature to the model overall?\n")

    for i, idx in enumerate(sorted_idx):
        bar = "█" * int(importances[idx] * 50)
        print(f"    {i+1}. {features[idx]:<20s} {importances[idx]:.4f}  {bar}")

    # Save chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax.barh(
        [features[i] for i in sorted_idx[::-1]],
        [importances[i] for i in sorted_idx[::-1]],
        color=colors
    )
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('🌲 Random Forest — Feature Importance (Global XAI)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar_item in bars:
        width = bar_item.get_width()
        ax.text(width + 0.005, bar_item.get_y() + bar_item.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  💾 Chart saved: {filepath}")

    return importances


# ──────────────────────────────────────────────
# 4. XAI Technique #2: SHAP Explanations
# ──────────────────────────────────────────────

def explain_with_shap(model, X_train, X_test, features, n_display=5):
    """
    Use SHAP (SHapley Additive exPlanations) to explain predictions.

    SHAP provides BOTH global and local explanations:
    - Global: Which features are most important overall?
    - Local: Why was THIS specific applicant approved/denied?
    """
    import shap

    print("\n" + "=" * 60)
    print("  📊 XAI TECHNIQUE #2: SHAP EXPLANATIONS")
    print("=" * 60)

    # Use TreeExplainer (optimized for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    # Handle different SHAP output formats across versions:
    # - Older: list of 2D arrays [class_0_array, class_1_array]
    # - Newer: 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_values_raw, list):
        shap_vals = np.array(shap_values_raw[1])
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_vals = shap_values_raw[:, :, 1]
    else:
        shap_vals = np.array(shap_values_raw)

    # Ensure 2D: (n_samples, n_features)
    shap_vals = np.array(shap_vals, dtype=float)

    # ── Global SHAP Summary Plot ──
    print("\n  🌍 Global SHAP Summary (all predictions):")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test, feature_names=features, show=False)
    plt.title('SHAP Summary Plot — Feature Impact on Loan Approval', fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'shap_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  💾 Chart saved: {filepath}")

    # ── Local SHAP Waterfall Plots (per applicant) ──
    print(f"\n  🔍 Local SHAP Explanations (first {n_display} applicants):\n")

    predictions = model.predict(X_test)

    for i in range(min(n_display, len(X_test))):
        row = X_test.iloc[i]
        prediction = predictions[i]
        result = "✅ APPROVED" if prediction == 1 else "❌ DENIED"

        print(f"  ── Applicant #{i+1}: {result} ──")
        print(f"    Income: ${row['income']:,.2f} | Credit: {row['credit_score']:.0f} | "
              f"Debt Ratio: {row['debt_ratio']:.3f} | Emp Years: {row['employment_years']:.0f} | "
              f"Loan: ${row['loan_amount']:,.2f}")
        print(f"    SHAP Feature Contributions:")

        shap_row = shap_vals[i]
        feature_shap = [(feat, float(val)) for feat, val in zip(features, shap_row)]
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, val in feature_shap:
            direction = "↑ APPROVE" if val > 0 else "↓ DENY   "
            bar = "█" * int(abs(val) * 30)
            print(f"      {feat:<20s} {val:>+8.4f}  {direction}  {bar}")

        # Save waterfall plot
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1])
            else:
                base_val = float(base_val)
            explanation = shap.Explanation(
                values=shap_vals[i].astype(float),
                base_values=base_val,
                data=X_test.iloc[i].values,
                feature_names=features
            )
            shap.plots.waterfall(explanation, show=False)
            plt.title(f'SHAP Waterfall — Applicant #{i+1} ({result})', fontsize=12, fontweight='bold')
            plt.tight_layout()
            filepath = os.path.join(OUTPUT_DIR, f'shap_waterfall_applicant_{i+1}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"    💾 Chart saved: {filepath}")
        except Exception as e:
            print(f"    ⚠️  Waterfall plot skipped: {e}")

        print()

    return shap_vals


# ──────────────────────────────────────────────
# 5. XAI Technique #3: LIME Explanations
# ──────────────────────────────────────────────

def explain_with_lime(model, X_train, X_test, features, n_display=5):
    """
    Use LIME (Local Interpretable Model-agnostic Explanations)
    to explain individual predictions.

    LIME creates a LOCAL surrogate model (simple linear model)
    around each prediction to explain it.
    """
    from lime.lime_tabular import LimeTabularExplainer

    print("\n" + "=" * 60)
    print("  📊 XAI TECHNIQUE #3: LIME EXPLANATIONS (Local)")
    print("=" * 60)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=features,
        class_names=['Denied', 'Approved'],
        mode='classification',
        random_state=42
    )

    predictions = model.predict(X_test)

    print(f"\n  🔍 LIME Explanations (first {n_display} applicants):\n")

    for i in range(min(n_display, len(X_test))):
        row = X_test.iloc[i]
        prediction = predictions[i]
        result = "✅ APPROVED" if prediction == 1 else "❌ DENIED"

        explanation = explainer.explain_instance(
            data_row=row.values,
            predict_fn=model.predict_proba,
            num_features=len(features)
        )

        print(f"  ── Applicant #{i+1}: {result} ──")
        print(f"    LIME Feature Contributions:")

        lime_list = explanation.as_list()
        for feat_desc, weight in lime_list:
            direction = "↑ APPROVE" if weight > 0 else "↓ DENY   "
            bar = "█" * int(abs(weight) * 30)
            print(f"      {feat_desc:<35s} {weight:>+8.4f}  {direction}  {bar}")

        # Save LIME plot
        try:
            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(10, 5)
            plt.title(f'LIME Explanation — Applicant #{i+1} ({result})', fontsize=12, fontweight='bold')
            plt.tight_layout()
            filepath = os.path.join(OUTPUT_DIR, f'lime_explanation_applicant_{i+1}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"    💾 Chart saved: {filepath}")
        except Exception as e:
            print(f"    ⚠️  LIME plot skipped: {e}")

        print()


# ──────────────────────────────────────────────
# 6. Human-Readable Explanations + Actionable Advice
# ──────────────────────────────────────────────

def generate_human_explanations(model, X_test, y_test, features, shap_values, n_display=5):
    """
    Generate simple, human-readable explanations that any
    non-technical person can understand.

    This is what the END USER sees — not SHAP values,
    but plain English reasoning + actionable advice.
    """
    predictions = model.predict(X_test)

    print("\n" + "=" * 60)
    print("  💬 HUMAN-READABLE EXPLANATIONS & ACTIONABLE ADVICE")
    print("=" * 60)

    # Define human-friendly thresholds
    thresholds = {
        'credit_score': {'good': 700, 'fair': 600, 'poor': 500},
        'debt_ratio': {'good': 0.3, 'fair': 0.5, 'poor': 0.7},
        'income': {'good': 80000, 'fair': 50000, 'poor': 30000},
        'employment_years': {'good': 5, 'fair': 2, 'poor': 0},
        'loan_amount': {'good': 20000, 'fair': 50000, 'poor': 80000}
    }

    for i in range(min(n_display, len(X_test))):
        row = X_test.iloc[i]
        prediction = predictions[i]
        actual = y_test.iloc[i]
        result = "✅ APPROVED" if prediction == 1 else "❌ DENIED"

        print(f"\n  {'─' * 56}")
        print(f"  Applicant #{i+1}: {result}")
        print(f"  {'─' * 56}")

        # ── Reasons ──
        shap_row = shap_values[i]
        feature_shap = sorted([(f, float(v)) for f, v in zip(features, shap_row)], key=lambda x: abs(x[1]), reverse=True)

        reasons_for = []
        reasons_against = []

        for feat, val in feature_shap:
            feat_val = row[feat]
            if feat == 'credit_score':
                if val > 0:
                    reasons_for.append(f"Good credit score ({feat_val:.0f})")
                else:
                    reasons_against.append(f"Low credit score ({feat_val:.0f})")
            elif feat == 'debt_ratio':
                if val > 0:
                    reasons_for.append(f"Low debt-to-income ratio ({feat_val:.1%})")
                else:
                    reasons_against.append(f"High debt-to-income ratio ({feat_val:.1%})")
            elif feat == 'income':
                if val > 0:
                    reasons_for.append(f"Strong income (${feat_val:,.0f}/year)")
                else:
                    reasons_against.append(f"Low income (${feat_val:,.0f}/year)")
            elif feat == 'employment_years':
                if val > 0:
                    reasons_for.append(f"Stable employment ({feat_val:.0f} years)")
                else:
                    reasons_against.append(f"Short employment history ({feat_val:.0f} years)")
            elif feat == 'loan_amount':
                if val > 0:
                    reasons_for.append(f"Reasonable loan amount (${feat_val:,.0f})")
                else:
                    reasons_against.append(f"High loan amount (${feat_val:,.0f})")

        if reasons_for:
            print(f"\n    ✅ Positive factors:")
            for r in reasons_for:
                print(f"       • {r}")

        if reasons_against:
            print(f"\n    ❌ Negative factors:")
            for r in reasons_against:
                print(f"       • {r}")

        # ── Actionable Advice ──
        if prediction == 0:  # Denied
            print(f"\n    💡 To improve your chances:")
            if row['credit_score'] < thresholds['credit_score']['good']:
                diff = thresholds['credit_score']['good'] - row['credit_score']
                print(f"       → Increase credit score by {diff:.0f} points (target: {thresholds['credit_score']['good']}+)")
            if row['debt_ratio'] > thresholds['debt_ratio']['good']:
                print(f"       → Reduce debt-to-income ratio below {thresholds['debt_ratio']['good']:.0%} (currently {row['debt_ratio']:.1%})")
            if row['loan_amount'] > thresholds['loan_amount']['fair']:
                suggested = thresholds['loan_amount']['fair']
                print(f"       → Consider a smaller loan (${suggested:,.0f} or less)")
            if row['employment_years'] < thresholds['employment_years']['good']:
                diff = thresholds['employment_years']['good'] - row['employment_years']
                print(f"       → {diff:.0f} more years of employment history would help")

        print(f"\n    📋 Prediction: {result} | Actual: {'APPROVED' if actual == 1 else 'DENIED'}")

    print("\n" + "=" * 60)
    print("  ✨ THAT'S THE POWER OF XAI!")
    print("  ✨ Same model, same data — but now every decision")
    print("  ✨ comes with reasoning, evidence, and actionable advice.")
    print("=" * 60)


# ──────────────────────────────────────────────
# 7. Summary Comparison
# ──────────────────────────────────────────────

def print_comparison_summary():
    """Print a side-by-side comparison of the two approaches."""

    print("\n" + "=" * 60)
    print("  📊 COMPARISON: WITHOUT XAI vs WITH XAI")
    print("=" * 60)
    print("""
    ┌──────────────────────┬──────────────────────────────┐
    │   WITHOUT XAI        │   WITH XAI                   │
    ├──────────────────────┼──────────────────────────────┤
    │ "DENIED"             │ "DENIED because:             │
    │                      │  • Low credit score (520)    │
    │                      │  • High debt ratio (72%)"    │
    ├──────────────────────┼──────────────────────────────┤
    │ No debugging info    │ Feature importance charts     │
    │                      │ SHAP waterfall plots          │
    │                      │ LIME local explanations       │
    ├──────────────────────┼──────────────────────────────┤
    │ "Why was I denied?"  │ "To improve: increase credit │
    │ "I don't know."      │  score to 650, reduce debt   │
    │                      │  ratio below 30%"            │
    ├──────────────────────┼──────────────────────────────┤
    │ Cannot detect bias   │ Can reveal if model uses     │
    │                      │ protected attributes         │
    ├──────────────────────┼──────────────────────────────┤
    │ Fails compliance     │ Meets GDPR, ECOA, EU AI Act  │
    └──────────────────────┴──────────────────────────────┘
    """)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 LOAN PREDICTION WITH XAI — EXPLAINABLE AI DEMO")
    print("=" * 60)

    # Step 1: Generate data (same as Project 1)
    print("\n🔄 Generating synthetic loan data...")
    data = generate_loan_data(n_samples=1000)
    print(f"   Generated {len(data)} loan applications")
    print(f"   Approval rate: {data['approved'].mean():.1%}")

    # Step 2: Train model (same as Project 1)
    print("\n🔄 Training Random Forest model...")
    model, X_train, X_test, y_test, accuracy, features = train_model(data)
    print(f"   Model accuracy: {accuracy:.2%}")

    # Step 3: XAI Technique #1 — Feature Importance
    print("\n🔄 Computing Feature Importance...")
    importances = explain_feature_importance(model, features)

    # Step 4: XAI Technique #2 — SHAP
    print("\n🔄 Computing SHAP explanations (this may take a moment)...")
    shap_values = explain_with_shap(model, X_train, X_test, features, n_display=5)

    # Step 5: XAI Technique #3 — LIME
    print("\n🔄 Computing LIME explanations...")
    explain_with_lime(model, X_train, X_test, features, n_display=5)

    # Step 6: Human-Readable Explanations
    print("\n🔄 Generating human-readable explanations...")
    generate_human_explanations(model, X_test, y_test, features, shap_values, n_display=5)

    # Step 7: Summary
    print_comparison_summary()

    print(f"\n📁 All charts saved to: {OUTPUT_DIR}")
    print(f"   Open the PNG files to see visual explanations!\n")
