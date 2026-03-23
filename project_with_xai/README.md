# Project WITH XAI — Explainable Loan Prediction

This project demonstrates the **same** loan prediction model as Project 1,
but now with **full XAI explanations** using SHAP, LIME, and Feature Importance.

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## What You'll See

1. **Feature Importance** — Which features matter most overall
2. **SHAP Explanations** — Per-feature contribution to each prediction
3. **LIME Explanations** — Local interpretable explanations per applicant
4. **Human-Readable Reasons** — "Denied because: high debt ratio, low credit score"
5. **Actionable Advice** — "To improve: increase credit score above 650"

## Generated Charts

After running, check these output files:

- `feature_importance.png` — Global feature importance bar chart
- `shap_summary.png` — SHAP beeswarm/summary plot
- `shap_waterfall_applicant_*.png` — SHAP waterfall plots per applicant
- `lime_explanation_applicant_*.png` — LIME explanation per applicant

## Compare with Project 1

Run `project_without_xai/main.py` first, then this project — the difference is dramatic!
