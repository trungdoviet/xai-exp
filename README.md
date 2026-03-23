# XAI Documentation & Demo Projects — Walkthrough

## What Was Built

### 1. Comprehensive Documentation
[XAI_EXPLAINED.md](file:///e:/Workspace/XAI/XAI_EXPLAINED.md) — A detailed guide covering:
- What, When, How, Why of XAI
- Pros & Cons with trade-off analysis
- Key ideas (SHAP, LIME, Feature Importance, Grad-CAM, Counterfactuals, etc.)
- Real-world use cases across healthcare, finance, autonomous vehicles, and NLP
- Comparison tables, ASCII diagrams, and technique deep dives

---

### 2. Project 1 — Black-Box Model (No XAI)
[project_without_xai/main.py](file:///e:/Workspace/XAI/project_without_xai/main.py)

Trains a Random Forest on synthetic loan data and outputs bare decisions:
```
Applicant #1:
    Decision: ❌ DENIED
    Reason:   ??? (NO EXPLANATION AVAILABLE)
```

---

### 3. Project 2 — Explainable Model (With XAI)
[project_with_xai/main.py](file:///e:/Workspace/XAI/project_with_xai/main.py)

Same model, but with full explanations using 3 XAI techniques + human-readable advice.

## Generated Charts (12 total)

### Feature Importance (Global)
![Feature importance chart showing credit_score and debt_ratio as top factors](C:/Users/Administrator/.gemini/antigravity/brain/08464469-5dae-46d1-b2de-9b2752ffcb8b/feature_importance.png)

### SHAP Summary Plot (Global)
![SHAP beeswarm plot showing feature impact distribution across all predictions](C:/Users/Administrator/.gemini/antigravity/brain/08464469-5dae-46d1-b2de-9b2752ffcb8b/shap_summary.png)

### SHAP Waterfall — Denied Applicant
![SHAP waterfall showing why Applicant 3 was denied](C:/Users/Administrator/.gemini/antigravity/brain/08464469-5dae-46d1-b2de-9b2752ffcb8b/shap_waterfall_applicant_3.png)

### LIME Explanation — Denied Applicant
![LIME explanation for denied Applicant 3](C:/Users/Administrator/.gemini/antigravity/brain/08464469-5dae-46d1-b2de-9b2752ffcb8b/lime_explanation_applicant_3.png)

## The Key Difference

| Without XAI | With XAI |
|---|---|
| "DENIED" | "DENIED because: low credit score (391), high debt ratio (89%)" |
| No debugging info | Feature importance + SHAP + LIME charts |
| "Why denied?" → "Don't know" | "To improve: +309 credit points, reduce debt below 30%" |
| Cannot detect bias | Reveals if model uses protected attributes |

## Verification Results

| Check | Result |
|---|---|
| Project 1 runs without errors | ✅ |
| Project 2 runs without errors | ✅ |
| Feature Importance chart generated | ✅ |
| SHAP Summary plot generated | ✅ |
| 5 SHAP Waterfall plots generated | ✅ |
| 5 LIME explanation plots generated | ✅ |
| Human-readable reasons printed | ✅ |
| Actionable advice printed for denied applicants | ✅ |
