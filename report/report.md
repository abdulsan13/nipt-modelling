# CMS Mathematical Modelling Competition 2026 — NIPT Data Analysis

## Overview

Non-invasive Prenatal Testing looks at cell-free foetal DNA in maternal blood to screen for chromosomal abnormalities. The dataset has 1,687 NIPT samples (1,082 male foetuses, 605 female). We had to do four things:

1. Model Y-chromosome concentration in male foetuses
2. Find the best NIPT timing based on maternal BMI
3. Build models that predict when Y-concentration hits the threshold
4. Classify abnormalities in female foetuses

---

## Data Cleaning

The raw data came as two Excel sheets (male and female), 31 columns covering demographics, sequencing metrics, chromosome Z-scores, and foetal health.

We parsed gestational age from `"11w+6"` format to decimal weeks (e.g. 11.86), forced all numeric columns into proper dtypes, parsed dates, and flagged dodgy samples based on GC content (outside 37-43%), raw reads (under 3M), alignment ratio (under 70%), and duplicate ratio (over 10%).

**Result:** 1,687 clean samples (1,082 male, 605 female). 38 flagged for quality issues.

---

## Problem 1: Y-Chromosome Concentration Model

We wanted to know how Y-concentration changes with gestational age, BMI, and maternal age. Used Ordinary Least Squares regression.

| Model | R² | Adj R² | AIC |
|---|---|---|---|
| Base (GA + BMI + Age) | 0.052 | 0.049 | -3,810 |
| + Interaction terms | 0.055 | 0.050 | -3,809 |
| + Sequencing covariates | 0.202 | 0.197 | -3,968 |

Gestational age pushes Y-concentration up (p < 0.001). More weeks pregnant = more foetal DNA. BMI pulls it down (p < 0.001) — dilution. Age has a small negative effect too.

The interaction terms (GA×BMI, GA×Age) weren't significant. GA's effect doesn't really change across BMI levels.

The interesting bit: adding blood draw number, raw reads, and GC content boosted the R² from 5% to 20%. Blood draw number was the strongest predictor, probably because it proxies for repeated testing — women who get tested more times tend to have higher observed concentrations.

![Y Concentration Overview](outputs/figures/y_concentration_overview.png)
![Correlation Heatmap](outputs/figures/correlation_heatmap.png)

---

## Problem 2: BMI Grouping and Optimal NIPT Timing

We grouped women by BMI and looked for the earliest week where testing is reliable for each group. Created a binary flag (Y-concentration >= 4%) and tried two grouping strategies: fixed clinical bins and quantile-based groups. "Optimal" = earliest week where at least 85% of the group hit the threshold.

| BMI Group | Optimal Week | % at Threshold |
|---|---|---|
| 20-28 | 12.4 weeks | 100% |
| 28-32 | 11.0 weeks | 92% |
| 32-36 | 11.3 weeks | 87% |
| 36-40 | 23.4 weeks | 91% |
| >=40 | 23.1 weeks | 100% |

BMI under 36? Test around week 11-12. BMI over 36? Wait until roughly week 23 — almost double the time. Makes sense: fat tissue dilutes foetal DNA so it takes longer to build up detectable levels.

We ran a Monte Carlo simulation (50 iterations) adding Gaussian noise to Y-concentration values. The optimal weeks didn't budge — zero variance. Solid enough.

![Threshold Probability — Fixed Groups](outputs/figures/problem_2:_threshold_probability_by_bmi_group_(fix.png)
![Threshold Probability — Quantile Groups](outputs/figures/problem_2:_threshold_probability_by_quantile_bmi_g.png)

---

## Problem 3: Multi-Factor Timing Model

Instead of just grouping by BMI, we trained two classifiers on 9 features to predict whether the threshold is reached.

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.77 |
| Random Forest | 0.96 |

LogReg's biggest coefficients: blood draw number (+1.77), gestational age (-1.39, flipped sign because it's correlated with blood draws), weight (-0.72). RF's feature importances: weight (17.5%), BMI (16.6%), raw reads (11.7%), alignment ratio (11%).

They tell different stories. LogReg says blood draw number dominates. RF says body composition (weight + BMI = 34%) matters most. I think the truth is somewhere in between — the number of draws matters, but so does body size.

![Feature Importance](outputs/figures/problem3_feature_importance.png)

---

## Problem 4: Female Abnormality Classification

No Y chromosomes here, so Y-concentration is off the table. Compared four approaches using Z-scores, X-concentration, sequencing metrics, and BMI.

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Z-score Rule (±3) | 82.1% | — | Simple, competitive |
| Logistic Regression | 79.1% | 0.84 | Best precision/recall balance |
| Random Forest | **90.1%** | 0.76 | Highest accuracy, poor abnormal recall |
| XGBoost | 89.0% | 0.75 | Same story as RF |

Only 67 out of 605 samples (11%) are abnormal. The ML models nail the normal cases but struggle with the abnormal ones. The Z-score rule — which needs zero training — holds its own against Logistic Regression.

![ROC Curves](outputs/figures/roc_curves_female.png)
![RF Feature Importance](outputs/figures/random_forest_feature_importance_—_female.png)
![XGBoost Feature Importance](outputs/figures/xgboost_feature_importance_—_female.png)

---

## Summary

- **Y-concentration model:** GA helps, BMI hurts, age hurts a bit. Sequencing data pushes explanatory power to 20%.
- **BMI timing:** High BMI (>=36) means testing at week 23 instead of week 11.
- **Multi-factor model:** Body size and number of blood draws are the main drivers. LogReg and RF disagree on which matters more.
- **Female classifier:** RF hits 90% accuracy but the Z-score rule (82%) holds up without training.

**Repository:** [github.com/abdulsan13/nipt-modelling](https://github.com/abdulsan13/nipt-modelling)