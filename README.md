# NIPT Modelling — CMS Mathematical Modelling Competition 2026

Modelling NIPT data to work out optimal test timing based on BMI and classify foetal abnormalities.

## Project structure

```
├── Dataset.xlsx              # Raw data
├── src/
│   ├── data_cleaning.py      # Load + clean
│   ├── male_models.py        # Problem 1: Y-concentration
│   ├── grouping.py           # Problems 2 & 3: BMI timing + multi-factor
│   └── female_classifier.py  # Problem 4: abnormality classification
├── outputs/
│   ├── figures/              # Plots
│   └── tables/               # CSVs
├── report/report.md          # Write-up
└── requirements.txt
```

## What's in here

1. **Y-concentration model** — Linear regression of Y-concentration vs GA, BMI, age
2. **BMI timing** — Group by BMI, find the best week to test each group
3. **Multi-factor timing** — LogReg + Random Forest for threshold prediction
4. **Female classifier** — Z-score rule, LR, RF, XGBoost for abnormality detection

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/data_cleaning.py
python src/male_models.py
python src/grouping.py
python src/female_classifier.py
```

## tl;dr

| Problem | Main finding |
|---|---|
| Y-concentration | GA (+), BMI (−), Age (−). Sequencing data pushes R² to 20%. |
| BMI timing | BMI >= 36 means testing at week 23 instead of week 11. |
| Multi-factor | Body size and number of blood draws matter most. |
| Female classifier | RF gets 90% accuracy, but the Z-score rule (82%) keeps up without any training. |