import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

MALE_PATH = "outputs/cleaned_male.csv"
FIGS_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"


def prep_male_data():
    male = pd.read_csv(MALE_PATH)
    male = male.dropna(subset=["y_concentration", "gestational_age_weeks", "bmi", "age"])
    male = male[male["y_concentration"] >= 0]
    return male


def plot_y_vs_ga(male, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(male["gestational_age_weeks"], male["y_concentration"], alpha=0.3, s=10)
    z = np.polyfit(male["gestational_age_weeks"], male["y_concentration"], 1)
    p = np.poly1d(z)
    weeks_sorted = np.sort(male["gestational_age_weeks"])
    ax.plot(weeks_sorted, p(weeks_sorted), "r-", linewidth=2)
    ax.set_xlabel("Gestational Age (weeks)")
    ax.set_ylabel("Y-chromosome Concentration")
    ax.set_title("Y Concentration vs GA")

    ax2 = axes[1]
    ax2.scatter(male["bmi"], male["y_concentration"], alpha=0.3, s=10)
    z2 = np.polyfit(male["bmi"], male["y_concentration"], 1)
    p2 = np.poly1d(z2)
    bmi_sorted = np.sort(male["bmi"])
    ax2.plot(bmi_sorted, p2(bmi_sorted), "r-", linewidth=2)
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Y-chromosome Concentration")
    ax2.set_title("Y Concentration vs BMI")

    ax3 = axes[2]
    top = male.sort_values("y_concentration", ascending=False).head(10)
    ax3.bar(top["woman_id"].astype(str), top["y_concentration"])
    ax3.set_xlabel("Woman ID")
    ax3.set_ylabel("Y-chromosome Concentration")
    ax3.set_title("Top 10 Y Concentrations")
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGS_DIR, "y_concentration_overview.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def correlation_heatmap(male, save=True):
    corr_cols = ["gestational_age_weeks", "bmi", "age", "height", "weight",
                 "y_concentration", "raw_reads", "gc_content", "blood_draw_number"]
    existing = [c for c in corr_cols if c in male.columns]
    corr = male[existing].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — Male Foetuses")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGS_DIR, "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def linear_regression(X, y, label=""):
    X = sm.add_constant(X)
    model = sm.OLS(y, X.astype(float)).fit()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(model.summary())
    return model


def base_model(male):
    X = male[["gestational_age_weeks", "bmi", "age"]]
    y = male["y_concentration"]
    return linear_regression(X, y, label="Base Model: Y ~ GA + BMI + Age")


def interaction_model(male):
    male = male.copy()
    male["ga_bmi"] = male["gestational_age_weeks"] * male["bmi"]
    male["ga_age"] = male["gestational_age_weeks"] * male["age"]
    X = male[["gestational_age_weeks", "bmi", "age", "ga_bmi", "ga_age"]]
    y = male["y_concentration"]
    return linear_regression(X, y, label="Interaction Model")


def multi_factor_model(male):
    X = male[["gestational_age_weeks", "bmi", "age", "blood_draw_number",
              "raw_reads", "gc_content"]]
    y = male["y_concentration"]
    return linear_regression(X, y, label="Multi-factor Model")


if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    male = prep_male_data()
    print(f"Male samples for modelling: {len(male)}")

    plot_y_vs_ga(male)
    correlation_heatmap(male)

    model1 = base_model(male)
    model2 = interaction_model(male)
    model3 = multi_factor_model(male)

    results = {
        "Base Model": {"R2": model1.rsquared, "Adj R2": model1.rsquared_adj,
                       "AIC": model1.aic, "BIC": model1.bic, "N": model1.nobs},
        "Interaction Model": {"R2": model2.rsquared, "Adj R2": model2.rsquared_adj,
                              "AIC": model2.aic, "BIC": model2.bic, "N": model2.nobs},
        "Multi-factor Model": {"R2": model3.rsquared, "Adj R2": model3.rsquared_adj,
                               "AIC": model3.aic, "BIC": model3.bic, "N": model3.nobs}
    }
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    results_df.to_csv(results_path)
    print(f"\nModel comparison saved: {results_path}")
    print(results_df.round(4))