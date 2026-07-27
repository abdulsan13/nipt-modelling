import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MALE_PATH = "outputs/cleaned_male.csv"
FIGS_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"

THRESHOLD = 0.04


def prep_male():
    male = pd.read_csv(MALE_PATH)
    male = male.dropna(subset=["gestational_age_weeks", "bmi", "y_concentration"])
    male = male[male["y_concentration"] >= 0]
    male["threshold_reached"] = (male["y_concentration"] >= THRESHOLD).astype(int)
    return male


def assign_bmi_groups_fixed(df):
    bins = [0, 20, 28, 32, 36, 40, 200]
    labels = ["<20", "20-28", "28-32", "32-36", "36-40", ">=40"]
    df["bmi_group"] = pd.cut(df["bmi"], bins=bins, labels=labels, right=False)
    return df


def assign_bmi_groups_quantile(df, n_groups=5):
    df["bmi_group_q"] = pd.qcut(df["bmi"], q=n_groups, labels=[f"Q{i+1}" for i in range(n_groups)])
    return df


def compute_group_timing(male, group_col="bmi_group"):
    results = []
    for grp in sorted(male[group_col].dropna().unique()):
        subset = male[male[group_col] == grp]
        for week in sorted(subset["gestational_age_weeks"].dropna().unique()):
            week_data = subset[(subset["gestational_age_weeks"] >= week - 0.5) &
                               (subset["gestational_age_weeks"] < week + 0.5)]
            if len(week_data) < 3:
                continue
            p = week_data["threshold_reached"].mean()
            results.append({"group": grp, "week": round(week, 1),
                            "n": len(week_data), "p_threshold": p})
    return pd.DataFrame(results)


def find_optimal_week(timing_df, target_p=0.85, group_col="group"):
    optimal = []
    for grp in timing_df[group_col].unique():
        grp_data = timing_df[timing_df[group_col] == grp].sort_values("week")
        best = grp_data[grp_data["p_threshold"] >= target_p]
        if not best.empty:
            row = best.iloc[0]
            optimal.append({"group": grp, "optimal_week": row["week"],
                            "p_at_optimal": row["p_threshold"],
                            "n_at_optimal": row["n"]})
        else:
            fallback = grp_data.loc[grp_data["p_threshold"].idxmax()]
            optimal.append({"group": grp, "optimal_week": fallback["week"],
                            "p_at_optimal": fallback["p_threshold"],
                            "n_at_optimal": fallback["n"]})
    return pd.DataFrame(optimal)


def plot_group_probs(timing_df, group_col="group", title="", save=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    for grp in timing_df[group_col].unique():
        grp_data = timing_df[timing_df[group_col] == grp].sort_values("week")
        ax.plot(grp_data["week"], grp_data["p_threshold"],
                marker="o", label=f"Group {grp}")
    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.7, label="85% target")
    ax.axhline(y=0.90, color="black", linestyle="--", alpha=0.7, label="90% target")
    ax.set_xlabel("Gestational Age (weeks)")
    ax.set_ylabel("P(Y concentration >= 4%)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fname = title.lower().replace(" ", "_").replace("%", "pct")[:50] + ".png"
        path = os.path.join(FIGS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def measurement_error_analysis(male, n_sims=50):
    np.random.seed(42)
    male_base = assign_bmi_groups_fixed(male.copy())

    all_sims = []
    for sim in range(n_sims):
        male_sim = male_base.copy()
        noise = np.random.normal(0, 0.01, size=len(male_sim))
        male_sim["y_conc_sim"] = np.maximum(0, male_sim["y_concentration"] + noise)
        male_sim["threshold_sim"] = (male_sim["y_conc_sim"] >= THRESHOLD).astype(int)

        timing_sim = compute_group_timing(male_sim, group_col="bmi_group")
        opt_sim = find_optimal_week(timing_sim, target_p=0.85, group_col="group")
        opt_sim["sim"] = sim
        all_sims.append(opt_sim)

    sim_results = pd.concat(all_sims, ignore_index=True)
    summary = sim_results.groupby("group")["optimal_week"].agg(["mean", "std", "min", "max"]).round(2)
    print("\nMeasurement Error Sensitivity (50 simulations):")
    print(summary.to_string())
    summary.to_csv(os.path.join(TABLES_DIR, "measurement_error_sensitivity.csv"))
    return summary


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    male = prep_male()
    print(f"Male samples for timing: {len(male)}")

    # Fixed BMI groups
    male = assign_bmi_groups_fixed(male)
    timing = compute_group_timing(male, group_col="bmi_group")
    optimal = find_optimal_week(timing, target_p=0.85, group_col="group")
    print("\nOptimal Timing — Fixed BMI Groups (85% target):")
    print(optimal.to_string(index=False))
    optimal.to_csv(os.path.join(TABLES_DIR, "problem2_optimal_timing_fixed.csv"), index=False)
    plot_group_probs(timing, group_col="group",
                     title="Problem 2: Threshold Probability by BMI Group (Fixed)")

    # Quantile BMI groups
    male = assign_bmi_groups_quantile(male, n_groups=5)
    timing_q = compute_group_timing(male, group_col="bmi_group_q")
    optimal_q = find_optimal_week(timing_q, target_p=0.85, group_col="group")
    print("\nOptimal Timing — Quantile BMI Groups (85% target):")
    print(optimal_q.to_string(index=False))
    optimal_q.to_csv(os.path.join(TABLES_DIR, "problem2_optimal_timing_quantile.csv"), index=False)
    plot_group_probs(timing_q, group_col="group",
                     title="Problem 2: Threshold Probability by Quantile BMI Group")

    # Measurement error
    summary = measurement_error_analysis(male, n_sims=50)


if __name__ == "__main__":
    main()