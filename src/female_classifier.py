import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import xgboost as xgb

FEMALE_PATH = "outputs/cleaned_female.csv"
FIGS_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"

FEATURES = [
    "z13", "z18", "z21", "zx", "x_concentration",
    "gc_content", "gc13", "gc18", "gc21",
    "raw_reads", "alignment_ratio", "duplicate_ratio",
    "unique_reads", "filtered_read_ratio",
    "bmi", "age", "height", "weight"
]


def prep_female():
    female = pd.read_csv(FEMALE_PATH)
    female["abnormal"] = female["aneuploidy"].notna().astype(int)
    return female


def zscore_rule_classifier(female, z_threshold=3.0):
    z_cols = ["z13", "z18", "z21", "zx"]
    female["abnormal_zscore"] = (female[z_cols].abs() > z_threshold).any(axis=1).astype(int)
    return female


def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    return cm


def plot_roc_curve(y_true, y_prob_dict, save=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in y_prob_dict.items():
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Female Abnormality Classification")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIGS_DIR, "roc_curves_female.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_feature_importance(model, feature_names, title, save=True):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        fi = np.abs(model.coef_[0])
    else:
        return
    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_df["feature"], fi_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        fname = title.lower().replace(" ", "_") + ".png"
        path = os.path.join(FIGS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    female = prep_female()
    print(f"Female samples: {len(female)}")
    print(f"  Abnormal (aneuploidy flagged): {female['abnormal'].sum()}")
    print(f"  Normal: {(female['abnormal'] == 0).sum()}")

    female = zscore_rule_classifier(female, z_threshold=3.0)
    rule_accuracy = (female["abnormal_zscore"] == female["abnormal"]).mean()
    print(f"\nZ-score Rule Accuracy (±3): {rule_accuracy:.4f}")

    existing_features = [c for c in FEATURES if c in female.columns]
    X = female[existing_features].dropna()
    y = female.loc[X.index, "abnormal"]
    print(f"\nModelling samples: {len(X)}, Abnormal: {y.sum()}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression (Balanced)")

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest (Balanced)")

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=scale_pos,
                                random_state=42, eval_metric="logloss")
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")

    plot_roc_curve(y_test, {
        "Logistic Regression": y_prob_lr,
        "Random Forest": y_prob_rf,
        "XGBoost": y_prob_xgb
    })

    plot_feature_importance(rf, existing_features, "Random Forest Feature Importance — Female")
    plot_feature_importance(xgb_clf, existing_features, "XGBoost Feature Importance — Female")

    summary_df = pd.DataFrame({
        "Model": ["Z-score Rule (3σ)"] + ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [rule_accuracy] + [
            (y_pred_lr == y_test).mean(),
            (y_pred_rf == y_test).mean(),
            (y_pred_xgb == y_test).mean()
        ]
    })
    summary_path = os.path.join(TABLES_DIR, "female_classifier_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()