# ============================================================
# Task 4: Churn Prediction Model
# Project: Customer Churn Analysis and Prediction
# Internship: SaiKet Systems – Data Science
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

# ── 1. Load & Preprocess ──────────────────────────────────────
df = pd.read_csv("Telco_Customer_Churn_Dataset___3_.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop_duplicates(inplace=True)
df.drop(columns=["customerID"], inplace=True)

# Encode target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Encode binary categoricals
binary_cols = [c for c in df.select_dtypes(include="object").columns
               if df[c].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-class categoricals
multi_cols = [c for c in df.select_dtypes(include="object").columns]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# ── 2. Feature Selection ──────────────────────────────────────
print("=" * 55)
print("STEP 1: Feature Selection")
print("=" * 55)

# Use a quick Random Forest to rank features
X_all = df.drop(columns=["Churn"])
y_all = df["Churn"]

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_all, y_all)

importances = pd.Series(rf_selector.feature_importances_, index=X_all.columns)
top_features = importances.nlargest(15).index.tolist()

print(f"Top 15 features selected:\n{top_features}")

X = X_all[top_features]
y = y_all

# ── 3. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for Logistic Regression
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 4. Train Models ───────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Model Training")
print("=" * 55)

# ── 4a. Logistic Regression ───────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
print("  Done.")

# ── 4b. Decision Tree + Hyperparameter Tuning ─────────────────
print("\nTraining Decision Tree (with GridSearchCV)...")
dt_params = {
    "max_depth":        [3, 5, 7, 10],
    "min_samples_split":[2, 5, 10],
    "criterion":        ["gini", "entropy"]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       dt_params, cv=5, scoring="f1", n_jobs=-1)
dt_grid.fit(X_train, y_train)
best_dt   = dt_grid.best_estimator_
y_pred_dt = best_dt.predict(X_test)
print(f"  Best params: {dt_grid.best_params_}")

# ── 4c. Random Forest + Hyperparameter Tuning ─────────────────
print("\nTraining Random Forest (with GridSearchCV)...")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 10, None],
    "max_features": ["sqrt", "log2"]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf   = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print(f"  Best params: {rf_grid.best_params_}")

# ── 5. Evaluate All Models ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Model Evaluation")
print("=" * 55)

def get_metrics(y_true, y_pred, name):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
        "Precision": round(precision_score(y_true, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred)    * 100, 2),
        "F1-Score":  round(f1_score(y_true, y_pred)        * 100, 2),
    }

results = pd.DataFrame([
    get_metrics(y_test, y_pred_lr, "Logistic Regression"),
    get_metrics(y_test, y_pred_dt, "Decision Tree"),
    get_metrics(y_test, y_pred_rf, "Random Forest"),
])

print("\nModel Comparison (%):")
print(results.to_string(index=False))

# Classification reports
for name, preds in [("Logistic Regression", y_pred_lr),
                    ("Decision Tree",       y_pred_dt),
                    ("Random Forest",       y_pred_rf)]:
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds,
                                target_names=["No Churn", "Churn"]))

# ── FIGURE 1: Metrics Comparison Bar Chart ────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Figure 7 — Model Performance Comparison",
             fontsize=15, fontweight="bold", color="white")

models   = results["Model"].tolist()
metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors   = ["#00d4ff", "#a78bfa", "#34d399", "#fb923c"]
x        = np.arange(len(models))
width    = 0.18

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i * width, results[metric], width,
                  label=metric, color=color,
                  edgecolor="#0f0f0f", linewidth=1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5,
                color="white", fontweight="bold")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.legend(framealpha=0.3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig7_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("\n→ Saved: fig7_model_comparison.png")

# ── FIGURE 2: Confusion Matrices ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figure 8 — Confusion Matrices",
             fontsize=15, fontweight="bold", color="white")

model_preds = [
    ("Logistic Regression", y_pred_lr),
    ("Decision Tree",       y_pred_dt),
    ("Random Forest",       y_pred_rf),
]

for ax, (name, preds) in zip(axes, model_preds):
    cm = confusion_matrix(y_test, preds)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name, color="white", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "#333")

plt.tight_layout()
plt.savefig("fig8_confusion_matrices.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig8_confusion_matrices.png")

# ── FIGURE 3: Feature Importances (Best Model) ───────────────
best_model_name = results.loc[results["F1-Score"].idxmax(), "Model"]
print(f"\nBest model by F1-Score: {best_model_name}")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(f"Figure 9 — Feature Importances ({best_model_name})",
             fontsize=14, fontweight="bold", color="white")

feat_imp = pd.Series(best_rf.feature_importances_, index=top_features).sort_values()
colors_fi = plt.cm.plasma(np.linspace(0.3, 0.9, len(feat_imp)))
feat_imp.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="#0f0f0f")
ax.set_xlabel("Importance Score")
ax.set_ylabel("")
ax.xaxis.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig9_feature_importances.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig9_feature_importances.png")

print("\n✔ Task 4 Complete.")
