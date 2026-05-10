# ============================================================
# Task 3: Customer Segmentation
# Project: Customer Churn Analysis and Prediction
# Internship: SaiKet Systems – Data Science
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Load & Clean ─────────────────────────────────────────────
df = pd.read_csv("Telco_Customer_Churn_Dataset___3_.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop_duplicates(inplace=True)

# Style
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
CHURN_COLORS = ["#00d4ff", "#ff4f81"]

print("=" * 55)
print("TASK 3: Customer Segmentation")
print("=" * 55)

# ── SEGMENT 1: Tenure Groups ──────────────────────────────────
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["New (0–12m)", "Early (13–24m)", "Mid (25–48m)", "Loyal (49–72m)"]
)

tenure_churn = df.groupby("TenureGroup", observed=True)["Churn"].apply(
    lambda x: (x == "Yes").sum() / len(x) * 100
).reset_index()
tenure_churn.columns = ["TenureGroup", "ChurnRate"]

print("\nChurn rate by Tenure Group:")
print(tenure_churn.to_string(index=False))

# ── SEGMENT 2: Monthly Charges Groups ────────────────────────
df["ChargeGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 65, 95, 120],
    labels=["Low (<$35)", "Medium ($35–65)", "High ($65–95)", "Premium (>$95)"]
)

charge_churn = df.groupby("ChargeGroup", observed=True)["Churn"].apply(
    lambda x: (x == "Yes").sum() / len(x) * 100
).reset_index()
charge_churn.columns = ["ChargeGroup", "ChurnRate"]

print("\nChurn rate by Monthly Charge Group:")
print(charge_churn.to_string(index=False))

# ── SEGMENT 3: Contract Type ──────────────────────────────────
contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").sum() / len(x) * 100
).reset_index()
contract_churn.columns = ["Contract", "ChurnRate"]

print("\nChurn rate by Contract Type:")
print(contract_churn.to_string(index=False))

# ── FIGURE 1: Segmentation Churn Rates ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Figure 5 — Churn Rate Across Customer Segments",
             fontsize=15, fontweight="bold", color="white")

seg_data = [
    (axes[0], tenure_churn,  "TenureGroup",  "Tenure Group",         "#a78bfa"),
    (axes[1], charge_churn,  "ChargeGroup",  "Monthly Charge Group", "#34d399"),
    (axes[2], contract_churn,"Contract",     "Contract Type",        "#fb923c"),
]

for ax, data, xcol, title, color in seg_data:
    bars = ax.bar(data[xcol], data["ChurnRate"], color=color,
                  edgecolor="#0f0f0f", linewidth=1.2, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.set_title(title, color="white", fontsize=12)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(data["ChurnRate"]) * 1.2)

plt.tight_layout()
plt.savefig("fig5_segmentation_churn.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("\n→ Saved: fig5_segmentation_churn.png")

# ── HIGH-VALUE AT-RISK CUSTOMERS ─────────────────────────────
# Definition: Monthly charges > 65th percentile
#             + Month-to-month contract
#             + Churned = Yes
threshold = df["MonthlyCharges"].quantile(0.65)

high_value_at_risk = df[
    (df["MonthlyCharges"] >= threshold) &
    (df["Contract"] == "Month-to-month") &
    (df["Churn"] == "Yes")
].copy()

print("\n" + "=" * 55)
print("HIGH-VALUE AT-RISK CUSTOMERS")
print("=" * 55)
print(f"Monthly charges threshold (65th percentile): ${threshold:.2f}")
print(f"Number of high-value at-risk customers     : {len(high_value_at_risk)}")
print(f"Average monthly charges                    : ${high_value_at_risk['MonthlyCharges'].mean():.2f}")
print(f"Average tenure                             : {high_value_at_risk['tenure'].mean():.1f} months")
print(f"Average total charges                      : ${high_value_at_risk['TotalCharges'].mean():.2f}")

# ── FIGURE 2: High-Value At-Risk Scatter ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 6 — High-Value At-Risk Customer Analysis",
             fontsize=15, fontweight="bold", color="white")

# Scatter: all customers colored by churn, highlight at-risk
churn_map = {"No": 0, "Yes": 1}
colors_all = df["Churn"].map({"No": "#00d4ff", "Yes": "#ff4f81"})

axes[0].scatter(df["tenure"], df["MonthlyCharges"],
                c=colors_all, alpha=0.3, s=10)
axes[0].scatter(high_value_at_risk["tenure"],
                high_value_at_risk["MonthlyCharges"],
                c="#ffdd57", alpha=0.9, s=30, label="High-Value At-Risk", zorder=5)
axes[0].axhline(threshold, color="#ffdd57", linestyle="--", linewidth=1, alpha=0.6)
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Monthly Charges ($)")
axes[0].set_title("Tenure vs Monthly Charges", color="white")

patch_no    = mpatches.Patch(color="#00d4ff", label="Retained")
patch_yes   = mpatches.Patch(color="#ff4f81", label="Churned")
patch_hv    = mpatches.Patch(color="#ffdd57", label="High-Value At-Risk")
axes[0].legend(handles=[patch_no, patch_yes, patch_hv], framealpha=0.3, fontsize=9)
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)

# Bar: count of high-value at-risk by tenure group
hv_tenure = high_value_at_risk["TenureGroup"].value_counts().sort_index()
axes[1].bar(hv_tenure.index, hv_tenure.values,
            color="#ffdd57", edgecolor="#0f0f0f", linewidth=1.2, width=0.5)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 str(int(bar.get_height())),
                 ha="center", fontsize=10, color="white", fontweight="bold")
axes[1].set_title("High-Value At-Risk by Tenure Group", color="white")
axes[1].set_xlabel("Tenure Group")
axes[1].set_ylabel("Number of Customers")
axes[1].tick_params(axis="x", rotation=15)
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig6_high_value_atrisk.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig6_high_value_atrisk.png")

# ── Summary Table ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SEGMENT SUMMARY TABLE")
print("=" * 55)
summary = df.groupby(["TenureGroup", "Contract"], observed=True).agg(
    Total=("Churn", "count"),
    Churned=("Churn", lambda x: (x == "Yes").sum()),
).reset_index()
summary["ChurnRate%"] = (summary["Churned"] / summary["Total"] * 100).round(1)
print(summary.to_string(index=False))

print("\n✔ Task 3 Complete.")
