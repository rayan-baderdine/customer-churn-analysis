# ============================================================
# Task 2: Exploratory Data Analysis (EDA)
# Project: Customer Churn Analysis and Prediction
# Internship: SaiKet Systems – Data Science
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Load & Quick Clean (same as Task 1) ──────────────────────
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
CHURN_COLORS = ["#00d4ff", "#ff4f81"]   # blue = No churn, pink = Churn

print("=" * 55)
print("TASK 2: Exploratory Data Analysis")
print("=" * 55)

# ── FIGURE 1: Churn Rate Overview ────────────────────────────
churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100

print(f"\nOverall Churn Rate : {churn_pct['Yes']:.2f}%")
print(f"Retained Customers : {churn_pct['No']:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 1 — Overall Churn Rate", fontsize=15, fontweight="bold", color="white", y=1.01)

# Pie chart
axes[0].pie(
    churn_counts, labels=["No Churn", "Churn"],
    colors=CHURN_COLORS, autopct="%1.1f%%",
    startangle=90, textprops={"color": "white", "fontsize": 12},
    wedgeprops={"edgecolor": "#0f0f0f", "linewidth": 2}
)
axes[0].set_title("Churn Distribution (Pie)", color="white")

# Bar chart
bars = axes[1].bar(["No Churn", "Churn"], churn_counts.values,
                   color=CHURN_COLORS, edgecolor="#0f0f0f", linewidth=1.5, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 str(val), ha="center", fontsize=12, color="white", fontweight="bold")
axes[1].set_title("Churn Count (Bar)", color="white")
axes[1].set_ylabel("Number of Customers")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig1_churn_rate.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig1_churn_rate.png")

# ── FIGURE 2: Demographics (Gender, Partner, Dependents) ─────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 2 — Customer Demographics vs Churn", fontsize=15,
             fontweight="bold", color="white")

demo_cols = ["gender", "Partner", "Dependents"]
demo_labels = ["Gender", "Has Partner", "Has Dependents"]

for ax, col, label in zip(axes, demo_cols, demo_labels):
    ct = pd.crosstab(df[col], df["Churn"])
    ct.plot(kind="bar", ax=ax, color=CHURN_COLORS, edgecolor="#0f0f0f",
            linewidth=1.2, rot=0, legend=False)
    ax.set_title(label, color="white", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=8, color="white")

axes[0].legend(["No Churn", "Churn"], loc="upper right", framealpha=0.3)
plt.tight_layout()
plt.savefig("fig2_demographics.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig2_demographics.png")

# Print stats
for col, label in zip(demo_cols, demo_labels):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    print(f"\n{label} churn rates (%):\n{ct['Yes'].round(1)}")

# ── FIGURE 3: Tenure Distribution vs Churn ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 3 — Tenure vs Churn", fontsize=15,
             fontweight="bold", color="white")

# Histogram: tenure by churn group
for label, color in zip(["No", "Yes"], CHURN_COLORS):
    subset = df[df["Churn"] == label]["tenure"]
    axes[0].hist(subset, bins=30, alpha=0.7, color=color,
                 label=f"{'No Churn' if label=='No' else 'Churned'}",
                 edgecolor="#0f0f0f")
axes[0].set_title("Tenure Distribution by Churn", color="white")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Count")
axes[0].legend(framealpha=0.3)
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)

# Box plot
churn_no  = df[df["Churn"] == "No"]["tenure"]
churn_yes = df[df["Churn"] == "Yes"]["tenure"]
bp = axes[1].boxplot([churn_no, churn_yes], labels=["No Churn", "Churned"],
                     patch_artist=True, notch=False,
                     medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], CHURN_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Tenure Boxplot by Churn", color="white")
axes[1].set_ylabel("Tenure (months)")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)

# Stats
print(f"\nMedian tenure — No Churn : {churn_no.median():.0f} months")
print(f"Median tenure — Churned  : {churn_yes.median():.0f} months")

plt.tight_layout()
plt.savefig("fig3_tenure_churn.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig3_tenure_churn.png")

# ── FIGURE 4: Contract Type & Payment Method vs Churn ────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 4 — Contract Type & Payment Method vs Churn",
             fontsize=15, fontweight="bold", color="white")

# Contract type
ct_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
ct_contract[["No", "Yes"]].plot(kind="bar", ax=axes[0], color=CHURN_COLORS,
                                edgecolor="#0f0f0f", rot=15, legend=False)
axes[0].set_title("Churn Rate by Contract Type", color="white")
axes[0].set_xlabel("")
axes[0].set_ylabel("Percentage (%)")
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)
axes[0].legend(["No Churn", "Churn"], framealpha=0.3)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.5, color="white")

# Payment method
ct_pay = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
ct_pay[["No", "Yes"]].plot(kind="bar", ax=axes[1], color=CHURN_COLORS,
                           edgecolor="#0f0f0f", rot=20, legend=False)
axes[1].set_title("Churn Rate by Payment Method", color="white")
axes[1].set_xlabel("")
axes[1].set_ylabel("Percentage (%)")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)
axes[1].legend(["No Churn", "Churn"], framealpha=0.3)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.5, color="white")

plt.tight_layout()
plt.savefig("fig4_contract_payment.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print("→ Saved: fig4_contract_payment.png")

# Stats
print(f"\nChurn rate by contract type (%):\n{ct_contract['Yes'].round(1)}")
print(f"\nChurn rate by payment method (%):\n{ct_pay['Yes'].round(1)}")

print("\n✔ Task 2 Complete. All 4 figures saved.")
