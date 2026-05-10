# ============================================================
# Task 1: Data Preparation
# Project: Customer Churn Analysis and Prediction
# Internship: SaiKet Systems – Data Science
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("Telco_Customer_Churn_Dataset___3_.csv")

print("=" * 55)
print("STEP 1: Initial Data Exploration")
print("=" * 55)
print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── 2. Handle Missing Values ─────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Handling Missing Values")
print("=" * 55)

# TotalCharges is stored as object (string) — convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(f"Missing values before handling:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Fill missing TotalCharges with median (robust to outliers)
median_tc = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

print(f"\nMissing values after handling:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print("→ No remaining missing values." if df.isnull().sum().sum() == 0 else "⚠ Still has missing values.")

# ── 3. Drop Irrelevant Columns ────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Data Preprocessing")
print("=" * 55)

# customerID is just an identifier — not useful for modeling
df.drop(columns=["customerID"], inplace=True)
print("Dropped 'customerID' column (non-informative identifier).")

# Remove duplicates if any
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Duplicate rows removed: {dupes}")

# ── 4. Categorical Variable Encoding ─────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Categorical Variable Encoding")
print("=" * 55)

# Identify categorical columns
cat_cols = df.select_dtypes(include="str").columns.tolist()
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Binary columns → Label Encoding (0/1)
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"  Label encoded '{col}'")

# Multi-class columns → One-Hot Encoding
multi_cols = [col for col in cat_cols if df[col].nunique() > 2]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print(f"\n  One-hot encoded columns: {multi_cols}")
print(f"\nDataset shape after encoding: {df.shape}")

# ── 5. Feature / Target Split ────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Dataset Splitting")
print("=" * 55)

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Features (X): {X.shape[1]} columns")
print(f"Target  (y): 'Churn'  |  Class distribution:\n{y.value_counts()}")

# 80% train / 20% test — stratified to preserve churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set : {X_train.shape[0]} samples")
print(f"Testing  set : {X_test.shape[0]} samples")

# ── 6. Save Processed Data ───────────────────────────────────
df.to_csv("processed_churn_data.csv", index=False)
print("\n✔ Processed dataset saved → 'processed_churn_data.csv'")
print("✔ Task 1 Complete.")
