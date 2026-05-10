# 📊 Customer Churn Analysis and Prediction

> **Data Science Internship Project — SaiKet Systems**  
> Analyzing and predicting customer churn in a telecommunications company using machine learning.

---

## 📌 Project Overview

Customer churn — when a customer stops doing business with a company — is one of the most critical challenges in the telecom industry. This project performs an end-to-end data science pipeline on a real-world telecom dataset to:

- Understand the patterns and drivers behind customer churn
- Segment customers by risk level
- Build and evaluate machine learning models to predict churn
- Identify high-value customers at risk of leaving

**Dataset:** Telco Customer Churn Dataset — 7,043 customers, 21 features  
**Target Variable:** `Churn` (Yes / No)

---

## 🗂️ Project Structure

```
customer-churn-analysis/
│
├── Task1_Data_Preparation/
│   └── task1_data_preparation.py
│
├── Task2_EDA/
│   ├── task2_eda.py
│   ├── fig1_churn_rate.png
│   ├── fig2_demographics.png
│   ├── fig3_tenure_churn.png
│   └── fig4_contract_payment.png
│
├── Task3_Segmentation/
│   ├── task3_segmentation.py
│   ├── fig5_segmentation_churn.png
│   └── fig6_high_value_atrisk.png
│
├── Task4_Churn_Prediction/
│   ├── task4_churn_prediction.py
│   ├── fig7_model_comparison.png
│   ├── fig8_confusion_matrices.png
│   └── fig9_feature_importances.png
│
├── Telco_Customer_Churn_Dataset___3_.csv
└── README.md
```

---

## ✅ Tasks Completed

### Task 1 — Data Preparation

- Loaded and explored the raw dataset (7,043 rows × 21 columns)
- Converted `TotalCharges` from string to numeric; filled 11 missing values with the median
- Removed 22 duplicate rows
- Dropped non-informative `customerID` column
- Applied **Label Encoding** for binary categorical columns
- Applied **One-Hot Encoding** for multi-class categorical columns
- Split dataset into **80% training / 20% testing** (stratified on `Churn`)
- Final processed shape: 7,021 rows × 31 columns

---

### Task 2 — Exploratory Data Analysis (EDA)

- Calculated and visualized overall churn rate
- Explored customer distribution by gender, partner status, and dependent status
- Analyzed tenure distribution and its relationship with churn
- Investigated churn variation across contract types and payment methods

**Key Findings:**
| Insight | Value |
|---|---|
| Overall churn rate | 26.5% |
| Churned customers median tenure | 10 months |
| Retained customers median tenure | 38 months |
| Month-to-month contract churn rate | 42.7% |
| Electronic check payment churn rate | 45.3% |
| Customers without dependents churn rate | 31.3% |

---

### Task 3 — Customer Segmentation

Customers were segmented across three dimensions:

**By Tenure Group:**
| Segment | Churn Rate |
|---|---|
| New (0–12 months) | 47.7% |
| Early (13–24 months) | 28.7% |
| Mid (25–48 months) | 20.4% |
| Loyal (49–72 months) | 9.5% |

**By Monthly Charges:**
| Segment | Churn Rate |
|---|---|
| Low (< $35) | 10.9% |
| Medium ($35–65) | 23.1% |
| High ($65–95) | 35.9% |
| Premium (> $95) | 32.3% |

**By Contract Type:**
| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

**High-Value At-Risk Customers:**

- Identified **697 customers** paying > $81.45/month on month-to-month contracts who churned
- Average monthly charges: **$94.02**
- Average revenue lost per customer: **$2,023**
- Most concentrated in the **New (0–12m)** tenure group

---

### Task 4 — Churn Prediction Model

Three machine learning models were trained, tuned, and evaluated:

**Model Comparison:**
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 79.6% | 64.1% | 52.4% | 57.7% |
| Decision Tree | 75.1% | 52.7% | 59.4% | 55.9% |
| **Random Forest** ✅ | **81.4%** | **69.7%** | 52.9% | **60.2%** |

**Best Model: Random Forest** (tuned with `max_depth=10`, `n_estimators=200`, `max_features='sqrt'`)

**Top Predictive Features:**

1. `TotalCharges` — cumulative spend
2. `tenure` — customer lifetime
3. `MonthlyCharges` — cost sensitivity
4. `InternetService_Fiber optic` — fiber users churn more
5. `PaymentMethod_Electronic check` — high-risk payment behavior
6. `Contract_Two year` — strong retention signal

---

## 🛠️ Tech Stack

| Tool         | Purpose                              |
| ------------ | ------------------------------------ |
| Python 3.x   | Core language                        |
| Pandas       | Data manipulation                    |
| NumPy        | Numerical operations                 |
| Matplotlib   | Data visualization                   |
| Scikit-learn | ML models, preprocessing, evaluation |

---

## ▶️ How to Run

**1. Clone the repository:**

```bash
git clone https://github.com/rayan-baderdine/customer-churn-analysis.git
cd customer-churn-analysis
```

**2. Install dependencies:**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**3. Run each task in order:**

```bash
python Task1_Data_Preparation/task1_data_preparation.py
python Task2_EDA/task2_eda.py
python Task3_Segmentation/task3_segmentation.py
python Task4_Churn_Prediction/task4_churn_prediction.py
```

> ⚠️ Make sure `Telco_Customer_Churn_Dataset___3_.csv` is in the same directory as the script you are running, or update the file path inside the script.

---

## 📈 Sample Visualizations

| Churn Rate                             | Demographics                             |
| -------------------------------------- | ---------------------------------------- |
| ![fig1](Task2_EDA/fig1_churn_rate.png) | ![fig2](Task2_EDA/fig2_demographics.png) |

| Tenure vs Churn                          | Contract & Payment                           |
| ---------------------------------------- | -------------------------------------------- |
| ![fig3](Task2_EDA/fig3_tenure_churn.png) | ![fig4](Task2_EDA/fig4_contract_payment.png) |

| Segmentation                                            | Model Comparison                                          |
| ------------------------------------------------------- | --------------------------------------------------------- |
| ![fig5](Task3_Segmentation/fig5_segmentation_churn.png) | ![fig7](Task4_Churn_Prediction/fig7_model_comparison.png) |

---

## 🏢 Internship

This project was completed as part of the **Data Science Internship Program** at **[SaiKet Systems](https://www.saiket.in)**.

---

## 👤 Author

**Rayan**  
M.S. Computer and Communication Engineering — Lebanese International University (LIU)  
[GitHub](https://github.com/rayan-baderdine) · [LinkedIn](www.linkedin.com/in/rayan-baderdine-5b32992a2)
