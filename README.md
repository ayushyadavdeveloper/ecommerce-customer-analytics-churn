# E-Commerce Customer Analytics, Segmentation & Churn Risk Prediction

## 📌 Project Overview

This project analyzes e-commerce customer transaction data to identify customer behavior, purchasing patterns, customer segments, and churn risk.

The project follows a complete data analytics workflow:

**Data → Cleaning → EDA → KPIs → Customer Segmentation → Churn Prediction → Insights → Business Actions**

An interactive Streamlit dashboard is provided to explore the analysis and customer-level risk information.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze overall e-commerce sales performance.
- Identify important sales and customer KPIs.
- Understand monthly sales trends.
- Analyze country-wise revenue.
- Identify top-performing products.
- Segment customers using RFM analysis and K-Means clustering.
- Build a leakage-safe churn prediction model.
- Identify customers with different levels of churn risk.
- Convert analytical findings into actionable business recommendations.

---

## 📊 Dataset

### UCI Online Retail II

The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

Dataset Source:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset contains transactions from a UK-based online retailer covering approximately two years.

### Main Columns

- `Invoice`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price`
- `Customer ID`
- `Country`

The original dataset contains more than 1 million transaction records.

---

## 🧹 Data Cleaning

The raw transaction data was cleaned before analysis.

Major cleaning steps included:

- Removed duplicate business transaction records.
- Identified and flagged cancellation transactions.
- Removed non-cancellation negative quantity adjustments.
- Removed negative-price bad-debt records.
- Handled zero-price transactions according to their business context.
- Kept missing Customer IDs for overall sales analysis.
- Excluded missing Customer IDs from customer-level RFM analysis.
- Removed internal/non-product adjustment codes from product analysis.
- Standardized Stock Codes and country values.
- Removed test transactions.
- Flagged extreme quantity values instead of blindly deleting them.

Three analytical views were created:

- `SALES_VIEW` — overall sales and business analysis.
- `RFM_BASE` — customer-level analysis.
- `PRODUCT_VIEW` — product-level analysis.

---

## 📈 Customer Analytics

The final customer analysis contains **5,870 usable customers**.

Customer-level features include:

- Recency
- Frequency
- Monetary value
- Average Order Value
- Total Quantity
- Customer Tenure
- Average Purchase Interval
- First Purchase Date
- Last Purchase Date
- Cancellation information

---

## 👥 RFM Segmentation

RFM analysis was performed using:

- **Recency** — how recently a customer purchased.
- **Frequency** — number of distinct purchase invoices.
- **Monetary** — customer's net spending.

Rank-based quintile scoring was used to handle the high number of tied frequency values.

The customers were grouped into RFM score tiers to understand differences in customer value and activity.

### Key observation

The highest RFM tier (13–15) contains approximately **22.1% of customers but contributes about 73.9% of revenue**.

This indicates a highly concentrated revenue contribution among the most valuable customers.

---

## 🔬 K-Means Customer Segmentation

K-Means clustering was applied using transformed and standardized:

- Recency
- Frequency
- Monetary

Different values of K were evaluated.

Four clusters were selected for business interpretability.

### Customer Clusters

| Cluster | Customers | Description |
|---|---:|---|
| Cluster 0 | 1,143 | Active, high-frequency, high-value repeat customers |
| Cluster 1 | 2,753 | Long-inactive / one-and-done customers |
| Cluster 2 | 1,873 | Recent but occasional/developing customers |
| Cluster 3 | 101 | Very high-value bulk/wholesale-like customers |

Cluster 3 represents a very small customer group with disproportionately high monetary value.

---

## 🤖 Churn Risk Prediction

A leakage-safe supervised learning approach was used for churn prediction.

### Observation Period

December 1, 2009 → June 12, 2011

### Prediction Cutoff

June 12, 2011

### Future Churn Window

June 12, 2011 → December 9, 2011

A customer was considered churned when there were no real non-cancelled purchase invoices during the future window.

The final eligible population contained **4,967 customers**.

### Features Used

- Recency at cutoff
- Frequency at cutoff
- Monetary value at cutoff
- Average Order Value
- Total Quantity
- Customer Tenure
- Average Purchase Interval
- Single-purchase indicator

The final current customer snapshot and clustering outputs were not used as model features, helping avoid future-information leakage.

---

## 🧠 Model Comparison

Two classification models were evaluated.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.93% | 69.32% | 74.69% | 71.90% | 79.07% |
| Random Forest | 72.74% | 68.65% | 79.71% | 73.77% | 80.17% |

The test set contained **994 customers**.

Random Forest was selected for the operational risk-scoring workflow based on the evaluation metrics and its use in the dashboard.

### Important Note

These model metrics represent the historical test-set evaluation.

The dashboard also applies the frozen trained model to the current customer snapshot as of **December 9, 2011** for operational risk scoring. These current risk scores should not be confused with the historical test-set evaluation metrics.

---

## ⚠️ Churn Risk Tiers

Customers are grouped into three risk levels based on predicted churn probability:

- **Low Risk:** probability < 0.40
- **Medium Risk:** probability 0.40–0.70
- **High Risk:** probability ≥ 0.70

Recency was the strongest predictive signal in the model analysis.

The model should be interpreted as identifying statistical risk patterns rather than proving that a particular factor causes churn.

---

## 💡 Business Insights & Actions

The analysis supports several business actions:

### 1. Reactivation

Long-inactive customers can be targeted with:

- Personalized offers
- Re-engagement campaigns
- Product recommendations
- Time-limited incentives

### 2. Early Intervention

Single-purchase customers can be targeted soon after their first transaction to encourage a second purchase.

### 3. High-Value Customer Management

The small high-value customer segment can receive closer account-level attention because of its substantial contribution to revenue.

### 4. Customer Risk Monitoring

The churn risk model can help prioritize customers for retention campaigns based on predicted risk.

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard with the following pages:

1. **KPI Overview**
2. **Sales & Business Trends**
3. **Customer Segmentation**
4. **Churn Risk Prediction**
5. **Business Insights & Actions**
6. **Customer Risk Explorer**

---

## 🖥️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ayushyadavdeveloper/ecommerce-customer-analytics-churn.git
cd ecommerce-customer-analytics-churn