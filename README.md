# 🛡️ SentinelPay — Online Payment Fraud Detection Management

An interactive **Data Science Programming** dashboard built with **Python, Streamlit, Pandas, Plotly, and Scikit-Learn** for real-time online payment fraud analytics, data quality auditing, machine learning model evaluation, and live transaction risk scoring.

---

## ✨ Key Modules Included

1. **🏠 Home & Project Overview**
   - Executive KPI summary (Gross Payment Volume, Fraud Rate, Dollar Volume at Risk, Champion ML Model ROC-AUC).
   - 30-day transaction volume vs. daily fraud incidence dual-axis trend.
   - Data Science methodology & leak-free ML architecture summary.

2. **📊 Interactive Fraud Analytics Dashboard**
   - Multi-dimensional filters by **Transaction Rail**, **Payment Channel**, **Risk Tier**, and **Amount Range**.
   - **Fraud Incidence by Payment Rail** (highlighting `TRANSFER` and `CASH_OUT` vulnerability).
   - **24-Hour Temporal Cycle** (highlighting the `00:00–05:00` automated account-takeover attack window).
   - **Sender Opening Balance vs. Transaction Amount Scatter** with the `100% Balance Drain Line (y = x)`.
   - **Device Risk Telemetry Violin & Box Distributions**.
   - **Merchant Category × Payment Channel Fraud Heatmap** and **Volume Hierarchy Sunburst**.

3. **🔍 Dataset Explorer & Automated Quality Audit**
   - Automated data quality audit (missing/NULL values, non-positive amounts, class imbalance ratio).
   - Interactive transaction search, filter, sort, and one-click CSV export.
   - Complete feature dictionary, Pearson correlation matrix heatmap, and descriptive statistics.
   - Built-in **Synthetic Dataset Regenerator** and **Custom CSV Uploader**.

4. **🤖 ML Model Performance, Comparison & Threshold Tuning**
   - Follows strict ML best practices: stratified 75%/25% Train/Test split **prior** to fitting `StandardScaler` and `OneHotEncoder` inside a `ColumnTransformer` pipeline.
   - Side-by-side comparison of **Random Forest**, **Gradient Boosting**, and **Logistic Regression (Baseline)**.
   - **Interactive Probability Threshold Tuner** (`0.10` to `0.90`) with live updates to Recall, Precision, F1-Score, Net Fraud Savings, and Confusion Matrix.
   - Multi-model **ROC Curve**, **Precision-Recall Curve**, and **Top 12 Feature Importances**.

5. **🛡️ Live Fraud Risk Simulator (Real-Time Scorer)**
   - One-click scenario presets (*Account Takeover Full Drain*, *High-Velocity Crypto Checkout*, *Everyday Legitimate Retail Purchase*) plus custom interactive sliders.
   - Real-time probability gauge, automated policy action (`AUTO-APPROVE`, `STEP-UP 2FA`, `MANUAL REVIEW`, `BLOCK & FREEZE`), multi-model consensus, and explainable risk factor attribution.

---

## 🚀 How to Run Locally

```bash
cd C:\Users\seenu\.gemini\antigravity\scratch\online-payment-fraud-detection
streamlit run app.py
```
