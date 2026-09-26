# E-Commerce Customer Analytics, Segmentation & Churn Risk Prediction

> ⚠️ Placeholder file. The full README (project overview, methodology,
> installation, findings, limitations, future scope) will be written in the
> next stage. This file exists now only so `streamlit run app.py` has a
> complete project folder to run against.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard covers: KPI Overview, Sales & Business Trends, Customer Segmentation
(RFM + K-Means, K=4), Churn Risk Prediction (leakage-safe, time-split model),
Business Insights & Actions, and a Customer Risk Explorer.

See `outputs/` for the Stage 2 cleaning audit, Stage 4 K-selection numbers,
and Stage 5 model comparison table. See `models/model_metadata.json` for the
full prediction design (observation period, cutoff, churn definition, feature
list, test-set metrics).
