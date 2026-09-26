import pandas as pd
import numpy as np

FEATURE_COLS = [
    'Recency_cutoff', 'Frequency_cutoff', 'Monetary_cutoff', 'AOV_cutoff',
    'Total_Quantity_cutoff', 'Tenure_cutoff', 'Avg_Interval_cutoff',
    'Is_SinglePurchase_asof_cutoff',
]

def score_current_customers(customer_features: pd.DataFrame, rf_model, interval_median: float) -> pd.DataFrame:
    """
    Apply the FROZEN, already-trained Random Forest to current (full-dataset-
    snapshot) customer features, to get an operational/current risk score for
    every customer -- not just the Stage 5 held-out test set.

    This is the standard final step of a churn model: train + validate on a
    historical cutoff, then score present-day features with the frozen model.
    It introduces no leakage because the model's parameters are untouched;
    only fresh, current inputs are being fed through it.

    customer_features must have the current-snapshot equivalents of the
    training features (Stage 3's Recency/Frequency/Monetary/etc., computed
    relative to "now" rather than the historical cutoff).
    """
    df = customer_features.copy()
    current_X = pd.DataFrame({
        'Recency_cutoff': df['Recency'],
        'Frequency_cutoff': df['Frequency'],
        'Monetary_cutoff': df['Monetary'],
        'AOV_cutoff': df['Average_Order_Value'],
        'Total_Quantity_cutoff': df['Total_Quantity'],
        'Tenure_cutoff': df['Customer_Tenure_Days'],
        'Avg_Interval_cutoff': df['Avg_Purchase_Interval_Days'].fillna(interval_median),
        'Is_SinglePurchase_asof_cutoff': (df['Frequency'] == 1).astype(int),
    })[FEATURE_COLS]

    df['Current_Risk_Probability'] = rf_model.predict_proba(current_X)[:, 1]
    df['Current_Predicted_Churn'] = rf_model.predict(current_X)

    def risk_tier(p):
        if p < 0.40:
            return 'Low'
        elif p < 0.70:
            return 'Medium'
        return 'High'

    df['Current_Risk_Tier'] = df['Current_Risk_Probability'].apply(risk_tier)
    return df
