import pandas as pd
import numpy as np

def find_cancel_only_customers(rfm_base: pd.DataFrame) -> list:
    """Customers whose ENTIRE row history is cancellations (no real purchase).
    These are excluded from the customer feature table -- their Recency/
    Frequency are structurally undefined (see Stage 3 sensitivity test #1)."""
    grp = rfm_base.groupby('Customer ID').agg(
        n_rows=('Invoice', 'size'), n_cancel=('IsCancellation', 'sum'))
    return grp[grp['n_rows'] == grp['n_cancel']].index.tolist()

def build_customer_features(rfm_base: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Build the customer-level RFM feature table.

    Definitions (finalized after Stage 3 sensitivity testing):
      - Monetary  = NET spend: sum(Quantity*Price) across ALL rows
                    (purchases + cancellations netted together)
      - Total_Quantity = NET units retained (same netting logic)
      - Frequency = count of DISTINCT NON-CANCELLED invoices only
                    (a return is not a new purchase event)
      - Recency   = days between snapshot_date and the last NON-CANCELLED
                    purchase (snapshot_date defaults to max(InvoiceDate)+1 day,
                    i.e. "today" for the full historical dataset; churn_model.py
                    overrides this with a historical cutoff date to stay leakage-safe)
      - Avg_Purchase_Interval_Days = (last-first)/(Frequency-1), NaN when
                    Frequency==1 (genuinely undefined, never fabricated as 0)
    """
    cancel_only_ids = find_cancel_only_customers(rfm_base)
    base = rfm_base[~rfm_base['Customer ID'].isin(cancel_only_ids)].copy()

    if snapshot_date is None:
        snapshot_date = base['InvoiceDate'].max() + pd.Timedelta(days=1)

    purchases = base[~base['IsCancellation']].copy()

    agg = base.groupby('Customer ID').agg(
        Monetary=('LineTotal', 'sum'),
        Total_Quantity=('Quantity', 'sum'),
    )
    purchase_agg = purchases.groupby('Customer ID').agg(
        Frequency=('Invoice', 'nunique'),
        Last_Purchase_Date=('InvoiceDate', 'max'),
        First_Purchase_Date=('InvoiceDate', 'min'),
    )
    features = agg.join(purchase_agg, how='inner')                                      

    features['Recency'] = (snapshot_date - features['Last_Purchase_Date']).dt.days
    features['Average_Order_Value'] = features['Monetary'] / features['Frequency']
    tenure = (features['Last_Purchase_Date'] - features['First_Purchase_Date']).dt.days
    features['Customer_Tenure_Days'] = tenure
    features['Avg_Purchase_Interval_Days'] = np.where(
        features['Frequency'] > 1, tenure / (features['Frequency'] - 1), np.nan)

    return features.reset_index().rename(columns={'Customer ID': 'Customer_ID'})
