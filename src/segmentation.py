import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

K_SELECTED = 4

def rfm_quintile_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Rank-based quintile scoring (rank(method='first') then qcut). Plain
    qcut on raw values fails here because 27.5% of customers tie at
    Frequency=1, which breaks quantile boundaries. Rank-based tie-breaking
    produces clean, equal-sized quintiles for R, F and M."""
    df = features.copy()

    def quintile_score(series, ascending_labels=True):
        ranked = series.rank(method='first')
        labels = [1, 2, 3, 4, 5] if ascending_labels else [5, 4, 3, 2, 1]
        return pd.qcut(ranked, 5, labels=labels).astype(int)

    df['R_score'] = quintile_score(df['Recency'], ascending_labels=False)                       
    df['F_score'] = quintile_score(df['Frequency'], ascending_labels=True)
    df['M_score'] = quintile_score(df['Monetary'], ascending_labels=True)
    df['RFM_Score'] = df['R_score'] + df['F_score'] + df['M_score']
    df['RFM_Segment'] = (df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str))
    return df

def run_kmeans_segmentation(features: pd.DataFrame, k: int = K_SELECTED):
    """Log-transform (handles the heavy Frequency/Monetary skew and the 22
    negative-Monetary edge cases via an offset shift), scale, then fit
    K-Means with the finalized K=4."""
    df = features.copy()
    monetary_offset = abs(df['Monetary'].min()) + 1
    X = np.column_stack([
        np.log1p(df['Recency']),
        np.log1p(df['Frequency']),
        np.log1p(df['Monetary'] + monetary_offset),
    ])
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df[f'Cluster_K{k}'] = km.fit_predict(X_scaled)
    return df, km
