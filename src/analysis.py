import pandas as pd

def compute_kpis(customer_master: pd.DataFrame, monthly_trends: pd.DataFrame) -> dict:
    total_revenue = monthly_trends['Revenue'].sum()
    total_orders = monthly_trends['Orders'].sum()
    total_customers = len(customer_master)
    avg_order_value = total_revenue / total_orders if total_orders else 0
    at_risk_count = int((customer_master['Current_Risk_Tier'] == 'High').sum())
    repeat_customers = int((customer_master['Frequency'] > 1).sum())
    repeat_rate = repeat_customers / total_customers * 100 if total_customers else 0
    return {
        'Total Revenue': total_revenue,
        'Total Customers': total_customers,
        'Total Orders': int(total_orders),
        'Average Order Value': avg_order_value,
        'High-Risk Customers': at_risk_count,
        'Repeat Customer Rate (%)': repeat_rate,
    }

                                                                                    
BUSINESS_INSIGHT_BLOCKS = [
    {
        "group": "Long-inactive customers (Cluster 1)",
        "observation": "46.9% of customers have an average Recency of 380 days and the lowest average Frequency (median 2 orders); this cluster also carries the highest average predicted churn risk (0.640).",
        "insight": "Nearly half the customer base has gone quiet for over a year, and the churn model independently confirms this group as highest-risk -- two separate analyses (unsupervised clustering, supervised prediction) agree.",
        "hypothesis": "Elapsed time since last purchase is the strongest available signal for disengagement, though the data cannot say whether this is due to product dissatisfaction, a one-off need being met, or switching to a competitor.",
        "recommendation": "Prioritize win-back/re-engagement campaigns for this segment given its size (46.9% of customers) and consistent risk signal.",
        "action": "Targeted re-engagement email/discount campaign; measure response via a controlled test rather than assuming effectiveness.",
    },
    {
        "group": "Single-purchase customers",
        "observation": "27.5% of customers have Frequency=1 (one order ever); their median Recency (387 days) is far higher than repeat customers' (60 days), and Is_SinglePurchase_asof_cutoff independently raised predicted risk in both models.",
        "insight": "A large share of the customer base never converts to a second purchase, and this pattern is detectable early.",
        "hypothesis": "The gap between 1st and 2nd purchase for customers who DO return has a 90th percentile of 258 days -- suggesting a natural conversion window worth acting within.",
        "recommendation": "Intervene with second-purchase incentives before the 258-day mark, rather than waiting for a fixed churn threshold to pass.",
        "action": "Automated 'second purchase' nudge sequence timed to typical repeat-purchase windows.",
    },
    {
        "group": "Developing / occasional customers (Cluster 2)",
        "observation": "31.9% of customers are recently active (avg Recency 42.9 days) but have low order counts (median 3) and moderate spend (avg Monetary GBP1,122); average predicted risk is moderate (0.506).",
        "insight": "This group is neither clearly loyal nor clearly at-risk -- they are active but haven't established a strong repeat pattern yet.",
        "hypothesis": "They may be newer customers still building purchase habits, or occasional buyers with a narrower need.",
        "recommendation": "Nurture rather than aggressively discount -- focus on building frequency through personalized product suggestions based on past purchases.",
        "action": "Cross-sell/upsell recommendations based on order history; monitor whether Frequency increases over subsequent quarters.",
    },
    {
        "group": "High-value customers (Clusters 0 and 3)",
        "observation": "Cluster 0 (19.5% of customers) contributes 39.5% of revenue with zero one-time buyers; Cluster 3 (1.7% of customers, cross-validated 99% against the Stage 3 bulk-buyer flag) contributes 36.6% of revenue alone. Combined average predicted risk is very low (0.164 and 0.033).",
        "insight": "A small fraction of customers accounts for the large majority of revenue, and they are currently low-risk.",
        "hypothesis": "Their low risk may reflect either genuine loyalty/business dependency (wholesale) or simply recency of very frequent ordering patterns.",
        "recommendation": "Protect this base with proactive account management rather than mass-market campaigns; treat any risk signal here as high priority regardless of its absolute probability.",
        "action": "Dedicated account contact for Cluster 3 (wholesale-like) customers; loyalty perks for Cluster 0 to sustain the pattern.",
    },
    {
        "group": "High-risk customers (Current_Risk_Tier = High)",
        "observation": "Applying the frozen, validated Random Forest to all 5,870 current customers flags a distinct High-risk group (Risk_Probability >= 0.70); risk probability correlates +0.78 with Recency and -0.45 with Frequency.",
        "insight": "The model's operational scoring lets the business identify specific at-risk individuals today, not just aggregate segments.",
        "hypothesis": "Recency dominance in both the coefficients and feature importances suggests time-since-purchase is the strongest available (not necessarily causal) predictor.",
        "recommendation": "Use the risk score to prioritize outreach effort, especially for any High-value + High-risk overlap customers, which are rare but disproportionately costly to lose.",
        "action": "Route High-risk customers into a retention workflow; A/B test interventions before scaling, since the model shows association, not proven causation.",
    },
]
