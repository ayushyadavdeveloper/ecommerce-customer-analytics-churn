import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.analysis import compute_kpis, BUSINESS_INSIGHT_BLOCKS

                                                                                
st.set_page_config(
    page_title="E-Commerce Analytics Hub",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

                                                                                
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8eaf0;
  }

  [data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
  }
  [data-testid="stSidebar"] .stRadio label { color: #c9cfe0 !important; }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #a78bfa !important; }

  [data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 16px 20px !important;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.25);
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #a5b4fc !important; font-size: 0.8rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f4ff !important; font-size: 1.8rem !important; font-weight: 700 !important; }

  h1 { color: #c4b5fd !important; font-weight: 700 !important; }
  h2 { color: #a5b4fc !important; font-weight: 600 !important; }
  h3 { color: #93c5fd !important; font-weight: 500 !important; }

  details {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
  }
  summary { color: #c4b5fd !important; }

  .stDataFrame { border-radius: 10px; overflow: hidden; }
  .stAlert { border-radius: 10px; }

  [data-testid="stTabs"] button { color: #a5b4fc !important; }
  [data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid #8b5cf6 !important;
    color: #c4b5fd !important;
  }

  hr { border-color: rgba(255,255,255,0.1); }

  .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    color: #e8eaf0 !important;
    border-color: rgba(255,255,255,0.15) !important;
  }

  .stMultiSelect > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.15) !important;
  }

  .dashboard-banner {
    background: linear-gradient(90deg, rgba(139,92,246,0.3), rgba(59,130,246,0.3));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
  }
  .dashboard-banner h2 { margin: 0 0 6px 0; color: #e0d9ff !important; }
  .dashboard-banner p  { margin: 0; color: #b0bcd8; font-size: 0.9rem; }

  .kpi-badge {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #c4b5fd;
    margin: 2px;
  }
  .risk-high   { background: rgba(239,68,68,0.2); border-color: rgba(239,68,68,0.5); color: #fca5a5; }
  .risk-medium { background: rgba(245,158,11,0.2); border-color: rgba(245,158,11,0.5); color: #fcd34d; }
  .risk-low    { background: rgba(34,197,94,0.2); border-color: rgba(34,197,94,0.5); color: #86efac; }
</style>
""", unsafe_allow_html=True)

                                                                                
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#c9cfe0"),
    title_font=dict(size=15, color="#c4b5fd"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(t=50, b=30, l=30, r=20),
    colorway=["#8b5cf6", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899"],
)

RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}

CLUSTER_NAMES = {
    0: "🔵 Active High-Value",
    1: "🔴 Long-Inactive / At-Risk",
    2: "🟡 Occasional / Developing",
    3: "💎 Bulk / Wholesale",
}
CLUSTER_DESC = {
    0: "Active, high-frequency, high-value repeat customers",
    1: "Long-inactive / largely one-and-done customers",
    2: "Occasional / developing customers",
    3: "High-value bulk/wholesale-like accounts",
}

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

                                                                                 
@st.cache_data
def load_data():
    customer_master  = pd.read_csv("data/processed/customer_master.csv")
    monthly_trends   = pd.read_csv("data/processed/monthly_trends.csv")
    country_revenue  = pd.read_csv("data/processed/country_revenue.csv")
    top_products     = pd.read_csv("data/processed/top_products.csv")
    test_predictions = pd.read_csv("data/processed/test_set_predictions.csv")
    with open("models/model_metadata.json") as f:
        metadata = json.load(f)
    k_selection      = pd.read_csv("outputs/stage4_kmeans_k_selection.csv")
    model_comparison = pd.read_csv("outputs/stage5_model_comparison.csv")
    return (customer_master, monthly_trends, country_revenue, top_products,
            test_predictions, metadata, k_selection, model_comparison)

(customer_master, monthly_trends, country_revenue, top_products,
 test_predictions, metadata, k_selection, model_comparison) = load_data()

                                                                                
with st.sidebar:
    st.markdown("## 📊 Analytics Hub")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        [
            "🏠 Executive Overview",
            "📈 Sales & Trends",
            "🧩 Customer Segments",
            "⚠️ Churn Risk",
            "💡 Business Insights",
            "🔍 Customer Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")
    risk_filter = st.multiselect(
        "Risk Tier", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    cluster_filter = st.multiselect(
        "Segment",
        options=[0, 1, 2, 3],
        format_func=lambda x: CLUSTER_NAMES[x],
        default=[0, 1, 2, 3],
    )

    st.markdown("---")
    st.caption(
        "📦 **Data:** UCI Online Retail II\n\n"
        "🔬 **Method:** Stages 1-5 — cleaning → RFM → K-Means → churn model\n\n"
        "🚫 Nothing is retrained here — reads pre-computed outputs only."
    )

                      
cm_filtered = customer_master[
    customer_master["Current_Risk_Tier"].isin(risk_filter) &
    customer_master["Cluster_K4"].isin(cluster_filter)
].copy()

                                                                                
                             
                                                                                
if page.startswith("🏠"):
    kpis = compute_kpis(customer_master, monthly_trends)

    st.markdown("""
    <div class="dashboard-banner">
      <h2>📊 Executive Analytics Dashboard</h2>
      <p>Real-time view of customer health, revenue performance, and churn risk signals — powered by validated ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",    f"£{kpis['Total Revenue']:,.0f}")
    c2.metric("👥 Total Customers",  f"{kpis['Total Customers']:,}")
    c3.metric("📦 Total Orders",     f"{kpis['Total Orders']:,}")
    c4.metric("🛒 Avg Order Value",  f"£{kpis['Average Order Value']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🔴 High-Risk",   f"{int((cm_filtered['Current_Risk_Tier']=='High').sum()):,}")
    c6.metric("🟡 Medium-Risk", f"{int((cm_filtered['Current_Risk_Tier']=='Medium').sum()):,}")
    c7.metric("🟢 Low-Risk",    f"{int((cm_filtered['Current_Risk_Tier']=='Low').sum()):,}")
    c8.metric("🔄 Repeat Rate", f"{kpis['Repeat Customer Rate (%)']:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📈 Revenue Momentum")
        monthly_trends["MA3"] = monthly_trends["Revenue"].rolling(3).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_trends["Month"], y=monthly_trends["Revenue"],
            name="Net Revenue", fill="tozeroy",
            line=dict(color="#8b5cf6", width=2.5),
            fillcolor="rgba(139,92,246,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=monthly_trends["Month"], y=monthly_trends["MA3"],
            name="3-Month MA", line=dict(color="#06b6d4", width=2, dash="dot"),
        ))
        apply_theme(fig)
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🎯 Risk Distribution")
        risk_counts = (
            cm_filtered["Current_Risk_Tier"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ["Tier", "Count"]
        fig = go.Figure(go.Pie(
            labels=risk_counts["Tier"],
            values=risk_counts["Count"],
            hole=0.55,
            marker=dict(colors=["#ef4444", "#f59e0b", "#22c55e"],
                        line=dict(color="#1a1a2e", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#e8eaf0"),
        ))
        apply_theme(fig)
        fig.update_layout(
            height=280, showlegend=False,
            annotations=[dict(text="Risk<br>Tiers", x=0.5, y=0.5,
                              font=dict(size=13, color="#c4b5fd"), showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 💼 Revenue by Segment")
        cluster_rev = cm_filtered.groupby("Cluster_K4")["Monetary"].sum().reset_index()
        cluster_rev["Name"] = cluster_rev["Cluster_K4"].map(CLUSTER_NAMES)
        cluster_rev["Pct"]  = cluster_rev["Monetary"] / cluster_rev["Monetary"].sum() * 100
        fig = px.bar(
            cluster_rev, x="Name", y="Monetary",
            text=cluster_rev["Pct"].apply(lambda x: f"{x:.1f}%"),
            color="Name",
            color_discrete_sequence=["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"],
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#e8eaf0"))
        apply_theme(fig)
        fig.update_layout(height=280, showlegend=False, xaxis_title="", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### 🔬 Recency vs. Churn Probability")
        sample = cm_filtered.sample(min(800, len(cm_filtered)), random_state=42)
        fig = px.scatter(
            sample, x="Recency", y="Current_Risk_Probability",
            color="Current_Risk_Tier",
            color_discrete_map=RISK_COLORS,
            opacity=0.7,
            hover_data={"Customer_ID": True, "Frequency": True, "Monetary": ":.0f"},
        )
        apply_theme(fig)
        fig.update_layout(height=280, xaxis_title="Recency (days)",
                           yaxis_title="Churn Probability")
        st.plotly_chart(fig, use_container_width=True)

    n_hv_hr = int(((cm_filtered["Cluster_K4"].isin([0, 3])) &
                   (cm_filtered["Current_Risk_Tier"] == "High")).sum())
    st.info(
        f"**⚡ Key Signal:** {n_hv_hr} high-value customers (Clusters 0 & 3) are currently "
        "flagged as **High-Risk** — disproportionately high revenue at stake. "
        "Prioritise for immediate retention outreach."
    )

                                                                                
                         
                                                                                
elif page.startswith("📈"):
    st.title("📈 Sales & Business Trends")

    available_months = sorted(monthly_trends["Month"].tolist())
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        month_range = st.select_slider(
            "Select Month Range",
            options=available_months,
            value=(available_months[0], available_months[-1]),
        )
    with col_ctrl2:
        show_ma = st.checkbox("Moving Average", value=True)

    mask = (monthly_trends["Month"] >= month_range[0]) & (monthly_trends["Month"] <= month_range[1])
    mt = monthly_trends[mask].copy()
    mt["MA3_Revenue"] = mt["Revenue"].rolling(3, min_periods=1).mean()

    st.markdown("### Revenue & Order Volume")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=mt["Month"], y=mt["Revenue"], name="Net Revenue",
        marker_color="rgba(139,92,246,0.7)",
    ), secondary_y=False)
    if show_ma:
        fig.add_trace(go.Scatter(
            x=mt["Month"], y=mt["MA3_Revenue"], name="Revenue MA(3)",
            line=dict(color="#c4b5fd", width=2, dash="dot"),
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=mt["Month"], y=mt["Orders"], name="Orders",
        line=dict(color="#06b6d4", width=2.5), mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, height=360)
    fig.update_yaxes(title_text="Revenue (£)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Order Count", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cancellation Impact")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mt["Month"], y=mt["Gross_Revenue"],
                              name="Gross Revenue", marker_color="rgba(59,130,246,0.7)"))
        fig.add_trace(go.Bar(x=mt["Month"], y=mt["Gross_Revenue"] - mt["Revenue"],
                              name="Cancelled", marker_color="rgba(239,68,68,0.7)"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=300,
                           title="Gross vs. Net Revenue")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mt["Cancel_Rate"] = mt["Cancelled_Orders"] / mt["Gross_Orders"] * 100
        fig = px.area(mt, x="Month", y="Cancel_Rate", title="Monthly Cancellation Rate (%)")
        fig.update_traces(line_color="#ef4444", fillcolor="rgba(239,68,68,0.15)")
        apply_theme(fig)
        fig.update_layout(height=300, yaxis_title="Cancellation Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌍 Geographic Revenue")
    top_n = st.slider("Show Top N Countries", 5, len(country_revenue), 15)
    top_countries = country_revenue.sort_values("Revenue", ascending=False).head(top_n)

    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🌐 Bubble", "📋 Table"])
    with tab1:
        fig = px.bar(
            top_countries.sort_values("Revenue"),
            x="Revenue", y="Country_clean", orientation="h",
            color="Revenue",
            color_continuous_scale=[[0, "#302b63"], [0.5, "#8b5cf6"], [1, "#c4b5fd"]],
            text=top_countries.sort_values("Revenue")["Revenue"].apply(lambda x: f"£{x/1e6:.2f}M"),
        )
        fig.update_traces(textposition="outside")
        apply_theme(fig)
        fig.update_layout(height=max(300, top_n * 32), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.scatter(
            top_countries, x="Orders", y="Revenue",
            size="Customers", color="Country_clean",
            hover_name="Country_clean", size_max=60,
        )
        apply_theme(fig)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        tc_display = top_countries.copy()
        tc_display["Revenue"] = tc_display["Revenue"].map("£{:,.0f}".format)
        st.dataframe(tc_display, use_container_width=True, hide_index=True)

    st.markdown("### 🛍️ Top Products")
    prod_n = st.slider("Top N Products", 5, len(top_products), 20)
    top_prod = top_products.sort_values("Revenue", ascending=False).head(prod_n)
    tab_p1, tab_p2 = st.tabs(["💰 By Revenue", "📦 By Quantity"])
    with tab_p1:
        fig = px.bar(top_prod.sort_values("Revenue"), x="Revenue", y="Description",
                      orientation="h", color="Revenue",
                      color_continuous_scale=[[0, "#302b63"], [1, "#8b5cf6"]])
        apply_theme(fig)
        fig.update_layout(height=max(300, prod_n * 28), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab_p2:
        fig = px.bar(top_prod.sort_values("Quantity_Sold"), x="Quantity_Sold",
                      y="Description", orientation="h", color="Quantity_Sold",
                      color_continuous_scale=[[0, "#0f172a"], [1, "#06b6d4"]])
        apply_theme(fig)
        fig.update_layout(height=max(300, prod_n * 28), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

                                                                                
                            
                                                                                
elif page.startswith("🧩"):
    st.title("🧩 Customer Segmentation")

    tier_order  = ["RFM 3-6 (low)", "RFM 7-9 (mid-low)", "RFM 10-12 (mid-high)", "RFM 13-15 (high)"]
    tier_colors = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e"]
    tier_col    = "RFM_Tier"

    st.markdown("### 📊 RFM Segmentation")
    tab_rfm1, tab_rfm2, tab_rfm3 = st.tabs(["Customer Count", "Revenue Share", "RFM Heatmap"])

    with tab_rfm1:
        tc = cm_filtered[tier_col].value_counts().reindex(tier_order).fillna(0).reset_index()
        tc.columns = ["RFM Tier", "Customers"]
        fig = px.bar(tc, x="RFM Tier", y="Customers",
                      color="RFM Tier", color_discrete_sequence=tier_colors, text="Customers")
        fig.update_traces(textposition="outside")
        apply_theme(fig)
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with tab_rfm2:
        rbt = cm_filtered.groupby(tier_col)["Monetary"].sum().reindex(tier_order).fillna(0).reset_index()
        fig = go.Figure(go.Pie(
            labels=rbt[tier_col], values=rbt["Monetary"], hole=0.5,
            marker=dict(colors=tier_colors, line=dict(color="#1a1a2e", width=2)),
            textinfo="label+percent", textfont=dict(color="#e8eaf0"),
        ))
        apply_theme(fig)
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_rfm3:
        hd = cm_filtered.groupby(["R_score", "F_score"])["Monetary"].mean().reset_index()
        hp = hd.pivot(index="R_score", columns="F_score", values="Monetary")
        fig = go.Figure(go.Heatmap(
            z=hp.values,
            x=[f"F={c}" for c in hp.columns],
            y=[f"R={r}" for r in hp.index],
            colorscale=[[0, "#0f0c29"], [0.5, "#8b5cf6"], [1, "#c4b5fd"]],
            text=hp.values, texttemplate="£%{z:.0f}",
        ))
        apply_theme(fig)
        fig.update_layout(title="Avg Revenue by R-score × F-score", height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗂️ K-Means Clusters (K=4)")

    col_left, col_right = st.columns([2, 3])
    with col_left:
        cc = cm_filtered["Cluster_K4"].value_counts().sort_index().reset_index()
        cc.columns = ["Cluster", "Count"]
        cc["Name"] = cc["Cluster"].map(CLUSTER_NAMES)
        fig = go.Figure(go.Pie(
            labels=cc["Name"], values=cc["Count"], hole=0.5,
            marker=dict(colors=["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"],
                        line=dict(color="#1a1a2e", width=2)),
            textinfo="label+percent",
        ))
        apply_theme(fig)
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        profile = cm_filtered.groupby("Cluster_K4").agg(
            N=("Customer_ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Med_Frequency=("Frequency", "median"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_AOV=("Average_Order_Value", "mean"),
            Avg_Risk=("Current_Risk_Probability", "mean"),
        ).round(2).reset_index()
        profile["Segment"] = profile["Cluster_K4"].map(CLUSTER_NAMES)
        st.dataframe(
            profile[["Segment", "N", "Avg_Recency", "Med_Frequency",
                       "Avg_Monetary", "Avg_AOV", "Avg_Risk"]].rename(columns={
                "N": "Customers", "Avg_Recency": "Avg Recency (d)",
                "Med_Frequency": "Med Freq", "Avg_Monetary": "Avg Revenue £",
                "Avg_AOV": "Avg AOV £", "Avg_Risk": "Avg Churn Risk",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### 🕸️ Cluster Profile Radar")
    rp = customer_master.groupby("Cluster_K4").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        AOV=("Average_Order_Value", "mean"),
        Risk=("Current_Risk_Probability", "mean"),
    )
    norm = (rp - rp.min()) / (rp.max() - rp.min() + 1e-9)
    norm["Recency"] = 1 - norm["Recency"]
    norm["Risk"]    = 1 - norm["Risk"]
    cats = ["Recency Score", "Frequency", "Monetary", "AOV", "Retention Score"]
    radar_colors = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]
    fig = go.Figure()
    for i, (cid, row_r) in enumerate(norm.iterrows()):
        if cid not in cluster_filter:
            continue
        vals = row_r.tolist() + [row_r.tolist()[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself",
            name=CLUSTER_NAMES[cid],
            line_color=radar_colors[i],
            opacity=0.85,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(color="#a5b4fc"),
            radialaxis=dict(color="#a5b4fc", gridcolor="rgba(255,255,255,0.1)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cfe0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📌 Why K=4? — Statistical Justification"):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(k_selection, x="K", y="Silhouette", markers=True,
                           title="Silhouette Score by K")
            fig.update_traces(line_color="#8b5cf6", marker_color="#c4b5fd")
            apply_theme(fig)
            fig.add_vline(x=4, line_dash="dash", line_color="#ef4444",
                          annotation_text="K=4 selected", annotation_font_color="#ef4444")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(k_selection, x="K", y="Inertia", markers=True,
                           title="Inertia (Elbow Method)")
            fig.update_traces(line_color="#06b6d4", marker_color="#67e8f9")
            apply_theme(fig)
            fig.add_vline(x=4, line_dash="dash", line_color="#ef4444")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔭 3D Customer Landscape")
    sample3d = cm_filtered.sample(min(1500, len(cm_filtered)), random_state=7)
    sample3d = sample3d.copy()
    sample3d["Segment"] = sample3d["Cluster_K4"].map(CLUSTER_NAMES)
    fig = px.scatter_3d(
        sample3d, x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        color_discrete_sequence=["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"],
        opacity=0.75,
        hover_data={"Customer_ID": True, "Current_Risk_Tier": True},
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cfe0"),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

                                                                                
                     
                                                                                
elif page.startswith("⚠️"):
    st.title("⚠️ Churn Risk Prediction")

    st.markdown(f"""
    <div class="dashboard-banner">
      <h2>Model Details</h2>
      <p>
        <b>Observation:</b> {metadata['observation_period'][0]} → {metadata['observation_period'][1]} &nbsp;|&nbsp;
        <b>Future window:</b> {metadata['future_window'][0]} → {metadata['future_window'][1]}
        ({metadata['future_window_days']} days) &nbsp;|&nbsp;
        <b>Churn def:</b> {metadata['churn_definition']}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Model Comparison (Test Set, n=994)")
    mc_display = model_comparison.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}",
    }).background_gradient(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                             cmap="Purples")
    st.dataframe(mc_display, use_container_width=True, hide_index=True)

    rf = model_comparison[model_comparison["Model"] == "Random Forest"].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{rf['Accuracy']:.3f}")
    c2.metric("Precision", f"{rf['Precision']:.3f}")
    c3.metric("Recall",    f"{rf['Recall']:.3f}")
    c4.metric("F1-Score",  f"{rf['F1']:.3f}")
    c5.metric("ROC-AUC",   f"{rf['ROC-AUC']:.3f}")

    st.markdown("---")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("### Confusion Matrix — Random Forest")
        cm_arr = confusion_matrix(
            test_predictions["Actual_Churn"],
            test_predictions["Predicted_Churn_RF"],
        )
        fig = go.Figure(go.Heatmap(
            z=cm_arr,
            x=["Predicted: Retained", "Predicted: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            colorscale=[[0, "#1a1a2e"], [1, "#8b5cf6"]],
            text=cm_arr, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False,
        ))
        apply_theme(fig)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_roc:
        st.markdown("### ROC Curve Comparison")
        fig = go.Figure()
        for col_name, label, color in [
            ("Churn_Probability_RF", "Random Forest", "#8b5cf6"),
            ("Churn_Probability_LR", "Logistic Regression", "#3b82f6"),
        ]:
            if col_name in test_predictions.columns:
                fpr, tpr, _ = roc_curve(test_predictions["Actual_Churn"],
                                         test_predictions[col_name])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, name=f"{label} (AUC={roc_auc:.3f})",
                    line=dict(color=color, width=2.5),
                ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="gray", dash="dash", width=1),
        ))
        apply_theme(fig)
        fig.update_layout(height=320, xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧬 Feature Importance — Random Forest")
    importances = pd.DataFrame({
        "Feature": metadata["feature_cols"],
        "Importance": [0.292325, 0.172241, 0.151594, 0.044471,
                        0.145005, 0.107331, 0.049908, 0.037126],
    }).sort_values("Importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=importances["Importance"], y=importances["Feature"],
        orientation="h",
        marker=dict(
            color=importances["Importance"],
            colorscale=[[0, "#302b63"], [0.5, "#8b5cf6"], [1, "#c4b5fd"]],
            showscale=False,
        ),
        text=importances["Importance"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))
    apply_theme(fig)
    fig.update_layout(height=360, xaxis_title="Importance Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Operational Risk Scoring")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            cm_filtered, x="Current_Risk_Probability", nbins=40,
            color="Current_Risk_Tier", color_discrete_map=RISK_COLORS,
            title="Churn Probability Distribution",
            barmode="overlay", opacity=0.8,
        )
        apply_theme(fig)
        fig.update_layout(height=300, xaxis_title="Risk Probability",
                           yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.70, 0.01)
        flagged   = int((cm_filtered["Current_Risk_Probability"] >= threshold).sum())
        rev_risk  = cm_filtered[cm_filtered["Current_Risk_Probability"] >= threshold]["Monetary"].sum()
        st.metric("Customers flagged", f"{flagged:,}")
        st.metric("Revenue at risk",   f"£{rev_risk:,.0f}")
        tier_thresh = (
            cm_filtered[cm_filtered["Current_Risk_Probability"] >= threshold]["Current_Risk_Tier"]
            .value_counts()
            .reset_index()
        )
        tier_thresh.columns = ["Tier", "Count"]
        fig = px.pie(tier_thresh, names="Tier", values="Count",
                      color="Tier", color_discrete_map=RISK_COLORS, hole=0.5)
        apply_theme(fig)
        fig.update_layout(height=240, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

                                                                                
                            
                                                                                
elif page.startswith("💡"):
    st.title("💡 Business Insights & Actions")
    st.caption("Evidence-driven: Observation → Insight → Hypothesis → Recommendation → Action")

    st.markdown("### 🗺️ Action Priority Matrix")
    priority_data = pd.DataFrame({
        "Segment": ["Long-Inactive (Cl.1)", "Single-Purchase",
                     "Developing (Cl.2)", "High-Value (Cl.0/3)", "High-Risk Overall"],
        "Urgency":        [5, 4, 2, 5, 5],
        "Revenue Impact": [3, 2, 3, 5, 5],
        "Effort":         [3, 2, 2, 4, 3],
        "Customers":      [2752, 1613, 1872, 1245,
                           int((customer_master["Current_Risk_Tier"] == "High").sum())],
    })
    fig = px.scatter(
        priority_data, x="Effort", y="Revenue Impact",
        size="Customers", color="Urgency", text="Segment",
        color_continuous_scale=[[0, "#302b63"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        size_max=60,
    )
    fig.update_traces(textposition="top center", textfont=dict(color="#e8eaf0", size=11))
    apply_theme(fig)
    fig.update_layout(height=380, xaxis_title="Effort (1=Low)", yaxis_title="Revenue Impact (1=Low)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    icons = ["🔴", "🟡", "🟠", "💎", "⚠️"]
    for i, block in enumerate(BUSINESS_INSIGHT_BLOCKS):
        with st.expander(f"{icons[i]} **{block['group']}**", expanded=(i == 0)):
            t1, t2, t3, t4, t5 = st.tabs(
                ["📍 Observation", "💡 Insight", "🔬 Hypothesis", "📋 Recommendation", "🚀 Action"]
            )
            t1.markdown(block["observation"])
            t2.markdown(block["insight"])
            t3.markdown(block["hypothesis"])
            t4.markdown(block["recommendation"])
            t5.markdown(f"**→ {block['action']}**")

                                                                                
                            
                                                                                
elif page.startswith("🔍"):
    st.title("🔍 Customer Risk Explorer")

    col_search, col_risk, col_seg = st.columns([2, 1, 1])
    with col_risk:
        risk_sel = st.multiselect("Risk Tier", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"], key="exp_risk")
    with col_seg:
        seg_sel = st.multiselect("Segment", [0, 1, 2, 3],
                                  format_func=lambda x: CLUSTER_NAMES[x],
                                  default=[0, 1, 2, 3], key="exp_seg")

    explorer_df = customer_master[
        customer_master["Current_Risk_Tier"].isin(risk_sel) &
        customer_master["Cluster_K4"].isin(seg_sel)
    ]
    with col_search:
        ids = sorted(explorer_df["Customer_ID"].astype(int).tolist())
        selected_id = st.selectbox(f"Select Customer ({len(ids):,} match)", ids)

    row = customer_master[customer_master["Customer_ID"].astype(int) == selected_id].iloc[0]

    risk_cls = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(
        row["Current_Risk_Tier"], "risk-low"
    )
    st.markdown(f"""
    <div class="dashboard-banner">
      <h2>Customer #{int(selected_id)}</h2>
      <p>
        Segment: <b>{CLUSTER_DESC.get(int(row['Cluster_K4']), '')}</b> &nbsp;|&nbsp;
        Country: <b>{row['Countries']}</b> &nbsp;|&nbsp;
        RFM: <b>{row['RFM_Segment']}</b>
        &nbsp;<span class="kpi-badge {risk_cls}">{row['Current_Risk_Tier']} Risk</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Churn Probability", f"{row['Current_Risk_Probability']:.1%}")
    c2.metric("⏱️ Recency (days)",    int(row["Recency"]))
    c3.metric("🔁 Frequency",         int(row["Frequency"]))
    c4.metric("💰 Lifetime Revenue",  f"£{row['Monetary']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🛒 Avg Order Value",      f"£{row['Average_Order_Value']:,.2f}")
    c6.metric("📦 Total Quantity",       int(row["Total_Quantity"]))
    c7.metric("📅 Tenure (days)",        int(row["Customer_Tenure_Days"]))
    c8.metric("🕐 Avg Purchase Interval", f"{row['Avg_Purchase_Interval_Days']:.0f}d")

    st.markdown("---")
    col_rfm, col_gauge = st.columns(2)

    with col_rfm:
        st.markdown("### 🧮 RFM Scores")
        fig = go.Figure()
        for dim, color in [("R_score", "#8b5cf6"), ("F_score", "#3b82f6"), ("M_score", "#06b6d4")]:
            label = dim.replace("_score", "").replace("R", "Recency").replace("F", "Frequency").replace("M", "Monetary")
            fig.add_trace(go.Bar(
                x=[row[dim]], y=[label],
                orientation="h", marker_color=color,
                name=label, showlegend=False, width=0.5,
            ))
        apply_theme(fig)
        fig.update_layout(height=200, xaxis=dict(range=[0, 5], dtick=1, **PLOTLY_LAYOUT["xaxis"]),
                           yaxis_title="", xaxis_title="Score (1-5)")
        st.plotly_chart(fig, use_container_width=True)

    with col_gauge:
        st.markdown("### 🎯 Churn Risk Gauge")
        prob = float(row["Current_Risk_Probability"])
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(suffix="%", font=dict(color="#e8eaf0", size=32)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#a5b4fc"),
                bar=dict(color="#8b5cf6"),
                steps=[
                    dict(range=[0, 30],   color="rgba(34,197,94,0.2)"),
                    dict(range=[30, 70],  color="rgba(245,158,11,0.2)"),
                    dict(range=[70, 100], color="rgba(239,68,68,0.2)"),
                ],
                threshold=dict(line=dict(color="#ef4444", width=3), thickness=0.8, value=70),
                bgcolor="rgba(0,0,0,0)",
            ),
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9cfe0"),
                           height=200, margin=dict(t=20, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚀 Recommended Action")
    cluster_actions = {
        0: "Offer **loyalty/retention perks** to sustain this customer's high-frequency repeat behavior.",
        1: "Deploy a **win-back/re-engagement campaign** — matches the highest-risk cluster profile.",
        2: "Nurture with **cross-sell/upsell campaigns** based on past order history.",
        3: "Assign **dedicated account management** — high-value wholesale-like relationship.",
    }
    base_action = cluster_actions.get(int(row["Cluster_K4"]), "Monitor this customer.")
    if row["Current_Risk_Tier"] == "High":
        st.error(f"⚠️ **HIGH RISK** — {prob:.1%} probability\n\n{base_action}\n\n**Priority: Immediate outreach.**")
    elif row["Current_Risk_Tier"] == "Medium":
        st.warning(f"🟠 **MEDIUM RISK** — {prob:.1%} probability\n\n{base_action}\n\n**Light-touch engagement recommended.**")
    else:
        st.success(f"✅ **LOW RISK** — {prob:.1%} probability\n\n{base_action}")

    st.markdown("### 📊 Percentile Ranking vs. All Customers")
    compare_cols = ["Recency", "Frequency", "Monetary", "Average_Order_Value", "Customer_Tenure_Days"]
    pct_df = pd.DataFrame({
        "Metric": compare_cols,
        "Percentile": [(customer_master[c] <= row[c]).mean() * 100 for c in compare_cols],
    })
    pct_df["Label"] = pct_df["Percentile"].apply(lambda x: f"{x:.0f}th")
    fig = go.Figure(go.Bar(
        x=pct_df["Percentile"], y=pct_df["Metric"],
        orientation="h", text=pct_df["Label"], textposition="outside",
        marker=dict(
            color=pct_df["Percentile"],
            colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
            showscale=False,
        ),
    ))
    apply_theme(fig)
    fig.update_layout(height=250, xaxis=dict(range=[0, 110], **PLOTLY_LAYOUT["xaxis"]),
                       xaxis_title="Percentile", yaxis_title="",
                       title="Customer Percentile vs. All Customers")
    st.plotly_chart(fig, use_container_width=True)
