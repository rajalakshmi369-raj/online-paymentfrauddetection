"""
SentinelPay — Online Payment Fraud Detection Management Dashboard
Data Science Programming Project
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import (
    MERCHANT_CATEGORIES,
    PAYMENT_CHANNELS,
    TRANSACTION_TYPES,
    generate_fraud_dataset,
    load_or_create_dataset,
)
from ml_engine import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    audit_dataset_quality,
    evaluate_at_threshold,
    score_single_transaction,
    train_and_evaluate_models,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelPay | Online Payment Fraud Detection Management",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global typography & spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Elegant KPI Card */
    .kpi-card {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        transition: all 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }
    .kpi-delta {
        font-size: 0.80rem;
        font-weight: 500;
    }

    /* Info / Insight Callout Card */
    .insight-box {
        background: rgba(30, 41, 59, 0.65);
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 0.95rem 1.2rem;
        margin: 0.75rem 0 1.25rem 0;
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* Module Feature Card */
    .module-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
    }
    .module-card h4 {
        margin-top: 0;
        color: #E2E8F0;
        font-size: 1.05rem;
    }
    .module-card p {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-bottom: 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.55)",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=48, b=20),
    legend=dict(
        bgcolor="rgba(15,23,42,0.7)",
        bordercolor="rgba(148,163,184,0.2)",
        borderwidth=1,
    ),
)

COLOR_LEGIT = "#10B981"
COLOR_FRAUD = "#F43F5E"
COLOR_PRIMARY = "#6366F1"
COLOR_ACCENT = "#38BDF8"


# ---------------------------------------------------------------------------
# Caching Data & ML Pipeline
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "data" / "transactions.csv"


@st.cache_data(show_spinner=False)
def get_base_dataset(n_samples: int = 6000) -> pd.DataFrame:
    return load_or_create_dataset(DATA_FILE, n_samples=n_samples)


@st.cache_resource(show_spinner=False)
def get_trained_ml_bundle(df: pd.DataFrame):
    return train_and_evaluate_models(df)


def render_kpi(label: str, value: str, subtitle: str, color: str = "#38BDF8") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color: {color};">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load Dataset & Models
# ---------------------------------------------------------------------------
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None

df_raw = st.session_state.custom_df if st.session_state.custom_df is not None else get_base_dataset(6000)
quality_report = audit_dataset_quality(df_raw)
ml_bundle = get_trained_ml_bundle(df_raw)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0 1rem 0; border-bottom: 1px solid rgba(148,163,184,0.18); margin-bottom: 1rem;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                🛡️ SentinelPay
            </div>
            <div style="font-size: 0.80rem; color: #94A3B8;">
                Online Payment Fraud Management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_module = st.radio(
        "Navigation Modules",
        options=[
            "🏠 Home & Overview",
            "📊 Fraud Analytics Dashboard",
            "🔍 Dataset Explorer & Audit",
            "🤖 ML Model Evaluation",
            "🛡️ Live Fraud Risk Simulator",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("##### 📌 Dataset Status")
    st.caption(
        f"**Records:** `{len(df_raw):,}` transactions\n\n"
        f"**Fraud Rate:** `{quality_report['fraud_rate_pct']:.2f}%` (`{quality_report['fraud_count']:,}` cases)\n\n"
        f"**Best ROC-AUC:** `{max(m.roc_auc for m in ml_bundle['models'].values()):.4f}`"
    )
    st.markdown("---")
    st.caption("Data Science Programming Project • Built with Python, Pandas, Scikit-Learn & Plotly")


# ===========================================================================
# MODULE 1: HOME & PROJECT OVERVIEW
# ===========================================================================
if selected_module == "🏠 Home & Overview":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Data Science Programming Project</div>
            <h1 class="hero-title">SentinelPay: Online Payment Fraud Detection & Management</h1>
            <p class="hero-subtitle">
                An end-to-end data science platform combining exploratory visual analytics, automated data quality auditing,
                leak-free machine learning pipelines, and real-time transaction risk scoring for digital payment ecosystems.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-level executive metrics
    total_volume = df_raw["amount"].sum()
    fraud_df = df_raw[df_raw["is_fraud"] == 1]
    fraud_volume = fraud_df["amount"].sum()
    avg_fraud_amt = fraud_df["amount"].mean()
    avg_legit_amt = df_raw[df_raw["is_fraud"] == 0]["amount"].mean()
    best_model_name = max(ml_bundle["models"], key=lambda k: ml_bundle["models"][k].roc_auc)
    best_model_auc = ml_bundle["models"][best_model_name].roc_auc

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi(
            "Total Transactions Analyzed",
            f"{len(df_raw):,}",
            f"${total_volume / 1e6:.2f}M Gross Payment Volume",
            "#38BDF8",
        )
    with k2:
        render_kpi(
            "Confirmed Fraud Cases",
            f"{len(fraud_df):,} ({quality_report['fraud_rate_pct']:.2f}%)",
            f"1 : {quality_report['imbalance_ratio']} Class Imbalance Ratio",
            "#F43F5E",
        )
    with k3:
        render_kpi(
            "Fraudulent Volume at Risk",
            f"${fraud_volume / 1e6:.2f}M",
            f"Avg Fraud Ticket: ${avg_fraud_amt:,.0f} (vs ${avg_legit_amt:,.0f} legit)",
            "#F59E0B",
        )
    with k4:
        render_kpi(
            "Champion ML Model ROC-AUC",
            f"{best_model_auc * 100:.2f}%",
            f"Model: {best_model_name} (Hold-Out Test)",
            "#10B981",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture & Module Walkthrough
    st.subheader("🧭 Platform Modules & Architecture")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="module-card">
                <h4>📊 Analytics Dashboard</h4>
                <p>Interactive multi-dimensional slices across payment rails, merchant categories, hourly temporal cycles, and account balance drain patterns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="module-card">
                <h4>🔍 Dataset Explorer & Audit</h4>
                <p>Complete schema dictionary, automated missing-value & anomaly checks, correlation matrix, dynamic row filtering, and CSV upload/export.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="module-card">
                <h4>🤖 ML Model Evaluation</h4>
                <p>Strict train/test split pipeline comparing Random Forest, Gradient Boosting & Logistic Regression with live decision-threshold tuning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="module-card">
                <h4>🛡️ Live Risk Simulator</h4>
                <p>Real-time transaction scoring engine with scenario presets, multi-model consensus probabilities, and explainable risk factor attribution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.25, 1.0])
    with col_left:
        st.subheader("📈 Daily Transaction Volume & Fraud Incidence Trend")
        daily_df = (
            df_raw.assign(date=df_raw["timestamp"].dt.date)
            .groupby("date", as_index=False)
            .agg(
                total_txns=("transaction_id", "count"),
                fraud_txns=("is_fraud", "sum"),
                total_amount=("amount", "sum"),
            )
        )
        daily_df["fraud_rate_pct"] = (daily_df["fraud_txns"] / daily_df["total_txns"]) * 100

        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Bar(
                x=daily_df["date"],
                y=daily_df["total_txns"],
                name="Total Transactions",
                marker_color="rgba(99, 102, 241, 0.45)",
                yaxis="y1",
            )
        )
        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["fraud_rate_pct"],
                name="Fraud Rate (%)",
                mode="lines+markers",
                line=dict(color=COLOR_FRAUD, width=2.5),
                marker=dict(size=6),
                yaxis="y2",
            )
        )
        fig_daily.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            title="30-Day Transaction Volume vs. Daily Fraud Rate (%)",
            yaxis=dict(title="Transactions / Day", gridcolor="rgba(148,163,184,0.1)"),
            yaxis2=dict(
                title="Fraud Rate (%)",
                overlaying="y",
                side="right",
                showgrid=False,
                ticksuffix="%",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_right:
        st.subheader("🔬 Data Science Methodology Pipeline")
        st.markdown(
            """
            <div class="insight-box">
                <strong>1. Domain Feature Engineering</strong><br>
                Beyond raw payment amounts, we compute <code>amount_to_balance_ratio</code> (detecting full-account drains),
                <code>balance_error_dest</code> (identifying mule accounts that immediately zero out funds), and off-hour velocity metrics.
            </div>
            <div class="insight-box" style="border-left-color: #10B981;">
                <strong>2. Leak-Free Preprocessing Architecture</strong><br>
                Following strict ML best practices, the dataset is partitioned via stratified <code>train_test_split</code>
                <strong>before</strong> fitting <code>StandardScaler</code> (numerical) and <code>OneHotEncoder</code> (nominal categorical) inside a Scikit-Learn <code>ColumnTransformer</code>.
            </div>
            <div class="insight-box" style="border-left-color: #F59E0B;">
                <strong>3. Cost-Sensitive Evaluation</strong><br>
                Because payment fraud is heavily imbalanced (~6–8% positive class), accuracy alone is misleading. We optimize for
                <strong>ROC-AUC</strong>, <strong>PR-AUC</strong>, <strong>Recall</strong> (catching fraud), and <strong>Precision</strong> (minimizing customer friction).
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# MODULE 2: FRAUD ANALYTICS DASHBOARD
# ===========================================================================
elif selected_module == "📊 Fraud Analytics Dashboard":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Exploratory Data Analysis & Visual Intelligence</div>
            <h1 class="hero-title">Interactive Fraud Analytics Dashboard</h1>
            <p class="hero-subtitle">
                Slice and filter payment activity across rails, channels, merchant verticals, and risk tiers to uncover anomalous behavioral signatures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Filter Bar
    with st.expander("🎛️ Filter Transactions (Payment Rail, Channel, Risk Tier & Amount Range)", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_types = st.multiselect(
                "Transaction Type",
                options=TRANSACTION_TYPES,
                default=TRANSACTION_TYPES,
            )
        with f2:
            sel_channels = st.multiselect(
                "Payment Channel",
                options=PAYMENT_CHANNELS,
                default=PAYMENT_CHANNELS,
            )
        with f3:
            sel_tiers = st.multiselect(
                "Risk Tier",
                options=["Low", "Medium", "High", "Critical"],
                default=["Low", "Medium", "High", "Critical"],
            )
        with f4:
            max_amt_val = float(df_raw["amount"].max())
            amt_range = st.slider(
                "Transaction Amount Range ($)",
                min_value=0.0,
                max_value=max_amt_val,
                value=(0.0, max_amt_val),
                step=500.0,
            )

    filtered_df = df_raw[
        (df_raw["transaction_type"].isin(sel_types))
        & (df_raw["payment_channel"].isin(sel_channels))
        & (df_raw["risk_tier"].isin(sel_tiers))
        & (df_raw["amount"].between(amt_range[0], amt_range[1]))
    ].copy()

    if filtered_df.empty:
        st.warning("No transactions match the current filter criteria. Please broaden your selection.")
        st.stop()

    filtered_df["Status"] = filtered_df["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})

    # Filtered KPI Row
    f_total = len(filtered_df)
    f_fraud = int(filtered_df["is_fraud"].sum())
    f_rate = (f_fraud / max(f_total, 1)) * 100
    f_vol = filtered_df["amount"].sum()
    f_fraud_vol = filtered_df.loc[filtered_df["is_fraud"] == 1, "amount"].sum()
    f_drained = int(filtered_df["is_account_drained"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Filtered Transactions", f"{f_total:,}", f"${f_vol / 1e6:.2f}M Total Filtered Volume", "#38BDF8")
    with c2:
        render_kpi("Detected Fraud Cases", f"{f_fraud:,}", f"{f_rate:.2f}% Fraud Incidence Rate", "#F43F5E")
    with c3:
        render_kpi("Fraud Exposure ($)", f"${f_fraud_vol:,.0f}", f"{(f_fraud_vol / max(f_vol, 1))*100:.1f}% of Dollar Volume", "#F59E0B")
    with c4:
        render_kpi("Full Account Drains", f"{f_drained:,}", "Sender balance reduced to $0.00", "#A855F7")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 Charts: Transaction Type Fraud Breakdown & Hourly Temporal Pattern
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        type_summary = (
            filtered_df.groupby("transaction_type", as_index=False)
            .agg(
                total=("transaction_id", "count"),
                fraud=("is_fraud", "sum"),
                fraud_vol=("amount", lambda s: s[filtered_df.loc[s.index, "is_fraud"] == 1].sum()),
            )
        )
        type_summary["fraud_rate"] = (type_summary["fraud"] / type_summary["total"]) * 100
        type_summary = type_summary.sort_values("fraud_rate", ascending=True)

        fig_type = go.Figure()
        fig_type.add_trace(
            go.Bar(
                y=type_summary["transaction_type"],
                x=type_summary["fraud_rate"],
                orientation="h",
                text=type_summary.apply(lambda r: f"{r['fraud_rate']:.1f}% ({int(r['fraud'])} cases)", axis=1),
                textposition="auto",
                marker=dict(
                    color=type_summary["fraud_rate"],
                    colorscale=[[0, "#38BDF8"], [0.5, "#6366F1"], [1.0, "#F43F5E"]],
                ),
                name="Fraud Rate (%)",
            )
        )
        fig_type.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            title="Fraud Incidence Rate by Payment Rail (%)",
            xaxis=dict(title="Fraud Rate (%)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with r1_col2:
        hourly_summary = (
            filtered_df.groupby("hour", as_index=False)
            .agg(
                total=("transaction_id", "count"),
                fraud=("is_fraud", "sum"),
            )
        )
        hourly_summary["fraud_rate"] = (hourly_summary["fraud"] / hourly_summary["total"]) * 100

        fig_hour = go.Figure()
        fig_hour.add_trace(
            go.Bar(
                x=hourly_summary["hour"],
                y=hourly_summary["total"],
                name="Transaction Count",
                marker_color="rgba(148, 163, 184, 0.28)",
                yaxis="y1",
            )
        )
        fig_hour.add_trace(
            go.Scatter(
                x=hourly_summary["hour"],
                y=hourly_summary["fraud_rate"],
                name="Hourly Fraud Rate (%)",
                mode="lines+markers",
                line=dict(color=COLOR_FRAUD, width=3),
                marker=dict(size=7, color=COLOR_FRAUD),
                yaxis="y2",
            )
        )
        # Highlight high-risk 00:00 - 05:00 window
        fig_hour.add_vrect(
            x0=-0.4,
            x1=5.4,
            fillcolor="rgba(244, 63, 94, 0.12)",
            line_width=0,
            annotation_text="Off-Hours Attack Window (00:00–05:00)",
            annotation_position="top left",
        )
        fig_hour.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            title="Hourly Transaction Volume vs. Fraud Rate Spike (24h Cycle)",
            xaxis=dict(title="Hour of Day (0–23)", dtick=2, gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Total Transactions", gridcolor="rgba(148,163,184,0.1)"),
            yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right", showgrid=False, ticksuffix="%"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <strong>📊 Analytical Finding — Rail & Temporal Vulnerability:</strong>
            <code>CASH_OUT</code> and <code>TRANSFER</code> rails exhibit more than <strong>2.5x</strong> the fraud incidence of routine utility or card payments.
            Furthermore, fraud rates surge sharply between <strong>00:00 and 05:00</strong> when automated account-takeover scripts operate while account holders are asleep.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 2 Charts: Sender Balance vs Amount Scatter (Account Drain) & Risk Signal Boxplot
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        sample_df = filtered_df.sample(n=min(1800, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="oldbalance_orig",
            y="amount",
            color="Status",
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD},
            opacity=0.75,
            hover_data=["transaction_id", "transaction_type", "device_risk_score", "velocity_24h"],
            title="Sender Initial Balance vs. Transaction Amount (Account Drain Signature)",
            labels={
                "oldbalance_orig": "Sender Balance Before Transaction ($)",
                "amount": "Transaction Amount ($)",
            },
        )
        # Add y = x diagonal line representing 100% balance drain
        max_axis = float(max(sample_df["oldbalance_orig"].max(), sample_df["amount"].max()))
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_axis],
                y=[0, max_axis],
                mode="lines",
                line=dict(color="#F59E0B", dash="dash", width=1.8),
                name="100% Balance Drain Line (y = x)",
            )
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with r2_col2:
        fig_box = px.violin(
            filtered_df,
            x="Status",
            y="device_risk_score",
            color="Status",
            box=True,
            points=False,
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD},
            title="Device Risk Telemetry Distribution: Legitimate vs. Fraudulent",
            labels={"device_risk_score": "Device Fingerprint Risk Score (0–100)"},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 3 Charts: Channel x Merchant Heatmap & Sunburst Hierarchy
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        pivot_fraud = (
            filtered_df.pivot_table(
                index="merchant_category",
                columns="payment_channel",
                values="is_fraud",
                aggfunc="mean",
            )
            * 100
        ).fillna(0.0)

        fig_heat = px.imshow(
            np.round(pivot_fraud, 1),
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#0F172A", "#4F46E5", "#F43F5E"],
            title="Fraud Rate (%) Matrix: Merchant Category × Payment Channel",
            labels=dict(x="Payment Channel", y="Merchant Category", color="Fraud %"),
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    with r3_col2:
        sun_df = (
            filtered_df.groupby(["risk_tier", "transaction_type", "Status"], as_index=False)
            .agg(volume=("amount", "sum"))
        )
        fig_sun = px.sunburst(
            sun_df,
            path=["risk_tier", "transaction_type", "Status"],
            values="volume",
            color="Status",
            color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraudulent": COLOR_FRAUD, "(?)": "#334155"},
            title="Payment Dollar Volume Hierarchy: Risk Tier → Rail → Fraud Status",
        )
        fig_sun.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_sun, use_container_width=True)


# ===========================================================================
# MODULE 3: DATASET EXPLORER & QUALITY AUDIT
# ===========================================================================
elif selected_module == "🔍 Dataset Explorer & Audit":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Data Engineering & Quality Assurance</div>
            <h1 class="hero-title">Dataset Explorer & Automated Quality Audit</h1>
            <p class="hero-subtitle">
                Inspect the transaction schema, verify missing-value and anomaly checks, explore feature correlations, or upload a custom CSV dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Quality Audit Cards
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_kpi(
            "Total Records & Features",
            f"{quality_report['total_rows']:,} × {quality_report['total_columns']}",
            "Complete schema validated",
            "#10B981",
        )
    with q2:
        render_kpi(
            "Missing / NULL Values",
            f"{quality_report['total_missing_cells']}",
            "Median/Mode imputers active in pipeline",
            "#38BDF8",
        )
    with q3:
        render_kpi(
            "Invalid / Non-Positive Amounts",
            f"{quality_report['invalid_amount_rows']}",
            "Automatically filtered pre-training",
            "#10B981",
        )
    with q4:
        render_kpi(
            "Target Class Balance",
            f"{quality_report['fraud_rate_pct']:.2f}% Fraud",
            f"{quality_report['legit_count']:,} Legit vs. {quality_report['fraud_count']:,} Fraud",
            "#F59E0B",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_browse, tab_schema, tab_corr, tab_manage = st.tabs(
        [
            "📋 Interactive Transaction Browser",
            "📖 Schema & Feature Dictionary",
            "🔗 Correlation & Statistical Summary",
            "⚙️ Dataset Generator / CSV Upload",
        ]
    )

    with tab_browse:
        b1, b2, b3 = st.columns([1.5, 1.0, 1.0])
        with b1:
            search_id = st.text_input("🔎 Search by Transaction ID (e.g., TXN-100045)", "")
        with b2:
            status_filter = st.selectbox("Filter by Class Label", ["All Transactions", "Fraudulent Only (is_fraud=1)", "Legitimate Only (is_fraud=0)"])
        with b3:
            sort_col = st.selectbox("Sort By Column", ["timestamp", "amount", "device_risk_score", "amount_to_balance_ratio", "velocity_24h"], index=1)

        view_df = df_raw.copy()
        if search_id.strip():
            view_df = view_df[view_df["transaction_id"].str.contains(search_id.strip(), case=False, na=False)]
        if status_filter == "Fraudulent Only (is_fraud=1)":
            view_df = view_df[view_df["is_fraud"] == 1]
        elif status_filter == "Legitimate Only (is_fraud=0)":
            view_df = view_df[view_df["is_fraud"] == 0]

        view_df = view_df.sort_values(sort_col, ascending=False)

        st.caption(f"Showing top `500` of `{len(view_df):,}` matching records.")
        st.dataframe(
            view_df.head(500),
            use_container_width=True,
            height=420,
        )

        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered Dataset as CSV",
            data=csv_bytes,
            file_name="sentinelpay_filtered_transactions.csv",
            mime="text/csv",
        )

    with tab_schema:
        schema_rows = [
            ("transaction_id", "Identifier", "Unique alphanumeric transaction identifier (e.g., TXN-100001)"),
            ("timestamp", "Datetime", "Exact date and time when the payment was initiated"),
            ("hour / is_night", "Numeric / Binary", "Hour of day (0–23) and indicator for high-risk 00:00–05:00 window"),
            ("transaction_type", "Nominal Categorical", "Payment rail: TRANSFER, CASH_OUT, ONLINE_PURCHASE, CARD_PAYMENT, UPI_P2P, BILL_PAYMENT"),
            ("payment_channel", "Nominal Categorical", "Initiation channel: Mobile App, Web Checkout, UPI / QR, POS Terminal, API Gateway"),
            ("merchant_category", "Nominal Categorical", "Merchant industry vertical (E-Commerce, Crypto, Luxury, etc.)"),
            ("amount", "Numeric (USD)", "Transaction dollar value"),
            ("oldbalance_orig / newbalance_orig", "Numeric (USD)", "Sender account balance before and after the transaction"),
            ("oldbalance_dest / newbalance_dest", "Numeric (USD)", "Recipient account balance before and after the transaction"),
            ("amount_to_balance_ratio", "Engineered Numeric", "Ratio of transaction amount to sender's opening balance (captures account drain)"),
            ("balance_error_dest", "Engineered Numeric", "Discrepancy between expected and actual recipient balance increase (mule layering signal)"),
            ("is_account_drained", "Engineered Binary", "1 if a funded account (> $500) is completely emptied to $0.00"),
            ("device_risk_score", "Numeric (0–100)", "Device telemetry risk score (emulator detection, browser spoofing, new device)"),
            ("velocity_24h", "Numeric (Count)", "Number of transactions initiated by the sender in the trailing 24 hours"),
            ("ip_country_mismatch", "Binary (0/1)", "1 if originating IP geolocation differs from the user's registered country"),
            ("is_flagged_rule", "Binary (0/1)", "Legacy static rule flag triggered only when amount > $50,000"),
            ("is_fraud", "Target Binary (0/1)", "Ground-truth target class: 1 = Confirmed Fraud, 0 = Legitimate"),
        ]
        st.table(pd.DataFrame(schema_rows, columns=["Feature Name", "Data Type", "Domain Description"]))

    with tab_corr:
        st.subheader("📈 Numerical Feature Correlation Matrix")
        corr_cols = NUMERIC_FEATURES + ["is_fraud"]
        corr_matrix = df_raw[corr_cols].corr()

        fig_corr = px.imshow(
            np.round(corr_matrix, 2),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix Across Numeric & Engineered Features",
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=520)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📊 Descriptive Statistics (Numerical Features)")
        st.dataframe(df_raw[corr_cols].describe().T.style.format("{:.2f}"), use_container_width=True)

    with tab_manage:
        col_gen, col_up = st.columns(2)
        with col_gen:
            st.markdown("#### 🔄 Regenerate Synthetic Benchmark Dataset")
            st.write("Generate a fresh PaySim-style transaction dataset with custom sample size and random seed.")
            new_n = st.number_input("Number of Transactions", min_value=1000, max_value=25000, value=6000, step=1000)
            new_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
            if st.button("🚀 Generate & Retrain Models"):
                new_df = generate_fraud_dataset(n_samples=int(new_n), random_state=int(new_seed))
                new_df.to_csv(DATA_FILE, index=False)
                st.session_state.custom_df = new_df
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Generated {len(new_df):,} transactions and retrained models!")
                st.rerun()

        with col_up:
            st.markdown("#### 📂 Upload Custom Transaction CSV")
            st.write("Upload a CSV containing the required feature columns to analyze your own dataset.")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    up_df = pd.read_csv(uploaded_file)
                    if "timestamp" in up_df.columns:
                        up_df["timestamp"] = pd.to_datetime(up_df["timestamp"])
                    missing_req = set(ALL_MODEL_FEATURES + ["is_fraud"]) - set(up_df.columns)
                    if missing_req:
                        st.error(f"Uploaded CSV is missing required columns: {sorted(missing_req)}")
                    else:
                        st.session_state.custom_df = up_df
                        st.success(f"Loaded custom dataset with {len(up_df):,} rows!")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not parse CSV: {exc}")


# ===========================================================================
# MODULE 4: ML MODEL EVALUATION & COMPARISON
# ===========================================================================
elif selected_module == "🤖 ML Model Evaluation":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Supervised Machine Learning & Cost-Sensitive Evaluation</div>
            <h1 class="hero-title">ML Model Performance, Comparison & Threshold Tuning</h1>
            <p class="hero-subtitle">
                Compare Random Forest, Gradient Boosting, and Logistic Regression trained via leak-free pipelines on a stratified 75% / 25% Train-Test split.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models_dict = ml_bundle["models"]

    # Model Comparison Leaderboard Table
    comp_rows = []
    for name, res in models_dict.items():
        comp_rows.append(
            {
                "Model Architecture": name,
                "ROC-AUC": res.roc_auc,
                "PR-AUC (Avg Precision)": res.pr_auc,
                "F1-Score (t=0.50)": res.f1,
                "Recall / Fraud Catch Rate": res.recall,
                "Precision": res.precision,
                "Accuracy": res.accuracy,
            }
        )
    comp_df = pd.DataFrame(comp_rows).sort_values("ROC-AUC", ascending=False)

    st.subheader("🏆 Multi-Model Benchmark Leaderboard (Hold-Out Test Partition)")
    st.caption(
        f"Evaluated on `{ml_bundle['test_count']:,}` unseen test transactions "
        f"(Trained on `{ml_bundle['train_count']:,}` transactions with strict pre-split scaling & encoding)."
    )
    st.dataframe(
        comp_df.style.format(
            {
                "ROC-AUC": "{:.4f}",
                "PR-AUC (Avg Precision)": "{:.4f}",
                "F1-Score (t=0.50)": "{:.4f}",
                "Recall / Fraud Catch Rate": "{:.2%}",
                "Precision": "{:.2%}",
                "Accuracy": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Model Selector & Decision Threshold Tuner
    st.subheader("🎛️ Interactive Decision Threshold & Cost-Benefit Analyzer")
    ctl1, ctl2 = st.columns([1.1, 1.4])
    with ctl1:
        active_model_name = st.selectbox(
            "Select Model for Deep-Dive Inspection",
            options=list(models_dict.keys()),
            index=0,
        )
    with ctl2:
        custom_threshold = st.slider(
            "Classification Probability Threshold (Lower = Catch More Fraud, Higher = Fewer False Alarms)",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
        )

    active_eval = models_dict[active_model_name]
    thresh_metrics = evaluate_at_threshold(active_eval, threshold=custom_threshold)

    tk1, tk2, tk3, tk4 = st.columns(4)
    with tk1:
        render_kpi(
            f"Recall @ t={custom_threshold:.2f}",
            f"{thresh_metrics['recall']*100:.1f}%",
            f"Caught {thresh_metrics['tp']} of {thresh_metrics['tp'] + thresh_metrics['fn']} frauds",
            "#10B981",
        )
    with tk2:
        render_kpi(
            f"Precision @ t={custom_threshold:.2f}",
            f"{thresh_metrics['precision']*100:.1f}%",
            f"{thresh_metrics['fp']} False Positives (Customer Friction)",
            "#38BDF8",
        )
    with tk3:
        render_kpi(
            f"F1-Score @ t={custom_threshold:.2f}",
            f"{thresh_metrics['f1']:.4f}",
            "Harmonic mean of Precision & Recall",
            "#6366F1",
        )
    with tk4:
        # Compare against static $50k legacy rule
        est_saved = thresh_metrics["tp"] * 6800 - thresh_metrics["fp"] * 25
        render_kpi(
            "Estimated Net Fraud Savings",
            f"${est_saved:,.0f}",
            f"Missed Frauds (FN): {thresh_metrics['fn']} cases",
            "#F59E0B",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix + ROC & PR Curves
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        cm = thresh_metrics["cm"]
        cm_labels = [
            [f"True Negative (TN)<br>{cm[0,0]:,}", f"False Positive (FP)<br>{cm[0,1]:,}"],
            [f"False Negative (FN)<br>{cm[1,0]:,}", f"True Positive (TP)<br>{cm[1,1]:,}"],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
                y=["Actual Legitimate (0)", "Actual Fraud (1)"],
                colorscale=[[0, "#0F172A"], [0.5, "#312E81"], [1.0, "#6366F1"]],
                showscale=False,
            )
        )
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=["Predicted Legitimate (0)", "Predicted Fraud (1)"][j],
                    y=["Actual Legitimate (0)", "Actual Fraud (1)"][i],
                    text=f"<b>{cm_labels[i][j]}</b>",
                    showarrow=False,
                    font=dict(color="#F8FAFC", size=14),
                )
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            height=390,
            title=f"Confusion Matrix — {active_model_name} (Threshold = {custom_threshold:.2f})",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with m_col2:
        fig_roc = go.Figure()
        palette = {"Random Forest": "#10B981", "Gradient Boosting": "#38BDF8", "Logistic Regression (Baseline)": "#F59E0B"}
        for name, res in models_dict.items():
            fig_roc.add_trace(
                go.Scatter(
                    x=res.fpr,
                    y=res.tpr,
                    mode="lines",
                    name=f"{name} (AUC = {res.roc_auc:.3f})",
                    line=dict(width=2.6, color=palette.get(name, "#6366F1")),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Baseline (AUC = 0.500)",
                line=dict(color="#64748B", dash="dash"),
            )
        )
        fig_roc.update_layout(
            **PLOTLY_LAYOUT,
            height=390,
            title="ROC Curve Comparison Across Models",
            xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="True Positive Rate (Recall)", gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Row 2: Feature Importance & Precision-Recall Curve
    m2_col1, m2_col2 = st.columns(2)

    with m2_col1:
        fi_df = ml_bundle["feature_importances"].head(12).sort_values("importance", ascending=True)
        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 12 Predictive Features (Random Forest Gini Importance)",
            labels={"importance": "Relative Importance Score", "feature": "Engineered / Encoded Feature"},
            color="importance",
            color_continuous_scale=["#38BDF8", "#6366F1", "#F43F5E"],
        )
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with m2_col2:
        fig_pr = go.Figure()
        for name, res in models_dict.items():
            fig_pr.add_trace(
                go.Scatter(
                    x=res.pr_recall,
                    y=res.pr_precision,
                    mode="lines",
                    name=f"{name} (PR-AUC = {res.pr_auc:.3f})",
                    line=dict(width=2.6, color=palette.get(name, "#6366F1")),
                )
            )
        fig_pr.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            title="Precision-Recall Curve (Critical for Imbalanced Fraud Data)",
            xaxis=dict(title="Recall (Sensitivity)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Precision (Positive Predictive Value)", gridcolor="rgba(148,163,184,0.1)"),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <strong>🤖 Model Selection Conclusion:</strong>
            Both tree-based ensembles (<strong>Random Forest</strong> and <strong>Gradient Boosting</strong>) significantly outperform the linear
            <strong>Logistic Regression baseline</strong> on Precision-Recall AUC because they capture non-linear interactions—such as a high
            <code>amount_to_balance_ratio</code> occurring specifically on <code>TRANSFER</code> or <code>CASH_OUT</code> rails during off-hours.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# MODULE 5: LIVE FRAUD RISK SIMULATOR
# ===========================================================================
elif selected_module == "🛡️ Live Fraud Risk Simulator":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Real-Time Inference & Explainable AI</div>
            <h1 class="hero-title">Live Transaction Fraud Risk Scorer</h1>
            <p class="hero-subtitle">
                Simulate incoming payment requests or load real-world attack presets to inspect model probability scores,
                engineered risk indicators, and automated policy decisions in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Scenario Presets
    st.markdown("##### ⚡ Load a Transaction Scenario Preset")
    p_col1, p_col2, p_col3 = st.columns(3)

    if "sim_preset" not in st.session_state:
        st.session_state.sim_preset = "ato_drain"

    with p_col1:
        if st.button("🚨 Preset 1: Account Takeover (Full Drain)", use_container_width=True):
            st.session_state.sim_preset = "ato_drain"
    with p_col2:
        if st.button("⚠️ Preset 2: High-Velocity Crypto Checkout", use_container_width=True):
            st.session_state.sim_preset = "crypto_velocity"
    with p_col3:
        if st.button("✅ Preset 3: Everyday Legitimate Retail Purchase", use_container_width=True):
            st.session_state.sim_preset = "legit_retail"

    presets = {
        "ato_drain": {
            "transaction_type": "TRANSFER",
            "payment_channel": "Web Checkout",
            "merchant_category": "Peer-to-Peer Transfer",
            "amount": 18450.0,
            "oldbalance_orig": 18500.0,
            "oldbalance_dest": 1200.0,
            "dest_credited": False,
            "hour": 2,
            "device_risk_score": 88.5,
            "velocity_24h": 7,
            "ip_country_mismatch": True,
        },
        "crypto_velocity": {
            "transaction_type": "ONLINE_PURCHASE",
            "payment_channel": "API Gateway",
            "merchant_category": "Crypto & Digital Assets",
            "amount": 4200.0,
            "oldbalance_orig": 9500.0,
            "oldbalance_dest": 45000.0,
            "dest_credited": True,
            "hour": 4,
            "device_risk_score": 64.0,
            "velocity_24h": 6,
            "ip_country_mismatch": True,
        },
        "legit_retail": {
            "transaction_type": "CARD_PAYMENT",
            "payment_channel": "Mobile App",
            "merchant_category": "E-Commerce & Retail",
            "amount": 125.0,
            "oldbalance_orig": 6400.0,
            "oldbalance_dest": 18000.0,
            "dest_credited": True,
            "hour": 14,
            "device_risk_score": 12.0,
            "velocity_24h": 1,
            "ip_country_mismatch": False,
        },
    }
    p = presets[st.session_state.sim_preset]

    st.markdown("<br>", unsafe_allow_html=True)

    sim_left, sim_right = st.columns([1.05, 1.25])

    with sim_left:
        st.subheader("📝 Transaction Parameters")
        c_a, c_b = st.columns(2)
        with c_a:
            in_type = st.selectbox(
                "Transaction Rail",
                TRANSACTION_TYPES,
                index=TRANSACTION_TYPES.index(p["transaction_type"]),
            )
            in_channel = st.selectbox(
                "Payment Channel",
                PAYMENT_CHANNELS,
                index=PAYMENT_CHANNELS.index(p["payment_channel"]),
            )
            in_merchant = st.selectbox(
                "Merchant Category",
                MERCHANT_CATEGORIES,
                index=MERCHANT_CATEGORIES.index(p["merchant_category"]),
            )
            in_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=1.0,
                max_value=500000.0,
                value=float(p["amount"]),
                step=100.0,
            )
            in_orig_bal = st.number_input(
                "Sender Opening Balance ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=float(p["oldbalance_orig"]),
                step=250.0,
            )

        with c_b:
            in_dest_bal = st.number_input(
                "Recipient Opening Balance ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=float(p["oldbalance_dest"]),
                step=250.0,
            )
            in_hour = st.slider("Hour of Day (0 = Midnight, 23 = 11 PM)", 0, 23, int(p["hour"]))
            in_device = st.slider("Device Fingerprint Risk Score (0–100)", 0.0, 100.0, float(p["device_risk_score"]), 0.5)
            in_velocity = st.slider("Sender 24h Transaction Velocity", 1, 20, int(p["velocity_24h"]))
            in_ip_mismatch = st.checkbox("Cross-Border IP / Geolocation Mismatch", value=bool(p["ip_country_mismatch"]))
            in_dest_zeroed = st.checkbox(
                "Recipient Mule Behavior (Funds rapidly moved/uncredited)",
                value=not bool(p["dest_credited"]),
            )

    new_dest_balance = in_dest_bal if in_dest_zeroed else (in_dest_bal + in_amount)

    txn_payload = {
        "transaction_type": in_type,
        "payment_channel": in_channel,
        "merchant_category": in_merchant,
        "amount": in_amount,
        "oldbalance_orig": in_orig_bal,
        "oldbalance_dest": in_dest_bal,
        "newbalance_dest": new_dest_balance,
        "hour": in_hour,
        "device_risk_score": in_device,
        "velocity_24h": in_velocity,
        "ip_country_mismatch": int(in_ip_mismatch),
    }

    score_result = score_single_transaction(ml_bundle["models"], txn_payload, primary_model="Random Forest")

    with sim_right:
        st.subheader("🎯 Real-Time Risk Assessment & Decision")

        prob_pct = score_result["primary_probability"] * 100
        badge_col = score_result["badge_color"]

        st.markdown(
            f"""
            <div style="background: rgba(17,24,39,0.9); border: 2px solid {badge_col}; border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:0.78rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Automated Policy Decision</div>
                        <div style="font-size:1.25rem; font-weight:800; color:#F8FAFC; margin-top:0.2rem;">{score_result['recommended_action']}</div>
                    </div>
                    <div style="background:{badge_col}; color:#0B0F19; font-weight:800; padding:0.4rem 0.9rem; border-radius:999px; font-size:0.9rem;">
                        {score_result['risk_level']} RISK
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC"}},
                title={"text": "Random Forest Fraud Probability Score", "font": {"size": 15, "color": "#94A3B8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                    "bar": {"color": badge_col},
                    "bgcolor": "rgba(30,41,59,0.6)",
                    "steps": [
                        {"range": [0, 22], "color": "rgba(16, 185, 129, 0.22)"},
                        {"range": [22, 45], "color": "rgba(59, 130, 246, 0.22)"},
                        {"range": [45, 75], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [75, 100], "color": "rgba(239, 68, 68, 0.28)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Multi-model consensus bar
        st.markdown("###### 🤖 Multi-Model Consensus Probabilities")
        mc1, mc2, mc3 = st.columns(3)
        for col_widget, (m_name, m_prob) in zip([mc1, mc2, mc3], score_result["model_scores"].items()):
            with col_widget:
                short_name = m_name.replace(" (Baseline)", "")
                st.metric(label=short_name, value=f"{m_prob * 100:.1f}%")

        # Explainable Risk Factor Table
        st.markdown("###### 🔍 Explainable Risk Factor Attribution")
        rf_df = pd.DataFrame(score_result["risk_factors"])
        st.dataframe(rf_df, use_container_width=True, hide_index=True)
