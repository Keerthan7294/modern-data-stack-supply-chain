"""
streamlit_app.py
----------------
Executive Supply Chain Inventory & Risk Forecasting Dashboard.
Connects to dbt marts tables in PostgreSQL and runs high-performance
NumPy Monte Carlo simulations (10,000 iterations) projecting future
inventory trajectories under lead time volatility.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text

# Page Configuration
st.set_page_config(
    page_title="Supply Chain Risk & Inventory Monte Carlo",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-overstocked {
        background-color: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Database & Cache Configurations
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "supply_chain_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
PARQUET_BACKUP = os.path.join(os.path.dirname(__file__), "data", "raw_supply_chain_orders.parquet")


@st.cache_data(ttl=60)
def load_marts_data():
    """
    Connects to the dbt marts table (analytics.fct_inventory_metrics) in PostgreSQL.
    Falls back gracefully to local parquet computation if PostgreSQL container is offline.
    """
    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(conn_str, connect_args={"connect_timeout": 3})
        query = "SELECT * FROM analytics.fct_inventory_metrics ORDER BY current_stock_level ASC;"
        df = pd.read_sql(query, engine)
        return df, "PostgreSQL (analytics.fct_inventory_metrics)"
    except Exception as e:
        # Fallback to computing mart metrics directly from parquet archive
        if os.path.exists(PARQUET_BACKUP):
            raw_df = pd.read_parquet(PARQUET_BACKUP)
            
            # Replicate dbt staging & marts aggregation in pandas
            # 1. Deduplicate by order_id
            dedup_df = raw_df.sort_values("order_date", ascending=False).drop_duplicates(subset=["order_id"]).copy()
            dedup_df["net_units_sold"] = dedup_df["units_ordered"].apply(lambda x: max(0, x))
            dedup_df["order_date"] = pd.to_datetime(dedup_df["order_date"]).dt.date
            
            # 2. Daily demand stats
            daily = dedup_df.groupby(["sku_id", "order_date"])["net_units_sold"].sum().reset_index()
            demand_stats = daily.groupby("sku_id")["net_units_sold"].agg(
                avg_daily_demand="mean",
                stddev_daily_demand=lambda s: s.std(ddof=1) if len(s) > 1 else 1.0,
                max_daily_demand="max"
            ).reset_index()
            
            # 3. Lead time stats
            delivered = dedup_df[dedup_df["actual_lead_time_days"].notnull()].copy()
            delivered["is_delayed"] = delivered["actual_lead_time_days"] > delivered["promised_lead_time_days"]
            lead_stats = delivered.groupby("sku_id").agg(
                avg_lead_time_days=("actual_lead_time_days", "mean"),
                stddev_lead_time_days=("actual_lead_time_days", lambda s: s.std(ddof=1) if len(s) > 1 else 1.5),
                max_lead_time_days=("actual_lead_time_days", "max"),
                delayed_shipment_rate_pct=("is_delayed", lambda s: round(100.0 * s.mean(), 2))
            ).reset_index()
            
            # 4. SKU metadata
            sku_meta = dedup_df.groupby("sku_id").agg(
                product_name=("product_name", "first"),
                category=("category", lambda s: s.iloc[0].strip().title()),
                supplier_name=("supplier_name", lambda s: s.iloc[0].strip()),
                warehouse_id=("warehouse_id", "first"),
                shipping_mode=("shipping_mode", "first"),
                unit_price=("unit_price", "mean"),
                current_stock_level=("current_stock_level", "first"),
                reorder_batch_size=("reorder_batch_size", "first"),
                total_orders_placed=("order_id", "nunique"),
                total_units_sold=("net_units_sold", "sum"),
                return_rate_pct=("units_ordered", lambda s: round(100.0 * (s < 0).mean(), 2))
            ).reset_index()
            
            # Merge
            marts = sku_meta.merge(demand_stats, on="sku_id", how="left").merge(lead_stats, on="sku_id", how="left")
            marts["avg_daily_demand"] = marts["avg_daily_demand"].round(2)
            marts["stddev_daily_demand"] = marts["stddev_daily_demand"].round(2).fillna(1.0)
            marts["avg_lead_time_days"] = marts["avg_lead_time_days"].round(2).fillna(7.0)
            marts["stddev_lead_time_days"] = marts["stddev_lead_time_days"].round(2).fillna(1.5)
            
            # Safety Stock: SS = 1.65 * sqrt(L * sig_D^2 + D^2 * sig_L^2)
            ss_term = (marts["avg_lead_time_days"] * (marts["stddev_daily_demand"] ** 2)) + \
                      ((marts["avg_daily_demand"] ** 2) * (marts["stddev_lead_time_days"] ** 2))
            marts["safety_stock_units"] = np.round(1.65 * np.sqrt(ss_term)).astype(int)
            marts["reorder_point_units"] = np.round((marts["avg_daily_demand"] * marts["avg_lead_time_days"]) + marts["safety_stock_units"]).astype(int)
            marts["days_of_supply"] = np.round(marts["current_stock_level"] / marts["avg_daily_demand"], 1)
            
            # Risk categories
            def classify(row):
                if row["current_stock_level"] <= row["safety_stock_units"]:
                    return "CRITICAL"
                elif row["current_stock_level"] <= row["reorder_point_units"]:
                    return "HIGH RISK"
                elif row["current_stock_level"] <= (row["reorder_point_units"] * 2.2):
                    return "HEALTHY"
                return "OVERSTOCKED"
                
            marts["stockout_risk_status"] = marts.apply(classify, axis=1)
            marts["is_reorder_recommended"] = marts["current_stock_level"] <= marts["reorder_point_units"]
            return marts, "Local Parquet Pipeline (Standby Mode)"
        else:
            st.error("No data found! Please run `python generate_data.py` first.")
            st.stop()


def run_vectorized_monte_carlo(
    n_sims: int,
    horizon_days: int,
    initial_stock: float,
    reorder_point: float,
    reorder_batch: float,
    mu_demand: float,
    std_demand: float,
    mu_lead_time: float,
    std_lead_time: float,
    random_seed: int = 42
):
    """
    High-performance NumPy Monte Carlo Engine.
    Simulates n_sims full inventory trajectories over horizon_days,
    modeling daily stochastic demand and variable supplier replenishment lead times.
    """
    np.random.seed(random_seed)
    
    # Generate daily stochastic demand matrix [n_sims, horizon_days]
    # Truncated at 0 to avoid phantom demand generation
    daily_demands = np.maximum(0.0, np.random.normal(loc=mu_demand, scale=std_demand, size=(n_sims, horizon_days)))
    
    # Pre-allocate inventory tracking array
    inventory_trajectories = np.zeros((n_sims, horizon_days))
    current_inventory = np.full(n_sims, float(initial_stock))
    order_in_transit = np.zeros(n_sims, dtype=bool)
    order_arrival_day = np.full(n_sims, -1, dtype=int)
    
    first_stockout_days = np.full(n_sims, np.nan)
    
    for day in range(horizon_days):
        # 1. Process arriving replenishment shipments
        arrived = (order_arrival_day == day)
        current_inventory[arrived] += reorder_batch
        order_in_transit[arrived] = False
        order_arrival_day[arrived] = -1
        
        # 2. Subtract daily consumer demand
        current_inventory -= daily_demands[:, day]
        inventory_trajectories[:, day] = current_inventory
        
        # Track first stockout occurrence for each trajectory
        hit_stockout_today = (current_inventory <= 0) & np.isnan(first_stockout_days)
        first_stockout_days[hit_stockout_today] = day + 1
        
        # 3. Replenishment policy: place order if stock <= ROP and no order is already in transit
        needs_order = (current_inventory <= reorder_point) & (~order_in_transit)
        if np.any(needs_order):
            num_orders = np.sum(needs_order)
            # Sample lead time from normal distribution with lead time variance
            sampled_leads = np.maximum(
                1, 
                np.round(np.random.normal(loc=mu_lead_time, scale=std_lead_time, size=num_orders))
            ).astype(int)
            
            order_arrival_day[needs_order] = day + sampled_leads
            order_in_transit[needs_order] = True
            
    # Calculate Percentile Risk Bands
    p10 = np.percentile(inventory_trajectories, 10, axis=0)
    p50 = np.percentile(inventory_trajectories, 50, axis=0)
    p90 = np.percentile(inventory_trajectories, 90, axis=0)
    
    # Calculate Risk Metrics
    has_stockout = np.any(inventory_trajectories <= 0, axis=1)
    stockout_prob = np.mean(has_stockout) * 100.0
    
    stockout_runs = first_stockout_days[~np.isnan(first_stockout_days)]
    expected_days_to_stockout = float(np.median(stockout_runs)) if len(stockout_runs) > 0 else None
    
    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "stockout_prob": stockout_prob,
        "expected_days_to_stockout": expected_days_to_stockout,
        "trajectories": inventory_trajectories,
        "daily_demands": daily_demands,
        "min_stocks": np.min(inventory_trajectories, axis=1)
    }


def main():
    # Load dbt marts data
    df_marts, source_desc = load_marts_data()
    
    # Header Banner
    col_header, col_source = st.columns([3, 1])
    with col_header:
        st.title("📦 Modern Data Stack: Supply Chain Analytics")
        st.markdown(
            "Executive Inventory Risk Engine powered by **PostgreSQL**, **dbt (src → stg → marts)**, and **Monte Carlo Simulations**."
        )
    with col_source:
        st.markdown(f"""
        <div style="background:#1e293b; padding:10px 14px; border-radius:8px; border:1px solid #334155; margin-top:12px;">
            <div style="font-size:0.75rem; color:#94a3b8;">DATA CONNECTOR</div>
            <div style="font-size:0.85rem; font-weight:600; color:#38bdf8;">{source_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Sidebar Controls
    st.sidebar.header("🎯 SKU & Simulation Parameters")
    
    # SKU Selector
    sku_options = df_marts["sku_id"].tolist()
    sku_display = {
        row["sku_id"]: f"{row['sku_id']} | {row['category']} - {row['product_name']} ({row['stockout_risk_status']})"
        for _, row in df_marts.iterrows()
    }
    
    selected_sku = st.sidebar.selectbox(
        "Select Target SKU",
        options=sku_options,
        format_func=lambda x: sku_display[x],
        index=0
    )
    
    # Extract SKU profile
    sku_row = df_marts[df_marts["sku_id"] == selected_sku].iloc[0]
    
    st.sidebar.markdown("### 🎲 Monte Carlo Stress Settings")
    
    # Variance Multiplier slider to simulate supplier disruption
    variance_mult = st.sidebar.slider(
        "Lead Time Variance Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.25,
        help="Simulate supply chain shocks (e.g. port congestion, raw material shortages)."
    )
    
    horizon_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=30,
        max_value=120,
        value=60,
        step=15
    )
    
    sim_iterations = st.sidebar.select_slider(
        "Simulation Iterations",
        options=[1000, 5000, 10000, 20000],
        value=10000,
        help="Higher iterations increase precision for tail-risk percentile bounds."
    )
    
    with st.sidebar.expander("⚙️ Manual Inventory Overrides (What-If)"):
        override_stock = st.number_input(
            "Current Stock Level", 
            min_value=0, 
            value=int(sku_row["current_stock_level"])
        )
        override_batch = st.number_input(
            "Reorder Batch Size (Q)", 
            min_value=50, 
            value=int(sku_row["reorder_batch_size"])
        )
        override_rop = st.number_input(
            "Reorder Point (ROP)", 
            min_value=10, 
            value=int(sku_row["reorder_point_units"])
        )
        
    # Baseline Parameters
    mu_demand = float(sku_row["avg_daily_demand"])
    std_demand = float(sku_row["stddev_daily_demand"])
    mu_lead = float(sku_row["avg_lead_time_days"])
    std_lead = float(sku_row["stddev_lead_time_days"]) * variance_mult
    
    # Execute Monte Carlo Simulation
    with st.spinner(f"Running {sim_iterations:,} Monte Carlo inventory iterations with NumPy..."):
        sim_start = time.time()
        sim_results = run_vectorized_monte_carlo(
            n_sims=sim_iterations,
            horizon_days=horizon_days,
            initial_stock=override_stock,
            reorder_point=override_rop,
            reorder_batch=override_batch,
            mu_demand=mu_demand,
            std_demand=std_demand,
            mu_lead_time=mu_lead,
            std_lead_time=std_lead
        )
        sim_runtime = time.time() - sim_start
        
    # Top KPI Cards Ribbon
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Stock Level</div>
            <div class="metric-value">{override_stock:,} <span style="font-size:1rem;color:#94a3b8;">units</span></div>
            <div class="metric-sub">Safety Stock: {sku_row['safety_stock_units']:,} units</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        prob = sim_results["stockout_prob"]
        prob_color = "#f87171" if prob > 20 else ("#fbbf24" if prob > 5 else "#34d399")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stockout Probability</div>
            <div class="metric-value" style="color:{prob_color};">{prob:.1f}%</div>
            <div class="metric-sub">{sim_iterations:,} Monte Carlo runs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        days_to_stockout = sim_results["expected_days_to_stockout"]
        val_str = f"{days_to_stockout:.0f} Days" if days_to_stockout is not None else "No Stockout"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Stockout Day</div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub">Median among depleted runs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Lead Time</div>
            <div class="metric-value">{mu_lead:.1f} ± {std_lead:.1f} <span style="font-size:1rem;color:#94a3b8;">d</span></div>
            <div class="metric-sub">Multiplier: {variance_mult:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        risk_class = sku_row["stockout_risk_status"].lower().replace(" ", "-")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">dbt Risk Status</div>
            <div class="metric-value"><span class="badge-{risk_class}">{sku_row['stockout_risk_status']}</span></div>
            <div class="metric-sub">Days of Supply: {sku_row['days_of_supply']} days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Tabs
    tab_sim, tab_dist, tab_portfolio, tab_lineage = st.tabs([
        "📈 Monte Carlo Risk Bands (p10, p50, p90)",
        "📊 Tail Risk & Minimum Inventory",
        "🌐 Portfolio Stockout Matrix (50 SKUs)",
        "🏗️ Modern Data Stack Architecture"
    ])
    
    with tab_sim:
        st.subheader(f"Projected Inventory Trajectory: {selected_sku} ({sku_row['product_name']})")
        st.caption(
            f"Simulated {sim_iterations:,} independent paths over {horizon_days} days. "
            f"Execution time: {sim_runtime:.3f} seconds. Confidence envelope shows 10th (p10), 50th (p50), and 90th (p90) risk percentiles."
        )
        
        days_axis = np.arange(1, horizon_days + 1)
        p10 = sim_results["p10"]
        p50 = sim_results["p50"]
        p90 = sim_results["p90"]
        
        fig = go.Figure()
        
        # Upper bound: p90 (transparent line for fill)
        fig.add_trace(go.Scatter(
            x=days_axis,
            y=p90,
            mode='lines',
            line=dict(color='rgba(56, 189, 248, 0.3)', width=1),
            name='p90 (Favorable Demand & Lead Time)',
            hoverinfo='x+y'
        ))
        
        # Shaded risk band between p10 and p90
        fig.add_trace(go.Scatter(
            x=days_axis,
            y=p10,
            mode='lines',
            line=dict(color='rgba(244, 63, 94, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(56, 189, 248, 0.12)',
            name='p10 (Stress Scenario / Delays)',
            hoverinfo='x+y'
        ))
        
        # Median trajectory: p50
        fig.add_trace(go.Scatter(
            x=days_axis,
            y=p50,
            mode='lines+markers',
            line=dict(color='#38bdf8', width=3),
            marker=dict(size=3),
            name='p50 (Median Expected Trajectory)'
        ))
        
        # Reference threshold lines
        fig.add_hline(
            y=override_rop,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=2,
            annotation_text=f"Reorder Point (ROP = {override_rop:,})",
            annotation_position="top left",
            annotation_font_color="#f59e0b"
        )
        
        fig.add_hline(
            y=sku_row["safety_stock_units"],
            line_dash="dot",
            line_color="#ec4899",
            line_width=1.5,
            annotation_text=f"Safety Stock ({sku_row['safety_stock_units']:,})",
            annotation_position="bottom left",
            annotation_font_color="#ec4899"
        )
        
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="#ef4444",
            line_width=2,
            annotation_text="STOCKOUT LEVEL (0 Units)",
            annotation_position="bottom right",
            annotation_font_color="#ef4444"
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            height=540,
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode="x unified",
            xaxis=dict(
                title="Simulation Day",
                gridcolor="#334155",
                dtick=10
            ),
            yaxis=dict(
                title="Projected On-Hand Stock (Units)",
                gridcolor="#334155"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation callout
        if prob > 25:
            st.error(
                f"🚨 **High Vulnerability Alert:** Under current lead time volatility ({variance_mult:.1f}x), "
                f"this SKU faces a **{prob:.1f}% probability of stockout** within the {horizon_days}-day horizon. "
                f"Recommendation: Increase Reorder Point by {int(std_demand * std_lead)} units or expedite supplier '{sku_row['supplier_name']}'."
            )
        elif prob > 5:
            st.warning(
                f"⚠️ **Moderate Risk Warning:** Stockout risk is **{prob:.1f}%**. "
                f"Buffer is susceptible to simultaneous demand spikes and shipping delay."
            )
        else:
            st.success(
                f"✅ **Resilient Inventory Level:** Stockout probability is low (**{prob:.1f}%**). "
                f"Safety stock buffer ({sku_row['safety_stock_units']} units) provides adequate dampening."
            )

    with tab_dist:
        st.subheader("Distribution of Minimum Simulated Inventory")
        st.caption("Distribution of the lowest stock point reached across all 10,000 iterations.")
        
        min_stocks = sim_results["min_stocks"]
        stockout_count = np.sum(min_stocks <= 0)
        
        dist_col1, dist_col2 = st.columns([2, 1])
        
        with dist_col1:
            fig_hist = px.histogram(
                x=min_stocks,
                nbins=60,
                color_discrete_sequence=["#38bdf8"],
                labels={"x": "Lowest Stock Level Reached (Units)"},
                title="Simulated Minimum Stock Level Density"
            )
            fig_hist.add_vline(x=0, line_color="#ef4444", line_width=2.5, line_dash="dash", annotation_text="Stockout Boundary (0)")
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                height=420,
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with dist_col2:
            st.markdown("### 📋 Risk Decomposition")
            st.markdown(f"""
            - **Total Iterations:** `{sim_iterations:,}`
            - **Stockout Runs:** `{stockout_count:,}`
            - **Stockout Probability:** `{prob:.2f}%`
            - **Worst Case Stock:** `{int(np.min(min_stocks)):,} units`
            - **Median Lowest Stock:** `{int(np.median(min_stocks)):,} units`
            - **Supplier:** `{sku_row['supplier_name']}`
            - **Shipping Mode:** `{sku_row['shipping_mode']}`
            - **Unit Price:** `${sku_row['unit_price']:.2f}`
            """)
            
    with tab_portfolio:
        st.subheader("Executive Inventory Portfolio Matrix")
        st.caption("All 50 SKUs aggregated from dbt mart table `fct_inventory_metrics`.")
        
        # Filter controls
        f_cat, f_risk = st.columns(2)
        with f_cat:
            cats = ["All Categories"] + sorted(df_marts["category"].unique().tolist())
            selected_cat = st.selectbox("Filter Category", cats)
        with f_risk:
            risks = ["All Risk Tiers"] + sorted(df_marts["stockout_risk_status"].unique().tolist())
            selected_risk = st.selectbox("Filter Risk Tier", risks)
            
        filtered_df = df_marts.copy()
        if selected_cat != "All Categories":
            filtered_df = filtered_df[filtered_df["category"] == selected_cat]
        if selected_risk != "All Risk Tiers":
            filtered_df = filtered_df[filtered_df["stockout_risk_status"] == selected_risk]
            
        # Display table
        st.dataframe(
            filtered_df[[
                "sku_id", "product_name", "category", "stockout_risk_status",
                "current_stock_level", "safety_stock_units", "reorder_point_units",
                "days_of_supply", "avg_daily_demand", "stddev_lead_time_days",
                "supplier_name"
            ]],
            column_config={
                "sku_id": "SKU",
                "product_name": "Product Name",
                "category": "Category",
                "stockout_risk_status": st.column_config.TextColumn("Risk Tier"),
                "current_stock_level": st.column_config.NumberColumn("Current Stock", format="%d"),
                "safety_stock_units": st.column_config.NumberColumn("Safety Stock", format="%d"),
                "reorder_point_units": st.column_config.NumberColumn("ROP", format="%d"),
                "days_of_supply": st.column_config.NumberColumn("Days of Supply", format="%.1f d"),
                "avg_daily_demand": st.column_config.NumberColumn("Avg Demand/Day", format="%.1f"),
                "stddev_lead_time_days": st.column_config.NumberColumn("Lead Volatility (σ)", format="%.1f d"),
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )

    with tab_lineage:
        st.subheader("Modern Data Stack Architecture & dbt Lineage")
        st.markdown(r"""
        ### End-to-End Pipeline Lineage:
        
        ```mermaid
        graph LR
            A[(Postgres: raw.supply_chain_orders)] -->|dbt source| B[src_supply_chain_orders]
            B -->|dedup & null handling| C[stg_inventory_orders]
            C -->|statistical aggregation & safety stock| D[(analytics.fct_inventory_metrics)]
            D -->|10,000-iter Monte Carlo| E[Streamlit Risk Bands]
        ```
        
        #### Layer Specifications:
        1. **Raw Storage (Docker PostgreSQL 16)**:
           - Ingests 100,000 rows generated by `generate_data.py` with Faker & pandas using streaming PostgreSQL `COPY`.
        2. **Layer 1: `src` (Raw Abstraction)**:
           - Declared in `sources.yml`. Wraps `raw.supply_chain_orders` with standardized aliases.
        3. **Layer 2: `stg` (Cleaned Staging)**:
           - Null handling via `COALESCE` on discounts and customer regions.
           - Casing standardization (`TRIM(INITCAP(category))`).
           - Window function deduplication: `ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_date DESC)`.
           - Lead time variance computation: `actual_lead_time_days - promised_lead_time_days`.
        4. **Layer 3: `marts` (Business Inventory Marts)**:
           - Evaluates SKU-level daily demand volatility ($\sigma_D$) and supplier lead time variance ($\sigma_L$).
           - Statistical Safety Stock formula at 95% service level ($Z = 1.65$):
             $$SS = 1.65 \times \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}$$
           - Reorder Point formula:
             $$ROP = (D \times L) + SS$$
           - Data quality tests: `not_null`, `unique`, and `accepted_values`.
        """)


if __name__ == "__main__":
    main()
