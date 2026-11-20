"""
======================================================================
CELERIS AUTOMOBILI INDUSTRIAL - DEMAND FORECASTING & PLANNING CONTROL TOWER
======================================================================
Unified local dashboard reading directly from pipeline outputs:
- Training history & unconstrained demand (train_stage_e.csv)
- 19-month forecast with P10 / P50 / P90 (forecast_stage_g_full.csv)
- Stage B/E/F cleaning, anomaly, and backtest reports + SKU master

Required Views:
1. Executive Overview (history + forecast timeline, mix, 19m forecast)
2. Data Health (issues by type/file, before vs after cleaning)
3. Demand Explorer (family -> SKU -> region -> customer, seasonality, zeros, lifecycle)
4. Anomaly Monitor (flagged points with type, reason, and action)
5. Forecast & Accuracy (forecast vs history, backtest WAPE & bias by family/region/horizon)

Optional Views Included:
6. P90 Procurement Plan & Macro What-If Simulator + Lifecycle Watchlist
======================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Celeris Demand Forecasting Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path(".")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
SUBMISSIONS = ROOT / "outputs" / "submissions"


def resolve_file(primary: Path, *fallbacks: Path):
    for p in (primary, *fallbacks):
        if p.exists():
            return p
    return None


@st.cache_data
def load_all_data():
    train_e_path = resolve_file(DATA_PROC / "train_stage_e.csv", Path("train_stage_e.csv"))
    fc_g_path = resolve_file(DATA_PROC / "forecast_stage_g_full.csv", Path("forecast_stage_g_full.csv"))
    b_sum_path = resolve_file(REPORTS / "stage_b_cleaning_summary.csv", Path("stage_b_cleaning_summary.csv"))
    neg_path = resolve_file(REPORTS / "stage_e_negative_demand_investigation.csv", Path("stage_e_negative_demand_investigation.csv"))
    f_sum_path = resolve_file(REPORTS / "stage_f_backtest_summary.csv", Path("stage_f_backtest_summary.csv"))
    f_fam_path = resolve_file(REPORTS / "stage_f_wape_by_family.csv", Path("stage_f_wape_by_family.csv"))
    f_reg_path = resolve_file(REPORTS / "stage_f_wape_by_region.csv", Path("stage_f_wape_by_region.csv"))
    f_hor_path = resolve_file(REPORTS / "stage_f_wape_by_horizon.csv", Path("stage_f_wape_by_horizon.csv"))
    sku_path = resolve_file(DATA_RAW / "sku_master.csv", Path("student_data/sku_master.csv"))

    train_e = pd.read_csv(train_e_path, low_memory=False) if train_e_path else pd.DataFrame()
    fc_g = pd.read_csv(fc_g_path) if fc_g_path else pd.DataFrame()
    b_sum = pd.read_csv(b_sum_path) if b_sum_path else pd.DataFrame()
    neg_df = pd.read_csv(neg_path, low_memory=False) if neg_path else pd.DataFrame()
    f_sum = pd.read_csv(f_sum_path) if f_sum_path else pd.DataFrame()
    f_fam = pd.read_csv(f_fam_path) if f_fam_path else pd.DataFrame()
    f_reg = pd.read_csv(f_reg_path) if f_reg_path else pd.DataFrame()
    f_hor = pd.read_csv(f_hor_path) if f_hor_path else pd.DataFrame()
    sku_m = pd.read_csv(sku_path) if sku_path else pd.DataFrame()

    return train_e, fc_g, b_sum, neg_df, f_sum, f_fam, f_reg, f_hor, sku_m


train_e, fc_g, b_sum, neg_df, f_sum, f_fam, f_reg, f_hor, sku_m = load_all_data()

if train_e.empty or fc_g.empty:
    st.error("Missing train_stage_e.csv or forecast_stage_g_full.csv. Run stage_e_f_g_pipeline.py first.")
    st.stop()

st.title("Celeris Automobili Industrial — Demand Forecasting & Planning Control Tower")
st.caption("History (2019-01 to 2026-10, unconstrained demand) + 19-Month Forecast (2026-11 to 2028-05)")

# Shared Sidebar Filters
st.sidebar.header("Shared Control Tower Filters")
part = st.sidebar.radio(
    "Control Tower Part",
    ["Combined (History + Forecast)", "Training History (2019-01..2026-10)", "Prediction Window (2026-11..2028-05)"],
)

families = sorted(train_e["Product_Family"].dropna().unique().tolist())
sel_families = st.sidebar.multiselect("Product Family", families, default=families)

regions = sorted(train_e["Region"].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("Region", regions, default=regions)

channels = sorted(train_e["Primary_Channel"].dropna().unique().tolist()) if "Primary_Channel" in train_e.columns else []
sel_channels = st.sidebar.multiselect("Sales Channel", channels, default=channels) if channels else []

customers = sorted(train_e["Primary_Customer"].dropna().unique().tolist()) if "Primary_Customer" in train_e.columns else []
sel_customers = st.sidebar.multiselect("Customer Name", customers, default=customers) if customers else []

sku_pool = sorted(train_e[train_e["Product_Family"].isin(sel_families)]["SKU_ID"].dropna().unique().tolist())
sel_skus = st.sidebar.multiselect("SKU ID (optional filter)", sku_pool, default=[])

# Apply shared filters
tr_f = train_e[train_e["Product_Family"].isin(sel_families) & train_e["Region"].isin(sel_regions)].copy()
fc_f = fc_g[fc_g["Product_Family"].isin(sel_families) & fc_g["Region"].isin(sel_regions)].copy()

if sel_channels and "Primary_Channel" in tr_f.columns:
    tr_f = tr_f[tr_f["Primary_Channel"].isin(sel_channels)]
    if "Primary_Channel" in fc_f.columns:
        fc_f = fc_f[fc_f["Primary_Channel"].isin(sel_channels)]

if sel_customers and "Primary_Customer" in tr_f.columns:
    tr_f = tr_f[tr_f["Primary_Customer"].isin(sel_customers)]
    if "Primary_Customer" in fc_f.columns:
        fc_f = fc_f[fc_f["Primary_Customer"].isin(sel_customers)]

if sel_skus:
    tr_f = tr_f[tr_f["SKU_ID"].isin(sel_skus)]
    fc_f = fc_f[fc_f["SKU_ID"].isin(sel_skus)]

hist_ts = tr_f.groupby("Date", as_index=False)["Unconstrained_Demand"].sum().rename(columns={"Unconstrained_Demand": "Units"})
hist_ts["Timeline_Type"] = "History (Unconstrained)"

fc_ts = fc_f.groupby("Date", as_index=False).agg(
    Units=("Pred_P50", "sum"),
    P10=("Pred_P10", "sum"),
    P90=("Pred_P90", "sum"),
)
fc_ts["Timeline_Type"] = "Forecast (P50)"

tabs = st.tabs([
    "1. Executive Overview",
    "2. Data Health",
    "3. Demand Explorer",
    "4. Anomaly Monitor",
    "5. Forecast & Accuracy",
    "6. P90 Procurement, What-If & Lifecycle",
])

# TAB 1: EXECUTIVE OVERVIEW
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    hist_total = tr_f["Unconstrained_Demand"].sum()
    fc_19m_total = fc_f["Units_Sold_Forecast"].sum() if "Units_Sold_Forecast" in fc_f.columns else fc_f["Pred_P50"].sum()
    p90_19m_total = fc_f["Pred_P90"].sum()
    stockout_uplift = (tr_f["Unconstrained_Demand"] - tr_f["Gross_Units_Sold"]).sum() if "Gross_Units_Sold" in tr_f.columns else 0.0

    c1.metric("Historical Unconstrained Units", f"{hist_total:,.0f}")
    c2.metric("19-Month Forecast (P50)", f"{fc_19m_total:,.0f}")
    c3.metric("19-Month Upper Band (P90)", f"{p90_19m_total:,.0f}")
    c4.metric("Stockout Demand Recovered", f"{stockout_uplift:,.0f}")

    st.subheader("Continuous Timeline: Historical Unconstrained Demand -> 19-Month Forecast")
    if part == "Training History (2019-01..2026-10)":
        comb_ts = hist_ts[["Date", "Units", "Timeline_Type"]].sort_values("Date")
    elif part == "Prediction Window (2026-11..2028-05)":
        comb_ts = fc_ts[["Date", "Units", "Timeline_Type"]].sort_values("Date")
    else:
        comb_ts = pd.concat([
            hist_ts[["Date", "Units", "Timeline_Type"]],
            fc_ts[["Date", "Units", "Timeline_Type"]],
        ], ignore_index=True).sort_values("Date")
    st.line_chart(comb_ts.pivot(index="Date", columns="Timeline_Type", values="Units"), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("19-Month Forecast Mix by Product Family")
        st.bar_chart(fc_f.groupby("Product_Family")["Pred_P50"].sum().sort_values(ascending=False), width="stretch")
    with col_b:
        st.subheader("19-Month Forecast Mix by Region")
        st.bar_chart(fc_f.groupby("Region")["Pred_P50"].sum().sort_values(ascending=False), width="stretch")

# TAB 2: DATA HEALTH
with tabs[1]:
    st.subheader("Cleaning & Data Quality Summary (Stage B / Stage E)")
    if not b_sum.empty:
        st.dataframe(b_sum, width="stretch")
    else:
        st.info("stage_b_cleaning_summary.csv not found in outputs/reports.")

    st.subheader("Filtered Training Dataset Completeness")
    miss_df = pd.DataFrame({
        "Column": tr_f.columns,
        "Missing_Count": tr_f.isna().sum().values,
        "Missing_Pct": (tr_f.isna().mean().values * 100).round(2),
    })
    st.dataframe(miss_df[miss_df["Missing_Count"] > 0], width="stretch")

# TAB 3: DEMAND EXPLORER
with tabs[2]:
    st.subheader("Seasonality & Zero-Demand Profile")
    seas = tr_f.groupby("Month", as_index=False).agg(
        Mean_Demand=("Unconstrained_Demand", "mean"),
        Total_Demand=("Unconstrained_Demand", "sum"),
        Zero_Month_Share=("Unconstrained_Demand", lambda s: (s == 0).mean()),
    )
    st.bar_chart(seas.set_index("Month")["Total_Demand"], width="stretch")
    st.dataframe(seas, width="stretch")

    st.subheader("Customer & Channel Drill-Down (Filtered History)")
    if {"Product_Family", "Region", "Primary_Customer", "Primary_Channel"}.issubset(tr_f.columns):
        cust_view = tr_f.groupby(["Product_Family", "Region", "Primary_Channel", "Primary_Customer"], as_index=False).agg(
            Total_Unconstrained=("Unconstrained_Demand", "sum"),
            Avg_Monthly=("Unconstrained_Demand", "mean"),
            Zero_Months=("Unconstrained_Demand", lambda s: int((s == 0).sum())),
        ).sort_values("Total_Unconstrained", ascending=False).head(50)
        st.dataframe(cust_view, width="stretch")

    st.subheader("Top SKUs by Unconstrained Demand")
    top_skus = tr_f.groupby(["Product_Family", "SKU_ID", "Region"], as_index=False).agg(
        Total_Unconstrained=("Unconstrained_Demand", "sum"),
        Avg_Monthly=("Unconstrained_Demand", "mean"),
        Zero_Months=("Unconstrained_Demand", lambda s: int((s == 0).sum())),
    ).sort_values("Total_Unconstrained", ascending=False).head(50)
    st.dataframe(top_skus, width="stretch")

# TAB 4: ANOMALY MONITOR
with tabs[3]:
    st.subheader("Anomaly Register (Errors vs. Real Events)")
    register_df = pd.DataFrame([
        {"Type": "Data Error", "Source": "train.csv", "Reason": "Duplicate Record_ID rows & impossible date years", "Action": "Kept latest Load_Timestamp; repaired Year/Month agreement"},
        {"Type": "Data Error", "Source": "train.csv", "Reason": "3,544 missing Units_Sold values", "Action": "Recovered 2,545 via Revenue/Price; flagged 999 unresolved"},
        {"Type": "Data Error / Adjustment", "Source": "train.csv", "Reason": "491 negative Units_Sold rows (returns/credits)", "Action": "Excluded from positive gross demand; logged in audit report"},
        {"Type": "Real Supply Constraint", "Source": "inventory_stockouts.csv", "Reason": "5,830 stockout records where shipments < requested units", "Action": "Recovered unconstrained demand via Requested_Units"},
        {"Type": "Real Business Event", "Source": "events_calendar.csv", "Reason": "COVID-19, chip shortage, MEA launch, price hikes, plant shutdowns", "Action": "Encoded time-scoped event & lifecycle flags in Stage E"},
    ])
    st.dataframe(register_df, width="stretch")

    st.subheader("Negative Demand Transactions (Returns / Credits / Adjustments)")
    if not neg_df.empty:
        neg_f = neg_df[neg_df["Product_Family"].isin(sel_families)]
        st.write(f"Showing {len(neg_f):,} negative transaction rows (excluded from positive gross demand; preserved for audit).")
        cols_show = [c for c in ["Record_ID", "Date", "Product_Family", "SKU_ID", "Region", "Sales_Channel", "Customer_Name", "Units_Sold", "Revenue_USD"] if c in neg_f.columns]
        st.dataframe(neg_f[cols_show].head(100), width="stretch")
    else:
        st.info("No negative transaction report found.")

    st.subheader("Structural & Supply Event Flags in Filtered History")
    ev_cols = [c for c in tr_f.columns if c.startswith("Event_")]
    if ev_cols:
        st.dataframe(tr_f[ev_cols].sum().rename("Active_Series_Months"), width="stretch")

# TAB 5: FORECAST & ACCURACY
with tabs[4]:
    st.subheader("Stage F Backtest WAPE & Bias Summary (2025-11 to 2026-10)")
    if not f_sum.empty:
        st.dataframe(f_sum, width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("WAPE by Product Family")
        if not f_fam.empty:
            st.dataframe(f_fam, width="stretch")
            st.bar_chart(f_fam.set_index("Product_Family")["WAPE"], width="stretch")
    with c_right:
        st.subheader("WAPE by Region")
        if not f_reg.empty:
            st.dataframe(f_reg, width="stretch")
            st.bar_chart(f_reg.set_index("Region")["WAPE"], width="stretch")

    st.subheader("WAPE Across Forecast Horizon Months (1 to 12)")
    if not f_hor.empty:
        st.line_chart(f_hor.set_index("Horizon_Month")["WAPE"], width="stretch")

# TAB 6: P90 PROCUREMENT, WHAT-IF & LIFECYCLE
with tabs[5]:
    st.subheader("Macro What-If Scenario Simulator (PMI & EV Adoption Sensitivity)")
    pmi_delta = st.slider("ISM Manufacturing PMI Shock (points)", min_value=-10.0, max_value=10.0, value=-3.0, step=0.5)
    elasticity = st.slider("Demand Sensitivity per PMI Point (%)", min_value=0.0, max_value=3.0, value=0.8, step=0.1)
    adj_factor = max(0.0, 1.0 + (pmi_delta * (elasticity / 100.0)))

    proc_plan = fc_f.groupby(["Date", "Product_Family"], as_index=False).agg(
        P50_Units=("Pred_P50", "sum"),
        P90_Procurement_Units=("Pred_P90", "sum"),
    )
    proc_plan["Scenario_Adjusted_P50"] = (proc_plan["P50_Units"] * adj_factor).round(0)
    st.metric("Scenario Demand Multiplier", f"{adj_factor:.3f}x")
    st.dataframe(proc_plan.head(100), width="stretch")

    if not sku_m.empty:
        st.subheader("Lifecycle Watchlist (EOL Announced, Discontinued & Planned Launches)")
        lc_watch = sku_m[sku_m["Lifecycle_Status"] != "Active"]
        st.dataframe(lc_watch, width="stretch")
