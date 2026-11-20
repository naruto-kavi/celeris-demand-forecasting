"""
CELERIS - STAGE E: UNCONSTRAINED DEMAND & ADVANCED FEATURE ENGINEERING
Builds leak-free training and test feature datasets incorporating:
- Stockout-recovered Unconstrained_Demand target (Requested_Units during stockouts)
- Negative-transaction separation (gross demand vs returns/adjustments)
- SKU Master lifecycle rules (Launch_Date, EOL_Date, Replaces_SKU demand transfer, EV/ICE)
- Events Calendar mapping (holidays, disruptions, price hikes, customer events through 2028-05)
- Macro indicator lags & powertrain interactions
- Ordering-cycle & intermittency descriptors
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "src") else Path(".")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

DATA_PROC.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def resolve_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def build_stage_e():
    tb = pd.read_csv(resolve_path(DATA_PROC / "train_stage_b.csv", Path("train_stage_b.csv")))
    stage_d = pd.read_csv(resolve_path(DATA_PROC / "forecasting_dataset_stage_d.csv", Path("forecasting_dataset_stage_d.csv")))
    macro = pd.read_csv(resolve_path(DATA_PROC / "macro_stage_b.csv", Path("macro_stage_b.csv")))
    sku_master = pd.read_csv(resolve_path(DATA_RAW / "sku_master.csv", Path("student_data/sku_master.csv")))
    events = pd.read_csv(resolve_path(DATA_RAW / "events_calendar.csv", Path("student_data/events_calendar.csv")))
    test = pd.read_csv(resolve_path(DATA_RAW / "test.csv", Path("student_data/test.csv")))

    # 1. Separate positive gross demand from negative returns/adjustments
    neg_tx = tb[tb["Units_Sold"] < 0].copy()
    neg_tx.to_csv(REPORTS / "stage_e_negative_demand_investigation.csv", index=False)

    tb_pos = (
        tb[tb["Units_Sold"] >= 0]
        .groupby(["Date", "SKU_ID", "Region"], as_index=False)
        .agg(
            Gross_Units_Sold=("Units_Sold", "sum"),
            Primary_Customer=("Customer_Name", "first"),
            Primary_Channel=("Sales_Channel", "first"),
        )
    )
    tb_neg = (
        tb[tb["Units_Sold"] < 0]
        .groupby(["Date", "SKU_ID", "Region"], as_index=False)
        .agg(Return_Units=("Units_Sold", lambda s: s.abs().sum()))
    )

    df = stage_d.merge(tb_pos, on=["Date", "SKU_ID", "Region"], how="left")
    df = df.merge(tb_neg, on=["Date", "SKU_ID", "Region"], how="left")
    df["Gross_Units_Sold"] = df["Gross_Units_Sold"].fillna(0.0).clip(lower=0.0)
    df["Return_Units"] = df["Return_Units"].fillna(0.0)

    # 2. Recover true Unconstrained_Demand during stockout months
    df["Unconstrained_Demand"] = np.where(
        df["Has_Inventory_Record"] == 1,
        np.maximum(df["Gross_Units_Sold"], df["Requested_Units"].fillna(0.0)),
        df["Gross_Units_Sold"],
    )

    # 3. Merge SKU Master lifecycle & powertrain metadata
    sm_cols = [
        "SKU_ID", "Size_Class", "Powertrain_Application",
        "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"
    ]
    df = df.merge(sku_master[sm_cols], on="SKU_ID", how="left")
    test_df = test.merge(sku_master[sm_cols], on="SKU_ID", how="left")

    test_df["Unconstrained_Demand"] = np.nan
    test_df["Gross_Units_Sold"] = np.nan
    test_df["Return_Units"] = 0.0
    test_df["Primary_Customer"] = test_df["Customer_Name"]
    test_df["Primary_Channel"] = test_df["Sales_Channel"]
    test_df["Is_Test"] = 1
    df["Is_Test"] = 0

    # Merge macro indicators onto test_df (macro extends through 2028-05)
    macro_cols = [c for c in macro.columns if c not in ["Date", "Year", "Month"]]
    test_df = test_df.merge(macro[["Date"] + macro_cols], on="Date", how="left")

    common_cols = [
        "Date", "Year", "Month", "SKU_ID", "Region", "Product_Family", "Business_Unit",
        "Size_Class", "Powertrain_Application", "Launch_Date", "EOL_Date",
        "Replaces_SKU", "Lifecycle_Status", "Primary_Customer", "Primary_Channel",
        "Unconstrained_Demand", "Gross_Units_Sold", "Return_Units", "Is_Test"
    ] + macro_cols

    full = pd.concat([df[common_cols], test_df[common_cols]], ignore_index=True)
    full["Date_dt"] = pd.to_datetime(full["Date"] + "-01")
    full = full.sort_values(["SKU_ID", "Region", "Date_dt"]).reset_index(drop=True)

    # 4. Lifecycle features & Business Rule #5 enforcement flags
    launch_dt = pd.to_datetime(full["Launch_Date"] + "-01")
    eol_dt = pd.to_datetime(full["EOL_Date"] + "-01", errors="coerce")

    full["Months_Since_Launch"] = (full["Date_dt"].dt.year - launch_dt.dt.year) * 12 + (full["Date_dt"].dt.month - launch_dt.dt.month)
    full["Months_To_EOL"] = np.where(
        eol_dt.notna(),
        (eol_dt.dt.year - full["Date_dt"].dt.year) * 12 + (eol_dt.dt.month - full["Date_dt"].dt.month),
        999,
    )
    full["Is_Active_Lifecycle"] = ((full["Months_Since_Launch"] >= 0) & (full["Months_To_EOL"] >= 0)).astype(int)
    full["Is_Ramp_Up"] = full["Months_Since_Launch"].between(0, 5).astype(int)
    full["Is_Phase_Out"] = full["Months_To_EOL"].between(0, 5).astype(int)
    full["Has_Replacement"] = full["Replaces_SKU"].notna().astype(int)

    # 5. Calendar & Events Calendar features (including 2026-11..2028-05 events)
    full["Quarter"] = full["Date_dt"].dt.quarter
    full["Month_Sin"] = np.sin(2 * np.pi * full["Month"] / 12.0)
    full["Month_Cos"] = np.cos(2 * np.pi * full["Month"] / 12.0)
    full["Time_Idx"] = (full["Date_dt"].dt.year - 2019) * 12 + (full["Date_dt"].dt.month - 1)
    full["Pre_Jan_Price_Hike"] = full["Month"].isin([11, 12]).astype(int)

    events["Start_YM"] = pd.to_datetime(events["Start_Date"].str.slice(0, 7) + "-01")
    events["End_YM"] = pd.to_datetime(events["End_Date"].fillna("2028-12").str.slice(0, 7) + "-01")
    event_types = ["Price Change", "Holiday", "Disruption", "Supply Constraint", "Customer Event", "Logistics"]
    for et in event_types:
        full[f"Event_{et.replace(' ', '_')}"] = 0

    for _, ev in events.iterrows():
        if ev["Event_Type"] not in event_types:
            continue
        col_name = f"Event_{ev['Event_Type'].replace(' ', '_')}"
        mask = (full["Date_dt"] >= ev["Start_YM"]) & (full["Date_dt"] <= ev["End_YM"])
        stype, sval = ev["Scope_Type"], str(ev["Scope_Value"])
        if stype == "Region":
            mask &= full["Region"] == sval
        elif stype == "Product_Family":
            mask &= full["Product_Family"] == sval
        elif stype == "Channel":
            mask &= full["Primary_Channel"] == sval
        elif stype == "Customer":
            parts = [p.strip() for p in sval.split("|")]
            if len(parts) == 2:
                mask &= (full["Primary_Customer"] == parts[0]) & (full["Region"] == parts[1])
            else:
                mask &= full["Primary_Customer"] == parts[0]
        full.loc[mask, col_name] = 1

    # 6. Macro lags & EV transition interaction features
    lead_macros = [
        "ISM_Manufacturing_PMI", "New_Vehicle_Orders_Index",
        "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days", "EV_Adoption_Index"
    ]
    for mc in lead_macros:
        for lag in [1, 3, 6]:
            full[f"{mc}_Lag_{lag}"] = full.groupby(["SKU_ID", "Region"], sort=False)[mc].shift(lag)

    full["EV_x_Adoption"] = (full["Powertrain_Application"] == "EV").astype(int) * full["EV_Adoption_Index"]
    full["ICE_x_Adoption"] = (full["Powertrain_Application"] == "ICE").astype(int) * full["EV_Adoption_Index"]

    # 7. Predecessor demand transfer for Planned Launches (Replaces_SKU)
    train_hist = full[full["Is_Test"] == 0].copy()
    recent_12m = (
        train_hist[train_hist["Date_dt"] >= "2025-11-01"]
        .groupby(["SKU_ID", "Region"])["Unconstrained_Demand"]
        .mean()
        .to_dict()
    )
    full["Predecessor_Recent_Mean"] = [
        recent_12m.get((rep, reg), np.nan) if pd.notna(rep) else np.nan
        for rep, reg in zip(full["Replaces_SKU"], full["Region"])
    ]

    # Save Stage E datasets
    train_e = full[full["Is_Test"] == 0].drop(columns=["Date_dt", "Is_Test"]).reset_index(drop=True)
    test_e = full[full["Is_Test"] == 1].drop(columns=["Date_dt", "Is_Test"]).reset_index(drop=True)

    train_e.to_csv(DATA_PROC / "train_stage_e.csv", index=False)
    test_e.to_csv(DATA_PROC / "test_stage_e.csv", index=False)
    print(f"Stage E complete -> train_stage_e.csv: {train_e.shape}, test_stage_e.csv: {test_e.shape}")
    return train_e, test_e


if __name__ == "__main__":
    build_stage_e()
