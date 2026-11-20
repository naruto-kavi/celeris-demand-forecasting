"""
======================================================================
CELERIS AUTOMOBILI INDUSTRIAL - STAGES E, F & G COMPLETE PIPELINE
======================================================================
Stage E: Unconstrained demand recovery, SKU lifecycle, events calendar,
         macro lags, and intermittency/cadence feature engineering.
Stage F: Chronological backtesting (WAPE + Bias by Family/Region/Horizon)
         comparing Seasonal Naive & Rolling Mean baselines against a
         LightGBM Ensemble (L1 + Tweedie) + P10/P50/P90 Quantiles.
Stage G: Recursive 19-month forecast (2026-11 to 2028-05), dashboard
         data exports, and validated submission.csv (58,219 rows).
======================================================================
"""

from pathlib import Path
import warnings
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. PATH CONFIGURATION
# ------------------------------------------------------------------
ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "src"
    else Path(".")
)
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
SUBMISSIONS = ROOT / "outputs" / "submissions"

for d in [DATA_PROC, REPORTS, SUBMISSIONS]:
  d.mkdir(parents=True, exist_ok=True)


def resolve_file(primary: Path, *fallbacks: Path) -> Path:
  for p in (primary, *fallbacks):
    if p.exists():
      return p
  raise FileNotFoundError(
      f"Could not locate file in {[str(x) for x in (primary, *fallbacks)]}"
  )


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  denom = np.sum(np.abs(y_true))
  return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 0.0


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  denom = np.sum(np.abs(y_true))
  return float(np.sum(y_pred - y_true) / denom) if denom > 0 else 0.0


# ==================================================================
# STAGE E: UNCONSTRAINED DEMAND & FEATURE ENGINEERING
# ==================================================================
def run_stage_e():
  print("=" * 72)
  print("STAGE E: UNCONSTRAINED DEMAND & DOMAIN FEATURE ENGINEERING")
  print("=" * 72)

  tb = pd.read_csv(
      resolve_file(
          DATA_PROC / "train_stage_b.csv", Path("train_stage_b.csv")
      )
  )
  stage_d = pd.read_csv(
      resolve_file(
          DATA_PROC / "forecasting_dataset_stage_d.csv",
          Path("forecasting_dataset_stage_d.csv"),
      )
  )
  macro = pd.read_csv(
      resolve_file(
          DATA_PROC / "macro_stage_b.csv", Path("macro_stage_b.csv")
      )
  )
  sku_master = pd.read_csv(
      resolve_file(
          DATA_RAW / "sku_master.csv", Path("student_data/sku_master.csv")
      )
  )
  events = pd.read_csv(
      resolve_file(
          DATA_RAW / "events_calendar.csv",
          Path("student_data/events_calendar.csv"),
      )
  )
  test = pd.read_csv(
      resolve_file(DATA_RAW / "test.csv", Path("student_data/test.csv"))
  )

  # 1. Separate negative returns/adjustments from gross sales
  neg_tx = tb[tb["Units_Sold"] < 0].copy()
  neg_tx.to_csv(
      REPORTS / "stage_e_negative_demand_investigation.csv", index=False
  )

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

  # 2. Recover true unconstrained demand during stockouts
  df["Unconstrained_Demand"] = np.where(
      df["Has_Inventory_Record"] == 1,
      np.maximum(df["Gross_Units_Sold"], df["Requested_Units"].fillna(0.0)),
      df["Gross_Units_Sold"],
  )

  # 3. Merge SKU Master lifecycle & powertrain attributes
  sm_cols = [
      "SKU_ID",
      "Size_Class",
      "Powertrain_Application",
      "Launch_Date",
      "EOL_Date",
      "Replaces_SKU",
      "Lifecycle_Status",
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

  macro_cols = [c for c in macro.columns if c not in ["Date", "Year", "Month"]]
  test_df = test_df.merge(macro[["Date"] + macro_cols], on="Date", how="left")

  common_cols = [
      "Date",
      "Year",
      "Month",
      "SKU_ID",
      "Region",
      "Product_Family",
      "Business_Unit",
      "Size_Class",
      "Powertrain_Application",
      "Launch_Date",
      "EOL_Date",
      "Replaces_SKU",
      "Lifecycle_Status",
      "Primary_Customer",
      "Primary_Channel",
      "Unconstrained_Demand",
      "Gross_Units_Sold",
      "Return_Units",
      "Is_Test",
  ] + macro_cols

  full = pd.concat([df[common_cols], test_df[common_cols]], ignore_index=True)
  full["Date_dt"] = pd.to_datetime(full["Date"] + "-01")
  full = full.sort_values(["SKU_ID", "Region", "Date_dt"]).reset_index(
      drop=True
  )

  # 4. Lifecycle rules (Launch_Date, EOL_Date, Ramp-Up, Phase-Out)
  launch_dt = pd.to_datetime(full["Launch_Date"] + "-01")
  eol_dt = pd.to_datetime(full["EOL_Date"] + "-01", errors="coerce")

  full["Months_Since_Launch"] = (full["Date_dt"].dt.year - launch_dt.dt.year) * 12 + (
      full["Date_dt"].dt.month - launch_dt.dt.month
  )
  full["Months_To_EOL"] = np.where(
      eol_dt.notna(),
      (eol_dt.dt.year - full["Date_dt"].dt.year) * 12
      + (eol_dt.dt.month - full["Date_dt"].dt.month),
      999,
  )
  full["Is_Active_Lifecycle"] = (
      (full["Months_Since_Launch"] >= 0) & (full["Months_To_EOL"] >= 0)
  ).astype(int)
  full["Is_Ramp_Up"] = full["Months_Since_Launch"].between(0, 5).astype(int)
  full["Is_Phase_Out"] = full["Months_To_EOL"].between(0, 5).astype(int)
  full["Has_Replacement"] = full["Replaces_SKU"].notna().astype(int)

  # 5. Calendar & Events Calendar indicators
  full["Quarter"] = full["Date_dt"].dt.quarter
  full["Month_Sin"] = np.sin(2 * np.pi * full["Month"] / 12.0)
  full["Month_Cos"] = np.cos(2 * np.pi * full["Month"] / 12.0)
  full["Time_Idx"] = (full["Date_dt"].dt.year - 2019) * 12 + (
      full["Date_dt"].dt.month - 1
  )
  full["Pre_Jan_Price_Hike"] = full["Month"].isin([11, 12]).astype(int)

  events["Start_YM"] = pd.to_datetime(
      events["Start_Date"].str.slice(0, 7) + "-01"
  )
  events["End_YM"] = pd.to_datetime(
      events["End_Date"].fillna("2028-12").str.slice(0, 7) + "-01"
  )
  event_types = [
      "Price Change",
      "Holiday",
      "Disruption",
      "Supply Constraint",
      "Customer Event",
      "Logistics",
  ]
  for et in event_types:
    full[f"Event_{et.replace(' ', '_')}"] = 0

  for _, ev in events.iterrows():
    if ev["Event_Type"] not in event_types:
      continue
    col_name = f"Event_{ev['Event_Type'].replace(' ', '_')}"
    mask = (full["Date_dt"] >= ev["Start_YM"]) & (
        full["Date_dt"] <= ev["End_YM"]
    )
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
        mask &= (full["Primary_Customer"] == parts[0]) & (
            full["Region"] == parts[1]
        )
      else:
        mask &= full["Primary_Customer"] == parts[0]
    full.loc[mask, col_name] = 1

  # 6. Macro lags & EV adoption interactions
  lead_macros = [
      "ISM_Manufacturing_PMI",
      "New_Vehicle_Orders_Index",
      "Auto_Production_YoY_Pct",
      "Vehicle_Inventory_Days",
      "EV_Adoption_Index",
  ]
  for mc in lead_macros:
    for lag in [1, 3, 6]:
      full[f"{mc}_Lag_{lag}"] = full.groupby(["SKU_ID", "Region"], sort=False)[
          mc
      ].shift(lag)

  full["EV_x_Adoption"] = (
      full["Powertrain_Application"] == "EV"
  ).astype(int) * full["EV_Adoption_Index"]
  full["ICE_x_Adoption"] = (
      full["Powertrain_Application"] == "ICE"
  ).astype(int) * full["EV_Adoption_Index"]

  # 7. Predecessor (Replaces_SKU) demand transfer for Planned Launches
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

  train_e = full[full["Is_Test"] == 0].reset_index(drop=True)
  test_e = full[full["Is_Test"] == 1].reset_index(drop=True)

  train_e.drop(columns=["Date_dt", "Is_Test"]).to_csv(
      DATA_PROC / "train_stage_e.csv", index=False
  )
  test_e.drop(columns=["Date_dt", "Is_Test"]).to_csv(
      DATA_PROC / "test_stage_e.csv", index=False
  )
  print(
      f"Saved train_stage_e.csv ({train_e.shape}) and test_stage_e.csv ({test_e.shape})"
  )
  return train_e, test_e


# ==================================================================
# HELPER: FAST CAUSAL LAG & RECURSIVE ROLLOUT ENGINE
# ==================================================================
LAG_COLS = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Lag_24",
    "Roll_Mean_3",
    "Roll_Mean_6",
    "Roll_Mean_12",
    "Roll_Std_3",
    "Zero_Rate_12",
    "YoY_Diff_12_24",
]

CAT_COLS = [
    "SKU_ID",
    "Region",
    "Product_Family",
    "Business_Unit",
    "Size_Class",
    "Powertrain_Application",
    "Primary_Customer",
    "Primary_Channel",
]


def attach_training_lags(df_in: pd.DataFrame) -> pd.DataFrame:
  df_out = df_in.sort_values(["SKU_ID", "Region", "Date_dt"]).copy()
  g = df_out.groupby(["SKU_ID", "Region"], sort=False)["Unconstrained_Demand"]

  for lag in [1, 2, 3, 6, 12, 24]:
    df_out[f"Lag_{lag}"] = g.shift(lag)

  s1 = g.shift(1)
  df_out["Roll_Mean_3"] = (
      s1.groupby([df_out["SKU_ID"], df_out["Region"]], sort=False)
      .rolling(3, min_periods=1)
      .mean()
      .reset_index(level=[0, 1], drop=True)
  )
  df_out["Roll_Mean_6"] = (
      s1.groupby([df_out["SKU_ID"], df_out["Region"]], sort=False)
      .rolling(6, min_periods=1)
      .mean()
      .reset_index(level=[0, 1], drop=True)
  )
  df_out["Roll_Mean_12"] = (
      s1.groupby([df_out["SKU_ID"], df_out["Region"]], sort=False)
      .rolling(12, min_periods=1)
      .mean()
      .reset_index(level=[0, 1], drop=True)
  )
  df_out["Roll_Std_3"] = (
      s1.groupby([df_out["SKU_ID"], df_out["Region"]], sort=False)
      .rolling(3, min_periods=2)
      .std()
      .reset_index(level=[0, 1], drop=True)
  )
  df_out["Zero_Rate_12"] = (
      (s1 == 0)
      .groupby([df_out["SKU_ID"], df_out["Region"]], sort=False)
      .rolling(12, min_periods=1)
      .mean()
      .reset_index(level=[0, 1], drop=True)
  )
  df_out["YoY_Diff_12_24"] = df_out["Lag_12"] - df_out["Lag_24"]

  for c in [
      "Lag_1",
      "Lag_2",
      "Lag_3",
      "Lag_6",
      "Lag_12",
      "Roll_Mean_3",
      "Roll_Mean_6",
      "Roll_Mean_12",
  ]:
    df_out[c] = df_out[c].fillna(df_out["Predecessor_Recent_Mean"])
  return df_out


def train_model_suite(tr_df: pd.DataFrame, feature_cols: list):
  tr_fit = tr_df[tr_df["Time_Idx"] >= 12].copy()
  for cc in CAT_COLS:
    tr_fit[cc] = tr_fit[cc].astype("category")

  m_l1 = LGBMRegressor(
      objective="regression_l1",
      n_estimators=400,
      learning_rate=0.05,
      num_leaves=63,
      min_child_samples=30,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      n_jobs=-1,
      verbose=-1,
  )
  m_tw = LGBMRegressor(
      objective="tweedie",
      tweedie_variance_power=1.2,
      n_estimators=400,
      learning_rate=0.05,
      num_leaves=63,
      min_child_samples=30,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      n_jobs=-1,
      verbose=-1,
  )
  m_p10 = LGBMRegressor(
      objective="quantile",
      alpha=0.10,
      n_estimators=300,
      learning_rate=0.05,
      num_leaves=63,
      min_child_samples=30,
      random_state=42,
      n_jobs=-1,
      verbose=-1,
  )
  m_p90 = LGBMRegressor(
      objective="quantile",
      alpha=0.90,
      n_estimators=300,
      learning_rate=0.05,
      num_leaves=63,
      min_child_samples=30,
      random_state=42,
      n_jobs=-1,
      verbose=-1,
  )

  m_l1.fit(tr_fit[feature_cols], tr_fit["Unconstrained_Demand"])
  m_tw.fit(tr_fit[feature_cols], tr_fit["Unconstrained_Demand"])
  m_p10.fit(tr_fit[feature_cols], tr_fit["Unconstrained_Demand"])
  m_p90.fit(tr_fit[feature_cols], tr_fit["Unconstrained_Demand"])
  return m_l1, m_tw, m_p10, m_p90


def recursive_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    models: tuple,
    feature_cols: list,
) -> pd.DataFrame:
  m_l1, m_tw, m_p10, m_p90 = models
  future_start = future_df["Date_dt"].min()

  hist_tail = history_df[
      history_df["Date_dt"] >= (future_start - pd.DateOffset(months=24))
  ][["Date_dt", "SKU_ID", "Region", "Unconstrained_Demand"]].copy()

  fut_shell = future_df[["Date_dt", "SKU_ID", "Region"]].copy()
  fut_shell["Unconstrained_Demand"] = np.nan

  combined = pd.concat([hist_tail, fut_shell], ignore_index=True)
  piv = combined.pivot(
      index="Date_dt", columns=["SKU_ID", "Region"], values="Unconstrained_Demand"
  ).sort_index()

  preds_list = []
  for dt in sorted(future_df["Date_dt"].unique()):
    pos = piv.index.get_loc(dt)
    step_lags = pd.DataFrame({
        "Lag_1": piv.iloc[pos - 1],
        "Lag_2": piv.iloc[pos - 2],
        "Lag_3": piv.iloc[pos - 3],
        "Lag_6": piv.iloc[pos - 6],
        "Lag_12": piv.iloc[pos - 12],
        "Lag_24": piv.iloc[pos - 24] if pos >= 24 else np.nan,
        "Roll_Mean_3": piv.iloc[max(0, pos - 3) : pos].mean(),
        "Roll_Mean_6": piv.iloc[max(0, pos - 6) : pos].mean(),
        "Roll_Mean_12": piv.iloc[max(0, pos - 12) : pos].mean(),
        "Roll_Std_3": piv.iloc[max(0, pos - 3) : pos].std(),
        "Zero_Rate_12": (piv.iloc[max(0, pos - 12) : pos] == 0).mean(),
    }).reset_index()
    step_lags["YoY_Diff_12_24"] = step_lags["Lag_12"] - step_lags["Lag_24"]

    drop_existing = [c for c in LAG_COLS if c in future_df.columns]
    step_rows = (
        future_df[future_df["Date_dt"] == dt]
        .drop(columns=drop_existing)
        .merge(step_lags, on=["SKU_ID", "Region"], how="left")
    )

    for c in [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_6",
        "Lag_12",
        "Roll_Mean_3",
        "Roll_Mean_6",
        "Roll_Mean_12",
    ]:
      step_rows[c] = step_rows[c].fillna(step_rows["Predecessor_Recent_Mean"])

    for cc in CAT_COLS:
      step_rows[cc] = step_rows[cc].astype("category")

    p1 = np.clip(m_l1.predict(step_rows[feature_cols]), 0, None)
    p2 = np.clip(m_tw.predict(step_rows[feature_cols]), 0, None)
    p50 = 0.55 * p1 + 0.45 * p2
    p10 = np.clip(m_p10.predict(step_rows[feature_cols]), 0, None)
    p90 = np.maximum(p50, np.clip(m_p90.predict(step_rows[feature_cols]), 0, None))

    active = step_rows["Is_Active_Lifecycle"].values == 1
    p50 = np.where(active, p50, 0.0)
    p10 = np.where(active, np.minimum(p10, p50), 0.0)
    p90 = np.where(active, p90, 0.0)

    piv.loc[
        dt, pd.MultiIndex.from_frame(step_rows[["SKU_ID", "Region"]])
    ] = p50

    out_step = step_rows[[
        "Date",
        "Date_dt",
        "SKU_ID",
        "Region",
        "Product_Family",
        "Business_Unit",
        "Primary_Customer",
        "Primary_Channel",
    ]].copy()
    out_step["Pred_P10"] = p10
    out_step["Pred_P50"] = p50
    out_step["Pred_P90"] = p90
    preds_list.append(out_step)

  return pd.concat(preds_list, ignore_index=True)


# ==================================================================
# STAGE F: BACKTESTING & WAPE EVALUATION
# ==================================================================
def run_stage_f(train_e: pd.DataFrame):
  print("\n" + "=" * 72)
  print("STAGE F: ROLLING-ORIGIN MULTI-STEP BACKTESTING (WAPE & BIAS)")
  print("=" * 72)

  hist = attach_training_lags(train_e)
  exclude_cols = {
      "Date",
      "Date_dt",
      "Launch_Date",
      "EOL_Date",
      "Replaces_SKU",
      "Lifecycle_Status",
      "Unconstrained_Demand",
      "Gross_Units_Sold",
      "Return_Units",
      "Is_Test",
  }
  feature_cols = [c for c in hist.columns if c not in exclude_cols]

  val_start = pd.Timestamp("2025-11-01")
  tr = hist[hist["Date_dt"] < val_start].copy()
  va = hist[hist["Date_dt"] >= val_start].copy()

  b1 = va["Lag_12"].fillna(0).clip(lower=0).values
  c6 = (
      tr[tr["Date_dt"] >= "2025-05-01"]
      .groupby(["SKU_ID", "Region"])["Unconstrained_Demand"]
      .mean()
      .rename("Cutoff_6M")
  )
  b2 = (
      va.merge(c6, on=["SKU_ID", "Region"], how="left")["Cutoff_6M"]
      .fillna(0)
      .clip(lower=0)
      .values
  )

  models = train_model_suite(tr, feature_cols)
  va_preds = recursive_forecast(tr, va, models, feature_cols)

  eval_df = va[[
      "Date",
      "Date_dt",
      "SKU_ID",
      "Region",
      "Product_Family",
      "Unconstrained_Demand",
  ]].merge(
      va_preds[["Date", "SKU_ID", "Region", "Pred_P10", "Pred_P50", "Pred_P90"]],
      on=["Date", "SKU_ID", "Region"],
      how="left",
  )
  eval_df["Baseline_Lag12"] = b1
  eval_df["Baseline_Roll6M"] = b2
  eval_df["Horizon_Month"] = (
      (eval_df["Date_dt"].dt.year - val_start.year) * 12
      + (eval_df["Date_dt"].dt.month - val_start.month)
      + 1
  )

  y_true = eval_df["Unconstrained_Demand"].values
  summary = pd.DataFrame([
      {
          "Model": "Baseline 1: Seasonal Naive (Lag_12)",
          "WAPE": wape(y_true, eval_df["Baseline_Lag12"].values),
          "Bias": forecast_bias(y_true, eval_df["Baseline_Lag12"].values),
      },
      {
          "Model": "Baseline 2: Cutoff 6-Month Mean",
          "WAPE": wape(y_true, eval_df["Baseline_Roll6M"].values),
          "Bias": forecast_bias(y_true, eval_df["Baseline_Roll6M"].values),
      },
      {
          "Model": "Global LightGBM Ensemble (L1 + Tweedie Recursive)",
          "WAPE": wape(y_true, eval_df["Pred_P50"].values),
          "Bias": forecast_bias(y_true, eval_df["Pred_P50"].values),
      },
  ])
  print("\nOverall 12-Month Multi-Step Backtest Results:")
  print(summary.to_string(index=False))

  by_family = (
      eval_df.groupby("Product_Family", as_index=False)
      .apply(
          lambda g: pd.Series({
              "WAPE": wape(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
              "Bias": forecast_bias(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
          })
      )
      .reset_index(drop=True)
  )
  by_region = (
      eval_df.groupby("Region", as_index=False)
      .apply(
          lambda g: pd.Series({
              "WAPE": wape(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
              "Bias": forecast_bias(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
          })
      )
      .reset_index(drop=True)
  )
  by_horizon = (
      eval_df.groupby("Horizon_Month", as_index=False)
      .apply(
          lambda g: pd.Series({
              "WAPE": wape(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
              "Bias": forecast_bias(
                  g["Unconstrained_Demand"].values, g["Pred_P50"].values
              ),
          })
      )
      .reset_index(drop=True)
  )

  summary.to_csv(REPORTS / "stage_f_backtest_summary.csv", index=False)
  by_family.to_csv(REPORTS / "stage_f_wape_by_family.csv", index=False)
  by_region.to_csv(REPORTS / "stage_f_wape_by_region.csv", index=False)
  by_horizon.to_csv(REPORTS / "stage_f_wape_by_horizon.csv", index=False)
  return feature_cols


# ==================================================================
# STAGE G: FULL RETRAIN & 19-MONTH FORECAST SUBMISSION (2026-11..2028-05)
# ==================================================================
def run_stage_g(
    train_e: pd.DataFrame, test_e: pd.DataFrame, feature_cols: list
):
  print("\n" + "=" * 72)
  print("STAGE G: FINAL 19-MONTH FORECAST & SUBMISSION GENERATION")
  print("=" * 72)

  full_hist = attach_training_lags(train_e)
  full_models = train_model_suite(full_hist, feature_cols)
  forecast_df = recursive_forecast(full_hist, test_e, full_models, feature_cols)

  forecast_df["Units_Sold_Forecast"] = (
      np.rint(np.clip(forecast_df["Pred_P50"], 0, None)).astype(int)
  )
  forecast_df["Forecast_P10_Int"] = (
      np.rint(np.clip(forecast_df["Pred_P10"], 0, None)).astype(int)
  )
  forecast_df["Forecast_P90_Int"] = (
      np.rint(np.clip(forecast_df["Pred_P90"], 0, None)).astype(int)
  )
  forecast_df.drop(columns=["Date_dt"]).to_csv(
      DATA_PROC / "forecast_stage_g_full.csv", index=False
  )

  raw_test = pd.read_csv(
      resolve_file(DATA_RAW / "test.csv", Path("student_data/test.csv"))
  )
  sub = raw_test[["Date", "SKU_ID", "Region"]].merge(
      forecast_df[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]],
      on=["Date", "SKU_ID", "Region"],
      how="left",
      validate="one_to_one",
  )

  assert len(sub) == 58219, f"Expected 58,219 rows, got {len(sub)}"
  assert (
      sub["Units_Sold_Forecast"].isna().sum() == 0
  ), "Submission contains missing values!"
  assert (
      sub["Units_Sold_Forecast"] >= 0
  ).all(), "Submission contains negative values!"
  assert pd.api.types.is_integer_dtype(
      sub["Units_Sold_Forecast"]
  ), "Forecasts must be integers!"

  sub_path = SUBMISSIONS / "submission.csv"
  sub.to_csv(sub_path, index=False)
  print(
      f"Submission verified & saved -> {sub_path} | Rows: {len(sub)} | Total Forecast Units: {sub['Units_Sold_Forecast'].sum():,}"
  )


if __name__ == "__main__":
  train_e_df, test_e_df = run_stage_e()
  feat_cols = run_stage_f(train_e_df)
  run_stage_g(train_e_df, test_e_df, feat_cols)