# Celeris Demand Forecasting & Planning Control Tower

End-to-end data cleaning, unconstrained demand estimation, multi-step backtesting, 19-month forecasting (`2026-11` to `2028-05`), and an interactive Streamlit planning control tower for **Celeris Automobili Industrial**.

## Overview

Celeris Automobili Industrial supplies automotive and industrial components across multiple product families, sales channels, customers, and regions. Historical shipment data (`2019-01` to `2026-10`) contains ERP data quality issues as well as real supply constraints and business events (such as COVID-19 disruptions, supply constraints, price adjustments, regional expansion, and the shift to electric vehicles).

This project delivers:
- **Cleaned Data & Audit Reports (Stages A–D):** Standardized dates, SKU IDs, regions, and channels; resolved duplicate `Record_ID` entries by latest `Load_Timestamp`; recovered missing product attributes and whole-number revenue/price unit records; reconciled macro and inventory stockout records; and preserved full audit reports.
- **Unconstrained Demand & Domain Features (Stage E):** Separates negative return/credit transactions (`491` rows, `-26,439` units) for audit, recovers unconstrained demand during stockout months via `Requested_Units` while preserving actual shipments, and engineers lifecycle (`Launch_Date`, `EOL_Date`, `Replaces_SKU`, `Powertrain_Application`), `events_calendar.csv`, and lagged macroeconomic features.
- **Backtesting & 19-Month Forecasting (Stages F–G):** Evaluates a recursive multi-step **LightGBM Ensemble (`L1 + Tweedie`)** against **Seasonal Naive (`Lag_12`)** and **Cutoff 6-Month Mean** baselines on **WAPE** and **Bias**, generating `P10` / `P50` / `P90` bands and a validated `58,219`-row forecast (`2,502,206` total units) in `submission.csv`.
- **Planning Control Tower (`app.py`):** A unified local and cloud Streamlit app reading directly from pipeline outputs with shared sidebar filters (`Product Family`, `Region`, `Sales Channel`, `Customer Name`, `SKU ID`).

## Key Backtest & Submission Highlights

| Model / Artifact | Metric / Details | Value |
| :--- | :--- | :--- |
| **Baseline 1: Seasonal Naive (`Lag_12`)** | 12-Month Multi-Step Backtest WAPE / Bias | `0.5018` WAPE (`+0.0614` Bias) |
| **Baseline 2: Cutoff 6-Month Mean** | 12-Month Multi-Step Backtest WAPE / Bias | `0.6070` WAPE (`+0.0303` Bias) |
| **Global LightGBM Ensemble (`L1 + Tweedie`)** | 12-Month Multi-Step Backtest WAPE / Bias | **`0.3858` WAPE** (`-0.1459` Bias) |
| **Final Test Forecast (`submission.csv`)** | Rows (`2026-11` to `2028-05`) / Total Units | `58,219` rows / `2,502,206` units (non-negative `int64`, `0` nulls) |

## Dashboard Views (`app.py`)

1. **Executive Overview:** Continuous timeline from historical unconstrained demand (`2019-01`–`2026-10`) into the 19-month forecast (`2026-11`–`2028-05`), headline P50/P90/stockout-recovery metrics, and mix by product family and region.
2. **Data Health:** Stage B cleaning summary (`244,566` training rows preserved, `2,545` unit records recovered from revenue/price, `113` macro rows) and dataset missingness checks.
3. **Demand Explorer:** Monthly seasonality, zero-month share, customer/channel breakdown, and top SKU-region demand tables.
4. **Anomaly Monitor:** Anomaly register distinguishing data errors from real supply/business events, negative demand transactions, and structural/supply event flags.
5. **Forecast & Accuracy:** Backtest WAPE and bias overall, by product family, by region, and across forecast horizons.
6. **P90 Procurement, What-If & Lifecycle:** Interactive `ISM_Manufacturing_PMI` shock simulator, `P90` procurement quantities, and EOL / discontinued / planned-launch watchlist.


## Live Demo

Application link: https://celeris-demand-forecasting.streamlit.app/


## 2. Backtest Benchmarks & Accuracy Breakdown (Stage F)

Evaluated on a 12-month recursive multi-step holdout window (`2025-11` to `2026-10`):

### Overall Model Comparison
| Model | 12-Month Recursive WAPE | Forecast Bias | Improvement vs. Seasonal Naive |
| :--- | :---: | :---: | :---: |
| **Baseline 1: Seasonal Naive (`Lag_12`)** | `0.5018` (`50.18%`) | `+0.0614` | — |
| **Baseline 2: Cutoff 6-Month Mean** | `0.6070` (`60.70%`) | `+0.0303` | `-10.52 pp` |
| **Global LightGBM Ensemble (`L1 + Tweedie`)** | **`0.3858` (`38.58%`)** | **`-0.1459`** | **`+11.61 pp` (23.1% relative error reduction)** |

### WAPE & Bias by Product Family and Region
| Product Family | Backtest WAPE | Bias | Region | Backtest WAPE | Bias |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Chassis & Structural** | `0.3671` | `-0.1073` | **North America** | `0.3472` | `-0.1099` |
| **Powertrain Components** | `0.3729` | `-0.1711` | **Europe** | `0.3640` | `-0.1432` |
| **Precision Machined Parts** | `0.3831` | `-0.1342` | **Asia Pacific** | `0.3853` | `-0.1702` |
| **Thermal Management** | `0.3846` | `-0.1436` | **India** | `0.3990` | `-0.1598` |
| **Brake & Safety Systems** | `0.3870` | `-0.1494` | **Middle East & Africa** | `0.4185` | `-0.1097` |
| **Hydraulic Systems** | `0.3876` | `-0.1457` | **Latin America** | `0.4207` | `-0.1663` |
| **Electrical Assemblies** | `0.4296` | `-0.1838` | — | — | — |

### Submission Verification (`outputs/submissions/submission.csv`)
- **Total Forecast Rows:** `58,219` (exact 1:1 key alignment with `test.csv` across `2026-11` to `2028-05`)
- **Total 19-Month Forecast (`P50`):** `2,502,206` units (`P90` upper planning band: `3,643,765` units)
- **Data Integrity Checks:** `0` missing values, `0` negative values, `0` duplicate `(Date, SKU_ID, Region)` keys, `int64` dtype

---

## 3. Data Cleaning & Anomaly Handling Summary

| Stage | Issue / Signal Identified | Resolution & Audit Artifact |
| :--- | :--- | :--- |
| **Stage A** | Duplicate `Record_ID` loads & impossible date years | Retained latest `Load_Timestamp`; repaired dates where `Year` and `Month` agreed (`outputs/reports/stage_a_summary.csv`) |
| **Stage B** | Missing product attributes, region aliases, & `3,544` missing `Units_Sold` | Standardized regions/SKUs; recovered `2,545` exact integer unit rows from `Revenue_USD / Unit_Price_USD`; flagged `999` unresolved (`outputs/reports/stage_b_cleaning_summary.csv`) |
| **Stage C** | Supply-constrained inventory records (`5,830` rows) | Validated `Requested + Carried_In = Shipped + Backordered + Lost` (`0` fulfillment gaps) (`outputs/reports/stage_c_cleaning_summary.csv`) |
| **Stage D–E** | Stockout demand censoring & `491` negative return/credit rows | Excluded negative rows into `stage_e_negative_demand_investigation.csv`; recovered `+94,990` unconstrained units during stockout months |
| **Stage F** | Lifecycle boundaries & external business events | Enforced zero demand outside `[Launch_Date, EOL_Date]`; mapped `Replaces_SKU` history to planned launches; encoded `events_calendar.csv` & lagged macro indicators |

---

## 4. Control Tower Dashboard (`app.py`)

The Streamlit Control Tower reads dynamically from pipeline outputs (`train_stage_e.csv`, `forecast_stage_g_full.csv`, `sku_master.csv`, and `outputs/reports/*.csv`) and shares sidebar filters (`Control Tower Part`, `Product Family`, `Region`, `Sales Channel`, `Customer Name`, `SKU ID`) across six views:

1. **Executive Overview:** Continuous timeline from historical unconstrained demand (`2019-01`–`2026-10`) into the 19-month forecast (`2026-11`–`2028-05`), KPI cards (`12,248,788` historical unconstrained units, `2,502,206` P50 forecast, `3,643,765` P90 upper band, `94,990` stockout units recovered), and forecast mix by product family and region.
2. **Data Health:** Cleaning operation counts, file reconciliation summaries, and feature completeness tables.
3. **Demand Explorer:** Monthly seasonality profile, zero-demand month share, customer/channel breakdown, and top SKU-region demand tables.
4. **Anomaly Monitor:** Structured Anomaly Register (separating data errors from real supply/market events), negative demand transaction log, and active event flags.
5. **Forecast & Accuracy:** Stage F backtest WAPE and bias tables overall, by product family, by region, and across forecast horizons.
6. **P90 Procurement, What-If & Lifecycle:** Interactive `ISM_Manufacturing_PMI` shock simulator, `P90` procurement quantities, and `EOL Announced` / `Discontinued` / `Planned Launch` SKU watchlist.

---

## 5. Installation & Quickstart

### Local Execution
```bash
# 1. Clone the repository
git clone https://github.com/naruto-kavi/celeris-demand-forecasting.git
cd celeris-demand-forecasting

# 2. Create and activate a virtual environment
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Stage E–G pipeline (feature engineering, backtesting, and submission generation)
python src/stage_e_f_g_pipeline.py

# 5. Launch the Planning Control Tower dashboard
streamlit run app.py
```

---

## 6. Repository Structure

```text
celeris-demand-forecasting/
├── app.py                                      # Unified Streamlit Planning Control Tower
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── data/
│   ├── raw/                                    # Original hackathon input files
│   └── processed/                              # Staged datasets (train_stage_e.csv, test_stage_e.csv, forecast_stage_g_full.csv)
├── outputs/
│   ├── reports/                                # Cleaning, anomaly, and Stage F WAPE/Bias evaluation reports
│   └── submissions/
│       └── submission.csv                      # Final 58,219-row non-negative integer forecast
└── src/
    ├── profile_data.py                         # Initial data profiling
    ├── investigate_data.py                     # Raw anomaly & schema investigation
    ├── data_cleaning.py                        # Stage A standardization & Record_ID deduplication
    ├── stage_b_investigation.py                # Stage B quality checks
    ├── stage_b2_investigation.py               # Stage B deep-dive diagnostics
    ├── stage_b_cleaning.py                     # Stage B product, region, target recovery & macro cleaning
    ├── stage_c_inventory_investigation.py      # Stage C stockout & fill-rate investigation
    ├── stage_c_cleaning.py                     # Stage C inventory reconciliation
    ├── stage_d_feature_engineering.py          # Stage D monthly aggregation & base features
    └── stage_e_f_g_pipeline.py                 # Stages E–G unconstrained demand, backtesting & recursive forecast
```