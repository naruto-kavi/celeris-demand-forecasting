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

## Local Setup & Execution

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run feature engineering, backtesting, and 19-month forecast generation
python src/stage_e_f_g_pipeline.py

# 4. Launch the Streamlit Control Tower
streamlit run app.py
```

## Repository Structure

- `app.py` — Main Streamlit Control Tower application
- `src/` — Data profiling, cleaning (Stages A–C), feature engineering (Stages D–E), backtesting & forecasting (Stages F–G), and source scripts
- `data/raw/` & `data/processed/` — Raw inputs and staged processed datasets (`train_stage_e.csv`, `test_stage_e.csv`, `forecast_stage_g_full.csv`, etc.)
- `outputs/reports/` — Cleaning, anomaly, and backtest evaluation reports
- `outputs/submissions/submission.csv` — Final `58,219`-row non-negative integer forecast submission