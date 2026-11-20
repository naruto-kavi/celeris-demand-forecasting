
"""
======================================================================
CELERIS - STAGE E DATASET PREPARATION
======================================================================

Purpose:
1. Load Stage D engineered training dataset.
2. Load original training and test datasets.
3. Investigate negative demand transactions.
4. Preserve negative transactions in an investigation report.
5. Exclude negative transactions from forecasting demand aggregation.
6. Prepare non-negative forecasting targets.
7. Prepare training and test datasets.
8. Validate dataset consistency.
9. Save final datasets and reports.

Negative demand policy:
Negative Units_Sold values are treated as potential returns,
cancellations, or accounting adjustments.

They are NOT converted to zero at transaction level.
They are excluded from the gross-demand forecasting target.

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "outputs" / "reports"

TRAIN_STAGE_B = DATA_PROCESSED / "train_stage_b.csv"
STAGE_D_FILE = DATA_PROCESSED / "forecasting_dataset_stage_d.csv"
INVENTORY_FILE = DATA_PROCESSED / "inventory_stage_c.csv"

TEST_FILE = DATA_RAW / "test.csv"

OUTPUT_TRAIN = DATA_PROCESSED / "train_stage_e.csv"
OUTPUT_TEST = DATA_PROCESSED / "test_stage_e.csv"

NEGATIVE_REPORT = REPORTS / "stage_e_negative_demand_investigation.csv"
NEGATIVE_YEAR_REPORT = REPORTS / "stage_e_negative_demand_by_year.csv"
DATASET_REPORT = REPORTS / "stage_e_dataset_validation.csv"


# ================================================================
# 2. CONFIGURATION
# ================================================================

KEY_COLUMNS = ["Date", "SKU_ID", "Region"]

TARGET_COLUMN = "Units_Sold"

NEGATIVE_POLICY = "Exclude negative transactions from gross demand"


# ================================================================
# 3. HELPER FUNCTIONS
# ================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"{name}: {df.shape}")

    return df


def standardize_dates(df):
    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError("Date column missing.")

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str),
        format="%Y-%m",
        errors="coerce"
    )

    invalid = df["Date"].isna().sum()

    if invalid:
        raise ValueError(
            f"Found {invalid} invalid dates."
        )

    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    return df


def standardize_keys(df):
    df = df.copy()

    for col in ["SKU_ID", "Region"]:
        if col not in df.columns:
            raise ValueError(f"Missing key column: {col}")

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    return df


def validate_unique_keys(df, name):
    duplicates = df.duplicated(KEY_COLUMNS).sum()

    print(f"{name} duplicate business keys: {duplicates}")

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate Date-SKU-Region records."
        )


def safe_numeric(df, columns):
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ================================================================
# 4. MAIN EXECUTION
# ================================================================

def main():

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print_section("CELERIS - STAGE E DATASET PREPARATION")

    validation_results = []

    # ------------------------------------------------------------
    # STEP 1: LOAD DATASETS
    # ------------------------------------------------------------

    print_section("1. LOADING DATASETS")

    train_raw = load_csv(
        TRAIN_STAGE_B,
        "Stage B training dataset"
    )

    stage_d = load_csv(
        STAGE_D_FILE,
        "Stage D forecasting dataset"
    )

    test = load_csv(
        TEST_FILE,
        "Test dataset"
    )

    # ------------------------------------------------------------
    # STEP 2: STANDARDIZE DATES AND KEYS
    # ------------------------------------------------------------

    print_section("2. STANDARDIZING DATES AND KEYS")

    train_raw = standardize_dates(train_raw)
    stage_d = standardize_dates(stage_d)
    test = standardize_dates(test)

    train_raw = standardize_keys(train_raw)
    stage_d = standardize_keys(stage_d)
    test = standardize_keys(test)

    # ------------------------------------------------------------
    # STEP 3: NUMERIC CONVERSION
    # ------------------------------------------------------------

    print_section("3. NUMERIC CONVERSION")

    train_raw = safe_numeric(
        train_raw,
        [
            "Units_Sold",
            "Revenue_USD",
            "Unit_Price_USD"
        ]
    )

    stage_d = safe_numeric(
        stage_d,
        [
            "Units_Sold",
            "Revenue_USD",
            "Unit_Price_USD",
            "Requested_Units",
            "Carried_In_Units",
            "Shipped_Units",
            "Backordered_Units",
            "Lost_Units",
            "Stockout_Days"
        ]
    )

    if train_raw["Units_Sold"].isna().any():
        raise ValueError(
            "Missing or invalid Units_Sold values in Stage B."
        )

    if stage_d["Units_Sold"].isna().any():
        raise ValueError(
            "Missing or invalid Units_Sold values in Stage D."
        )

    # ------------------------------------------------------------
    # STEP 4: NEGATIVE DEMAND INVESTIGATION
    # ------------------------------------------------------------

    print_section("4. NEGATIVE DEMAND INVESTIGATION")

    negative = train_raw[
        train_raw["Units_Sold"] < 0
    ].copy()

    positive = train_raw[
        train_raw["Units_Sold"] > 0
    ].copy()

    zero = train_raw[
        train_raw["Units_Sold"] == 0
    ].copy()

    negative.to_csv(
        NEGATIVE_REPORT,
        index=False
    )

    print("Negative demand rows:", len(negative))
    print("Negative demand units:", negative["Units_Sold"].sum())
    print("Negative revenue:", negative["Revenue_USD"].sum())

    print("Positive demand rows:", len(positive))
    print("Zero demand rows:", len(zero))

    # Year-wise negative transaction report

    if not negative.empty:

        negative_by_year = (
            negative.groupby("Year")
            .agg(
                Negative_Records=("Units_Sold", "size"),
                Negative_Units=("Units_Sold", "sum"),
                Negative_Revenue=("Revenue_USD", "sum")
            )
            .reset_index()
        )

    else:

        negative_by_year = pd.DataFrame(
            columns=[
                "Year",
                "Negative_Records",
                "Negative_Units",
                "Negative_Revenue"
            ]
        )

    negative_by_year.to_csv(
        NEGATIVE_YEAR_REPORT,
        index=False
    )

    print("\nNegative demand by year:")
    print(negative_by_year.to_string(index=False))

    validation_results.append({
        "Check": "Negative demand transactions",
        "Value": len(negative),
        "Status": "INVESTIGATED"
    })

    # ------------------------------------------------------------
    # STEP 5: CREATE GROSS DEMAND DATASET
    # ------------------------------------------------------------

    print_section("5. CREATING GROSS DEMAND DATASET")

    # Preserve only non-negative transactions.
    # Do not modify original Stage B data.

    valid_sales = train_raw[
        train_raw["Units_Sold"] >= 0
    ].copy()

    print("Original transaction rows:", len(train_raw))
    print("Valid gross-demand rows:", len(valid_sales))
    print("Excluded negative transactions:", len(negative))

    if valid_sales.empty:
        raise ValueError("No valid gross-demand records found.")

    # ------------------------------------------------------------
    # STEP 6: AGGREGATE MONTHLY DEMAND
    # ------------------------------------------------------------

    print_section("6. AGGREGATING MONTHLY DEMAND")

    aggregation = {
        "Units_Sold": "sum",
        "Revenue_USD": "sum",
        "Unit_Price_USD": "mean",
        "Product_Family": "first",
        "Business_Unit": "first",
        "SKU_Description": "first"
    }

    monthly = (
        valid_sales
        .groupby(KEY_COLUMNS, as_index=False)
        .agg(aggregation)
    )

    print("Monthly forecasting rows:", len(monthly))
    print("Unique SKUs:", monthly["SKU_ID"].nunique())
    print("Unique regions:", monthly["Region"].nunique())

    if (monthly["Units_Sold"] < 0).any():
        raise ValueError(
            "Negative targets remain after aggregation."
        )

    validate_unique_keys(
        monthly,
        "Monthly gross-demand dataset"
    )

    # ------------------------------------------------------------
    # STEP 7: MERGE STAGE D FEATURES
    # ------------------------------------------------------------

    print_section("7. MERGING ENGINEERED FEATURES")

    # Keep Stage D engineered features.
    # Replace its demand target with the newly prepared
    # gross-demand target.

    feature_columns = [
        col for col in stage_d.columns
        if col not in [
            "Units_Sold",
            "Revenue_USD",
            "Unit_Price_USD"
        ]
    ]

    stage_d_features = stage_d[feature_columns].copy()

    train = monthly.merge(
        stage_d_features,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one"
    )

    print("Training rows after merge:", len(train))

    if len(train) != len(monthly):
        raise ValueError(
            "Unexpected row count after Stage D feature merge."
        )

    # ------------------------------------------------------------
    # STEP 8: CHECK FEATURE AVAILABILITY
    # ------------------------------------------------------------

    print_section("8. FEATURE AVAILABILITY")

    missing_features = (
        train.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("Top missing columns:")
    print(missing_features.head(20).to_string())

    # ------------------------------------------------------------
    # STEP 9: PREPARE TEST DATASET
    # ------------------------------------------------------------

    print_section("9. PREPARING TEST DATASET")

    test["Year"] = pd.to_numeric(
        test["Year"],
        errors="coerce"
    )

    test["Month"] = pd.to_numeric(
        test["Month"],
        errors="coerce"
    )

    if test[["Year", "Month"]].isna().any().any():
        raise ValueError(
            "Invalid Year or Month in test dataset."
        )

    validate_unique_keys(
        test,
        "Test dataset"
    )

    # Combine historical gross demand with test calendar
    # to create historical lag features for future periods.

    historical = train[
        KEY_COLUMNS + ["Units_Sold"]
    ].copy()

    historical["Dataset_Type"] = "train"

    future = test[KEY_COLUMNS].copy()
    future["Units_Sold"] = np.nan
    future["Dataset_Type"] = "test"

    combined = pd.concat(
        [historical, future],
        ignore_index=True
    )

    combined = combined.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # STEP 10: CREATE TEST LAG FEATURES
    # ------------------------------------------------------------

    print_section("10. GENERATING TEST LAG FEATURES")

    combined = combined.sort_values(
        ["SKU_ID", "Region", "Date"]
    ).reset_index(drop=True)

    grouped = combined.groupby(
        ["SKU_ID", "Region"],
        sort=False
    )

    for lag in [1, 2, 3, 6, 12]:

        combined[f"Demand_Lag_{lag}"] = (
            grouped["Units_Sold"].shift(lag)
        )

    for window in [3, 6, 12]:

        combined[f"Demand_Rolling_Mean_{window}"] = (
            grouped["Units_Sold"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=1
                ).mean()
            )
        )

    combined["Demand_Rolling_Std_3"] = (
        combined.groupby(
            ["SKU_ID", "Region"]
        )["Units_Sold"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3,
                min_periods=2
            ).std()
        )
    )

    test_lags = combined[
        combined["Dataset_Type"] == "test"
    ].copy()

    lag_columns = [
        "Demand_Lag_1",
        "Demand_Lag_2",
        "Demand_Lag_3",
        "Demand_Lag_6",
        "Demand_Lag_12",
        "Demand_Rolling_Mean_3",
        "Demand_Rolling_Mean_6",
        "Demand_Rolling_Mean_12",
        "Demand_Rolling_Std_3"
    ]

    test_lags = test_lags[
        KEY_COLUMNS + lag_columns
    ]

    # ------------------------------------------------------------
    # STEP 11: MERGE TEST LAGS
    # ------------------------------------------------------------

    print_section("11. MERGING TEST LAG FEATURES")

    test_final = test.merge(
        test_lags,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one"
    )

    # ------------------------------------------------------------
    # STEP 12: MERGE MACRO FEATURES
    # ------------------------------------------------------------

    print_section("12. MERGING MACRO AND CALENDAR FEATURES")

    macro_path = DATA_PROCESSED / "macro_stage_b.csv"

    if macro_path.exists():

        macro = pd.read_csv(macro_path)

        macro = standardize_dates(macro)

        macro_columns = [
            col for col in macro.columns
            if col not in ["Year", "Month"]
        ]

        macro = macro[macro_columns]

        test_final = test_final.merge(
            macro,
            on="Date",
            how="left",
            validate="many_to_one",
            suffixes=("", "_Macro")
        )

        print("Macro features merged.")

    else:
        print("Macro dataset not found. Skipping macro merge.")

    # ------------------------------------------------------------
    # STEP 13: CREATE CALENDAR FEATURES
    # ------------------------------------------------------------

    print_section("13. CREATING CALENDAR FEATURES")

    for df in [train, test_final]:

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month

        df["Quarter"] = df["Date"].dt.quarter

        df["Month_Sin"] = np.sin(
            2 * np.pi * df["Month"] / 12
        )

        df["Month_Cos"] = np.cos(
            2 * np.pi * df["Month"] / 12
        )

    # ------------------------------------------------------------
    # STEP 14: INVENTORY FEATURE HANDLING
    # ------------------------------------------------------------

    print_section("14. INVENTORY FEATURE VALIDATION")

    inventory_features = [
        "Requested_Units",
        "Carried_In_Units",
        "Shipped_Units",
        "Backordered_Units",
        "Lost_Units",
        "Stockout_Days",
        "Has_Inventory_Record"
    ]

    for col in inventory_features:

        if col not in train.columns:
            train[col] = np.nan

        if col not in test_final.columns:
            test_final[col] = np.nan

    # ------------------------------------------------------------
    # STEP 15: FINAL TARGET VALIDATION
    # ------------------------------------------------------------

    print_section("15. FINAL TARGET VALIDATION")

    if train["Units_Sold"].isna().any():
        raise ValueError(
            "Missing training target values."
        )

    if (train["Units_Sold"] < 0).any():
        raise ValueError(
            "Negative demand found in final training dataset."
        )

    print(
        "Negative target records:",
        (train["Units_Sold"] < 0).sum()
    )

    print(
        "Zero demand records:",
        (train["Units_Sold"] == 0).sum()
    )

    print(
        "Positive demand records:",
        (train["Units_Sold"] > 0).sum()
    )

    # ------------------------------------------------------------
    # STEP 16: BUSINESS KEY VALIDATION
    # ------------------------------------------------------------

    print_section("16. BUSINESS KEY VALIDATION")

    validate_unique_keys(
        train,
        "Final training dataset"
    )

    validate_unique_keys(
        test_final,
        "Final test dataset"
    )

    # ------------------------------------------------------------
    # STEP 17: FEATURE ALIGNMENT
    # ------------------------------------------------------------

    print_section("17. FEATURE ALIGNMENT")

    # Training and test datasets do not have the same target
    # and historical revenue fields. Preserve the target only
    # in the training dataset.

    excluded_from_test = [
        "Units_Sold",
        "Revenue_USD"
    ]

    common_columns = [
        col for col in train.columns
        if col not in excluded_from_test
        and col in test_final.columns
    ]

    print("Common feature columns:", len(common_columns))

    print(
        "Training columns:",
        len(train.columns)
    )

    print(
        "Test columns:",
        len(test_final.columns)
    )

    # ------------------------------------------------------------
    # STEP 18: SAVE FINAL DATASETS
    # ------------------------------------------------------------

    print_section("18. SAVING FINAL DATASETS")

    train.to_csv(
        OUTPUT_TRAIN,
        index=False
    )

    test_final.to_csv(
        OUTPUT_TEST,
        index=False
    )

    print("Training dataset saved:")
    print(OUTPUT_TRAIN)

    print("\nTest dataset saved:")
    print(OUTPUT_TEST)

    # ------------------------------------------------------------
    # STEP 19: SAVE VALIDATION REPORT
    # ------------------------------------------------------------

    validation_results.extend([
        {
            "Check": "Final training rows",
            "Value": len(train),
            "Status": "PASS"
        },
        {
            "Check": "Final test rows",
            "Value": len(test_final),
            "Status": "PASS"
        },
        {
            "Check": "Negative final targets",
            "Value": int((train["Units_Sold"] < 0).sum()),
            "Status": "PASS"
        },
        {
            "Check": "Duplicate training keys",
            "Value": int(train.duplicated(KEY_COLUMNS).sum()),
            "Status": "PASS"
        },
        {
            "Check": "Duplicate test keys",
            "Value": int(test_final.duplicated(KEY_COLUMNS).sum()),
            "Status": "PASS"
        },
        {
            "Check": "Negative transaction policy",
            "Value": NEGATIVE_POLICY,
            "Status": "DOCUMENTED"
        }
    ])

    report = pd.DataFrame(validation_results)

    report.to_csv(
        DATASET_REPORT,
        index=False
    )

    # ------------------------------------------------------------
    # STEP 20: FINAL SUMMARY
    # ------------------------------------------------------------

    print_section("STAGE E COMPLETED SUCCESSFULLY")

    print("Training shape:", train.shape)
    print("Test shape:", test_final.shape)

    print("Training date range:")
    print(
        train["Date"].min(),
        "to",
        train["Date"].max()
    )

    print("\nTest date range:")
    print(
        test_final["Date"].min(),
        "to",
        test_final["Date"].max()
    )

    print("\nNegative transactions investigated:", len(negative))

    print("Negative transactions removed from gross demand aggregation:",
          len(negative))

    print("\nOUTPUT FILES")

    print("1.", OUTPUT_TRAIN)
    print("2.", OUTPUT_TEST)
    print("3.", NEGATIVE_REPORT)
    print("4.", NEGATIVE_YEAR_REPORT)
    print("5.", DATASET_REPORT)

    print_section("STAGE E DATASET PREPARATION FINISHED")


# ================================================================
# EXECUTION
# ================================================================

if __name__ == "__main__":
    main()
