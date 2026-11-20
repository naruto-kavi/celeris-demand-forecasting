
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train_stage_b.csv"
MACRO_PATH = ROOT / "data" / "processed" / "macro_stage_b.csv"
INVENTORY_PATH = ROOT / "data" / "processed" / "inventory_stage_c.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CELERIS - STAGE D FEATURE ENGINEERING")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\n1. LOADING DATASETS")

sales = pd.read_csv(TRAIN_PATH)
macro = pd.read_csv(MACRO_PATH)
inventory = pd.read_csv(INVENTORY_PATH)

print("Sales rows:", len(sales))
print("Macro rows:", len(macro))
print("Inventory rows:", len(inventory))

# --------------------------------------------------
# 2. STANDARDIZE DATES
# --------------------------------------------------

print("\n2. STANDARDIZING DATES")

sales["Date"] = pd.to_datetime(
    sales["Date"], errors="coerce"
).dt.to_period("M").dt.to_timestamp()

macro["Date"] = pd.to_datetime(
    macro["Date"], errors="coerce"
).dt.to_period("M").dt.to_timestamp()

inventory["Date"] = pd.to_datetime(
    inventory["Date"], errors="coerce"
).dt.to_period("M").dt.to_timestamp()

if sales["Date"].isna().any():
    raise ValueError("Invalid sales dates.")

if macro["Date"].isna().any():
    raise ValueError("Invalid macro dates.")

if inventory["Date"].isna().any():
    raise ValueError("Invalid inventory dates.")

# --------------------------------------------------
# 3. CONVERT NUMERIC COLUMNS
# --------------------------------------------------

print("\n3. CONVERTING NUMERIC COLUMNS")

sales_numeric = [
    "Units_Sold",
    "Unit_Price_USD",
    "Revenue_USD"
]

for col in sales_numeric:
    sales[col] = pd.to_numeric(sales[col], errors="coerce")

inventory_numeric = [
    "Requested_Units",
    "Carried_In_Units",
    "Shipped_Units",
    "Backordered_Units",
    "Lost_Units",
    "Stockout_Days"
]

for col in inventory_numeric:
    inventory[col] = pd.to_numeric(
        inventory[col], errors="coerce"
    )

# --------------------------------------------------
# 4. AGGREGATE SALES MONTHLY
# --------------------------------------------------

print("\n4. AGGREGATING MONTHLY DEMAND")

group_columns = ["Date", "SKU_ID", "Region"]

monthly_sales = (
    sales.groupby(group_columns, as_index=False)
    .agg(
        Units_Sold=("Units_Sold", "sum"),
        Revenue_USD=("Revenue_USD", "sum"),
        Unit_Price_USD=("Unit_Price_USD", "mean"),
        Product_Family=("Product_Family", "first"),
        Business_Unit=("Business_Unit", "first"),
        SKU_Description=("SKU_Description", "first"),
        Customer_Count=("Customer_Name", "nunique"),
        Sales_Channel_Count=("Sales_Channel", "nunique")
    )
)

print("Monthly sales rows:", len(monthly_sales))

# --------------------------------------------------
# 5. MERGE MACROECONOMIC DATA
# --------------------------------------------------

print("\n5. MERGING MACRO INDICATORS")

dataset = monthly_sales.merge(
    macro,
    on="Date",
    how="left",
    validate="many_to_one"
)

# --------------------------------------------------
# 6. AGGREGATE AND MERGE INVENTORY
# --------------------------------------------------

print("\n6. MERGING INVENTORY INFORMATION")

inventory_group = (
    inventory.groupby(group_columns, as_index=False)
    .agg(
        Requested_Units=("Requested_Units", "sum"),
        Carried_In_Units=("Carried_In_Units", "sum"),
        Shipped_Units=("Shipped_Units", "sum"),
        Backordered_Units=("Backordered_Units", "sum"),
        Lost_Units=("Lost_Units", "sum"),
        Stockout_Days=("Stockout_Days", "max"),
        Inventory_Records=("SKU_ID", "size")
    )
)

dataset = dataset.merge(
    inventory_group,
    on=group_columns,
    how="left",
    validate="one_to_one"
)

# Preserve missing inventory as missing rather than assuming
# that an absent inventory record means zero stock.

dataset["Has_Inventory_Record"] = (
    dataset["Inventory_Records"].notna().astype(int)
)

# --------------------------------------------------
# 7. CREATE CALENDAR FEATURES
# --------------------------------------------------

print("\n7. CREATING CALENDAR FEATURES")

dataset["Year"] = dataset["Date"].dt.year
dataset["Month"] = dataset["Date"].dt.month
dataset["Quarter"] = dataset["Date"].dt.quarter

dataset["Month_Sin"] = np.sin(
    2 * np.pi * dataset["Month"] / 12
)

dataset["Month_Cos"] = np.cos(
    2 * np.pi * dataset["Month"] / 12
)

# --------------------------------------------------
# 8. CREATE TIME SERIES FEATURES
# --------------------------------------------------

print("\n8. CREATING LAG AND ROLLING FEATURES")

dataset = dataset.sort_values(
    ["SKU_ID", "Region", "Date"]
).reset_index(drop=True)

grouped = dataset.groupby(
    ["SKU_ID", "Region"]
)["Units_Sold"]

for lag in [1, 2, 3, 6, 12]:
    dataset[f"Demand_Lag_{lag}"] = grouped.shift(lag)

for window in [3, 6, 12]:
    dataset[f"Demand_Rolling_Mean_{window}"] = (
        dataset.groupby(["SKU_ID", "Region"])["Units_Sold"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=window,
                min_periods=window
            ).mean()
        )
    )

dataset["Demand_Rolling_Std_3"] = (
    dataset.groupby(["SKU_ID", "Region"])["Units_Sold"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).std()
    )
)

# --------------------------------------------------
# 9. CREATE INVENTORY PERFORMANCE FEATURES
# --------------------------------------------------

print("\n9. CREATING INVENTORY FEATURES")

dataset["Backorder_Rate"] = (
    dataset["Backordered_Units"]
    .div(dataset["Requested_Units"].replace(0, np.nan))
)

dataset["Lost_Demand_Rate"] = (
    dataset["Lost_Units"]
    .div(dataset["Requested_Units"].replace(0, np.nan))
)

dataset["Inventory_Fulfillment_Rate"] = (
    dataset["Shipped_Units"]
    .div(dataset["Requested_Units"].replace(0, np.nan))
)

# --------------------------------------------------
# 10. VALIDATION
# --------------------------------------------------

print("\n10. VALIDATING FINAL DATASET")

if dataset.duplicated(group_columns).any():
    raise ValueError("Duplicate monthly SKU-region keys found.")

dataset = dataset.replace([np.inf, -np.inf], np.nan)

print("Final rows:", len(dataset))
print("Unique SKUs:", dataset["SKU_ID"].nunique())
print("Unique regions:", dataset["Region"].nunique())
print("Date range:", dataset["Date"].min(), "to", dataset["Date"].max())

# --------------------------------------------------
# 11. SAVE OUTPUT
# --------------------------------------------------

output_path = OUTPUT_DIR / "forecasting_dataset_stage_d.csv"

dataset["Date"] = dataset["Date"].dt.strftime("%Y-%m")

dataset.to_csv(output_path, index=False)

# Save feature missing-value report.

missing_report = (
    dataset.isna()
    .sum()
    .reset_index()
)

missing_report.columns = ["Column", "Missing_Count"]

missing_report["Missing_Percentage"] = (
    missing_report["Missing_Count"] / len(dataset) * 100
)

missing_report.to_csv(
    REPORT_DIR / "stage_d_feature_missing_report.csv",
    index=False
)

# Save feature summary.

feature_summary = pd.DataFrame({
    "Metric": [
        "Original sales rows",
        "Monthly forecasting rows",
        "Unique SKUs",
        "Unique regions",
        "Number of features",
        "Inventory records matched",
        "Rows with missing inventory"
    ],
    "Value": [
        len(sales),
        len(dataset),
        dataset["SKU_ID"].nunique(),
        dataset["Region"].nunique(),
        len(dataset.columns),
        int(dataset["Has_Inventory_Record"].sum()),
        int((dataset["Has_Inventory_Record"] == 0).sum())
    ]
})

feature_summary.to_csv(
    REPORT_DIR / "stage_d_feature_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("STAGE D FEATURE ENGINEERING COMPLETED")
print("=" * 70)
print("Output:", output_path)
print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))
print("Missing report:", REPORT_DIR / "stage_d_feature_missing_report.csv")
print("=" * 70)
