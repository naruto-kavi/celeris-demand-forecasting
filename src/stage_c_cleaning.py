
from pathlib import Path
import pandas as pd
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INVENTORY_PATH = ROOT / "data" / "raw" / "inventory_stockouts.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train_stage_b.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CELERIS - STAGE C INVENTORY CLEANING")
print("=" * 65)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

inventory = pd.read_csv(INVENTORY_PATH)
train = pd.read_csv(TRAIN_PATH)

original_rows = len(inventory)

print("\n1. SOURCE DATA")
print("Inventory rows:", original_rows)
print("Training rows:", len(train))


# --------------------------------------------------
# 2. STANDARDIZE DATES
# --------------------------------------------------

print("\n2. STANDARDIZING DATES")

inventory["Date"] = pd.to_datetime(
    inventory["Date"],
    errors="coerce"
)

invalid_dates = inventory[inventory["Date"].isna()].copy()

invalid_dates.to_csv(
    REPORT_DIR / "stage_c_invalid_dates.csv",
    index=False
)

if not invalid_dates.empty:
    raise ValueError(
        f"Found {len(invalid_dates)} invalid inventory dates."
    )

inventory["Date"] = (
    inventory["Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# --------------------------------------------------
# 3. STANDARDIZE SKU IDENTIFIERS
# --------------------------------------------------

print("\n3. STANDARDIZING SKU IDENTIFIERS")


def normalize_sku(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().upper()

    value = (
        value.replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    match = re.fullmatch(r"([A-Z]+)(\d+)", value)

    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"

    return value


inventory["SKU_ID"] = inventory["SKU_ID"].apply(
    normalize_sku
)

train["SKU_ID"] = train["SKU_ID"].apply(
    normalize_sku
)


# --------------------------------------------------
# 4. STANDARDIZE REGIONS
# --------------------------------------------------

print("\n4. STANDARDIZING REGIONS")


def normalize_region(value):

    if pd.isna(value) or str(value).strip() == "":
        return pd.NA

    key = re.sub(
        r"[^a-z]",
        "",
        str(value).lower()
    )

    mapping = {
        "india": "India",
        "apac": "Asia Pacific",
        "asiapacific": "Asia Pacific",
        "europe": "Europe",
        "eu": "Europe",
        "latam": "Latin America",
        "latinamerica": "Latin America",
        "na": "North America",
        "namerica": "North America",
        "northamerica": "North America",
        "mea": "Middle East & Africa",
        "middleeastafrica": "Middle East & Africa",
        "middleeastandafrica": "Middle East & Africa",
    }

    return mapping.get(key, str(value).strip())


inventory["Region"] = inventory["Region"].apply(
    normalize_region
)

train["Region"] = train["Region"].apply(
    normalize_region
)

print(inventory["Region"].value_counts(dropna=False))


# --------------------------------------------------
# 5. NUMERIC CONVERSION
# --------------------------------------------------

print("\n5. NUMERIC CONVERSION")

numeric_columns = [
    "Requested_Units",
    "Carried_In_Units",
    "Shipped_Units",
    "Backordered_Units",
    "Lost_Units",
    "Fill_Rate_Pct",
    "Stockout_Days"
]

numeric_report = []

for column in numeric_columns:

    original = inventory[column].copy()

    cleaned = (
        original.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({
            "": pd.NA,
            "-": pd.NA,
            "nan": pd.NA
        })
    )

    converted = pd.to_numeric(
        cleaned,
        errors="coerce"
    )

    numeric_report.append({
        "Column": column,
        "Missing_Before": int(original.isna().sum()),
        "Missing_After": int(converted.isna().sum()),
        "Newly_Missing": int(
            (original.notna() & converted.isna()).sum()
        )
    })

    inventory[column] = converted

print(inventory[numeric_columns].isna().sum())

pd.DataFrame(numeric_report).to_csv(
    REPORT_DIR / "stage_c_numeric_conversion_report.csv",
    index=False
)


# --------------------------------------------------
# 6. MISSING VALUE INVESTIGATION
# --------------------------------------------------

print("\n6. MISSING VALUE INVESTIGATION")

missing_records = inventory[
    inventory[numeric_columns].isna().any(axis=1)
    | inventory[["SKU_ID", "Region"]].isna().any(axis=1)
].copy()

missing_records.to_csv(
    REPORT_DIR / "stage_c_missing_values_review.csv",
    index=False
)


# --------------------------------------------------
# 7. NEGATIVE VALUE VALIDATION
# --------------------------------------------------

print("\n7. NEGATIVE VALUE VALIDATION")

quantity_columns = [
    "Requested_Units",
    "Carried_In_Units",
    "Shipped_Units",
    "Backordered_Units",
    "Lost_Units",
    "Stockout_Days"
]

negative_mask = (
    inventory[quantity_columns]
    .lt(0)
    .any(axis=1)
    |
    inventory["Fill_Rate_Pct"].lt(0)
)

negative_records = inventory[
    negative_mask
].copy()

negative_records.to_csv(
    REPORT_DIR / "stage_c_negative_inventory_values.csv",
    index=False
)

print("Negative-value records:", len(negative_records))


# --------------------------------------------------
# 8. FULFILLMENT RECONCILIATION
# --------------------------------------------------

print("\n8. FULFILLMENT RECONCILIATION")

inventory["Unshipped_Units"] = (
    inventory["Requested_Units"]
    - inventory["Shipped_Units"]
)

inventory["Fulfillment_Gap"] = (
    inventory["Requested_Units"]
    + inventory["Carried_In_Units"]
    - inventory["Shipped_Units"]
    - inventory["Backordered_Units"]
    - inventory["Lost_Units"]
)

# Keep this as a diagnostic.
# Do not assume every unshipped unit must be classified
# as either backordered or lost.

reconciliation_issues = inventory[
    inventory["Fulfillment_Gap"].abs() > 0
].copy()

reconciliation_issues.to_csv(
    REPORT_DIR / "stage_c_fulfillment_reconciliation.csv",
    index=False
)

print(
    "Fulfillment differences:",
    len(reconciliation_issues)
)


# --------------------------------------------------
# 9. FILL RATE VALIDATION
# --------------------------------------------------

print("\n9. FILL RATE VALIDATION")

inventory["Calculated_Fill_Rate_Pct"] = (
    inventory["Shipped_Units"]
    .div(
        (
            inventory["Requested_Units"]
            + inventory["Carried_In_Units"]
        ).replace(0, pd.NA)
    )
    .mul(100)
)
inventory["Fill_Rate_Difference"] = (
    inventory["Fill_Rate_Pct"]
    - inventory["Calculated_Fill_Rate_Pct"]
)

# A tolerance of 1 percentage point is used
# to avoid reporting minor rounding differences.

fill_rate_issues = inventory[
    inventory["Fill_Rate_Difference"].abs() > 1
].copy()

fill_rate_issues.to_csv(
    REPORT_DIR / "stage_c_fill_rate_review.csv",
    index=False
)

print("Fill-rate discrepancies:", len(fill_rate_issues))


# --------------------------------------------------
# 10. DUPLICATE BUSINESS KEY CHECK
# --------------------------------------------------

print("\n10. DUPLICATE BUSINESS KEY CHECK")

business_keys = [
    "Date",
    "SKU_ID",
    "Region"
]

duplicate_mask = inventory.duplicated(
    subset=business_keys,
    keep=False
)

duplicates = inventory[
    duplicate_mask
].copy()

duplicates.to_csv(
    REPORT_DIR / "stage_c_duplicate_business_keys.csv",
    index=False
)

duplicate_groups = (
    duplicates.groupby(
        business_keys,
        dropna=False
    ).ngroups
    if not duplicates.empty
    else 0
)

print("Duplicate rows:", len(duplicates))
print("Duplicate groups:", duplicate_groups)


# --------------------------------------------------
# 11. SKU RECONCILIATION
# --------------------------------------------------

print("\n11. SKU RECONCILIATION")

inventory_skus = set(
    inventory["SKU_ID"].dropna().unique()
)

training_skus = set(
    train["SKU_ID"].dropna().unique()
)

unknown_skus = inventory_skus - training_skus

missing_inventory_skus = training_skus - inventory_skus

print("Inventory unique SKUs:", len(inventory_skus))
print("Training unique SKUs:", len(training_skus))
print("Unknown inventory SKUs:", len(unknown_skus))
print("Training SKUs without inventory records:",
      len(missing_inventory_skus))

pd.DataFrame({
    "Unknown_Inventory_SKU": sorted(unknown_skus)
}).to_csv(
    REPORT_DIR / "stage_c_unknown_inventory_skus.csv",
    index=False
)

pd.DataFrame({
    "Training_SKU_Without_Inventory": sorted(
        missing_inventory_skus
    )
}).to_csv(
    REPORT_DIR / "stage_c_training_skus_without_inventory.csv",
    index=False
)


# --------------------------------------------------
# 12. FILL RATE AND STOCKOUT SUMMARY
# --------------------------------------------------

print("\n12. INVENTORY PERFORMANCE SUMMARY")

inventory_summary = pd.DataFrame([{
    "Total_Requested_Units": inventory["Requested_Units"].sum(),
    "Total_Shipped_Units": inventory["Shipped_Units"].sum(),
    "Total_Backordered_Units": inventory["Backordered_Units"].sum(),
    "Total_Lost_Units": inventory["Lost_Units"].sum(),
    "Average_Fill_Rate_Pct": inventory["Fill_Rate_Pct"].mean(),
    "Total_Stockout_Days": inventory["Stockout_Days"].sum(),
    "Records_With_Stockouts": int(
        inventory["Stockout_Days"].gt(0).sum()
    )
}])

inventory_summary.to_csv(
    REPORT_DIR / "stage_c_inventory_performance_summary.csv",
    index=False
)

print(inventory_summary.to_string(index=False))


# --------------------------------------------------
# 13. SAVE CLEANED INVENTORY
# --------------------------------------------------

print("\n13. SAVING CLEANED INVENTORY")

inventory["Date"] = inventory["Date"].dt.strftime(
    "%Y-%m"
)

output_path = OUTPUT_DIR / "inventory_stage_c.csv"

inventory.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 14. CLEANING SUMMARY
# --------------------------------------------------

summary = [
    ("Original inventory rows", original_rows),
    ("Final inventory rows", len(inventory)),
    ("Rows removed", original_rows - len(inventory)),
    ("Unique SKUs", inventory["SKU_ID"].nunique()),
    ("Unique regions", inventory["Region"].nunique()),
    ("Missing-value records", len(missing_records)),
    ("Negative-value records", len(negative_records)),
    ("Duplicate business-key rows", len(duplicates)),
    ("Duplicate business-key groups", duplicate_groups),
    ("Fulfillment reconciliation differences",
     len(reconciliation_issues)),
    ("Fill-rate discrepancies", len(fill_rate_issues)),
    ("Unknown inventory SKUs", len(unknown_skus)),
    ("Training SKUs without inventory records",
     len(missing_inventory_skus))
]

summary_df = pd.DataFrame(
    summary,
    columns=["Operation", "Count"]
)

summary_df.to_csv(
    REPORT_DIR / "stage_c_cleaning_summary.csv",
    index=False
)


# --------------------------------------------------
# 15. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 65)
print("STAGE C CLEANING COMPLETED")
print("=" * 65)

print("Original rows:", original_rows)
print("Final rows:", len(inventory))
print("Rows removed:", original_rows - len(inventory))
print("Duplicate rows:", len(duplicates))
print("Negative records:", len(negative_records))
print("Fulfillment differences:", len(reconciliation_issues))
print("Fill-rate discrepancies:", len(fill_rate_issues))

print("\nCleaned output:")
print(output_path)

print("\nInvestigation reports:")
print(REPORT_DIR)

print("=" * 65)
