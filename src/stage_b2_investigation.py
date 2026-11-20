
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ROOT / "data" / "processed" / "train_stage_a.csv"
MASTER_PATH = ROOT / "data" / "raw" / "sku_master.csv"
MACRO_PATH = ROOT / "data" / "raw" / "macro_indicators.csv"

# Read as strings so original values are not silently converted.
train = pd.read_csv(TRAIN_PATH, dtype=str)
master = pd.read_csv(MASTER_PATH, dtype=str)
macro = pd.read_csv(MACRO_PATH)

def normalize_sku(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    value = value.replace(" ", "").replace("_", "").replace("-", "")

    match = re.fullmatch(r"([A-Z]+)(\d+)", value)

    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"

    return value

print("=" * 65)
print("CELERIS - STAGE B2 INVESTIGATION")
print("=" * 65)

# -------------------------------------------------
# B2.1 Recoverable product information
# -------------------------------------------------

print("\nB2.1 PRODUCT INFORMATION")

train["SKU_Normalized"] = train["SKU_ID"].apply(normalize_sku)
master["SKU_Normalized"] = master["SKU_ID"].apply(normalize_sku)

# Build lookup from master data.
master_lookup = master.drop_duplicates("SKU_Normalized").set_index(
    "SKU_Normalized"
)

for column in ["Product_Family", "SKU_Description"]:
    master_values = train["SKU_Normalized"].map(master_lookup[column])
    missing_mask = train[column].isna() | train[column].str.strip().eq("")

    recoverable = missing_mask & master_values.notna()

    print(f"{column}:")
    print("  Missing in training:", int(missing_mask.sum()))
    print("  Recoverable from master:", int(recoverable.sum()))
    print("  Not recoverable:", int((missing_mask & master_values.isna()).sum()))

# Save records where product details can be recovered.
product_recovery = train[
    train["Product_Family"].isna()
    | train["SKU_Description"].isna()
].copy()

product_recovery["Master_Product_Family"] = product_recovery[
    "SKU_Normalized"
].map(master_lookup["Product_Family"])

product_recovery["Master_SKU_Description"] = product_recovery[
    "SKU_Normalized"
].map(master_lookup["SKU_Description"])

product_recovery.to_csv(
    REPORT_DIR / "stage_b2_product_recovery_candidates.csv",
    index=False
)

# -------------------------------------------------
# B2.2 Units sold conversion investigation
# -------------------------------------------------

print("\nB2.2 UNITS SOLD VALUE INVESTIGATION")

raw_units = train["Units_Sold"]
numeric_units = pd.to_numeric(raw_units, errors="coerce")

original_missing = raw_units.isna() | raw_units.str.strip().eq("")
invalid_units = (
    ~original_missing
    & numeric_units.isna()
)

print("Originally missing:", int(original_missing.sum()))
print("Non-empty but non-numeric:", int(invalid_units.sum()))
print("Total missing after numeric conversion:", int(numeric_units.isna().sum()))

invalid_values = train.loc[invalid_units, ["Record_ID", "Date", "SKU_ID", "Units_Sold"]]

print("\nInvalid Units_Sold examples:")
print(invalid_values.head(30).to_string(index=False))

invalid_values.to_csv(
    REPORT_DIR / "stage_b2_invalid_units_values.csv",
    index=False
)

# -------------------------------------------------
# B2.3 Zero and negative sales analysis
# -------------------------------------------------

print("\nB2.3 ZERO AND NEGATIVE SALES")

price = pd.to_numeric(train["Unit_Price_USD"], errors="coerce")
revenue = pd.to_numeric(train["Revenue_USD"], errors="coerce")

zero_price_mask = price.eq(0)
zero_units_mask = numeric_units.eq(0)
negative_units_mask = numeric_units.lt(0)
negative_revenue_mask = revenue.lt(0)

print("Zero price + zero units:",
      int((zero_price_mask & zero_units_mask).sum()))

print("Zero price + nonzero units:",
      int((zero_price_mask & numeric_units.notna() & ~zero_units_mask).sum()))

print("Zero price + missing units:",
      int((zero_price_mask & numeric_units.isna()).sum()))

print("Negative units:", int(negative_units_mask.sum()))
print("Negative revenue:", int(negative_revenue_mask.sum()))

negative_records = train[
    negative_units_mask | negative_revenue_mask
].copy()

negative_records.to_csv(
    REPORT_DIR / "stage_b2_negative_sales_records.csv",
    index=False
)

zero_price_records = train[zero_price_mask].copy()
zero_price_records.to_csv(
    REPORT_DIR / "stage_b2_zero_price_records.csv",
    index=False
)

# -------------------------------------------------
# B2.4 Region alias investigation
# -------------------------------------------------

print("\nB2.4 REGION STANDARDIZATION")

print("Current region values:")
print(train["Region"].value_counts(dropna=False).to_string())

# Preserve the original region field; only report proposed mappings.
region_aliases = {
    "india": "India",
    "europe": "Europe",
    "asiapacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "latinamerica": "Latin America",
    "latam": "Latin America",
    "northamerica": "North America",
    "namerica": "North America",
    "na": "North America",
    "middleeastafrica": "Middle East & Africa",
    "middleeastandafrica": "Middle East & Africa",
    "mea": "Middle East & Africa",
    "meanda": "Middle East & Africa",
}

def region_key(value):
    if pd.isna(value):
        return None
    return re.sub(r"[^a-z]", "", str(value).lower())

region_report = train[["Region", "Raw_Region"]].copy()
region_report["Normalized_Key"] = region_report["Region"].apply(region_key)
region_report["Proposed_Region"] = region_report["Normalized_Key"].map(
    region_aliases
)

region_report.to_csv(
    REPORT_DIR / "stage_b2_region_mapping_review.csv",
    index=False
)

# -------------------------------------------------
# B2.5 Missing macro indicator rows
# -------------------------------------------------

print("\nB2.5 MACRO MISSING VALUES")

macro_missing = macro[macro.isna().any(axis=1)].copy()

print(macro_missing.to_string(index=False))

macro_missing.to_csv(
    REPORT_DIR / "stage_b2_macro_missing_rows.csv",
    index=False
)

# -------------------------------------------------
# B2.6 Final summary
# -------------------------------------------------

print("\n" + "=" * 65)
print("STAGE B2 INVESTIGATION COMPLETED")
print("Reports saved in:", REPORT_DIR)
print("=" * 65)
