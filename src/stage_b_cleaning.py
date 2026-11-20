
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train_stage_a.csv"
MASTER_PATH = ROOT / "data" / "raw" / "sku_master.csv"
MACRO_PATH = ROOT / "data" / "raw" / "macro_indicators.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CELERIS - STAGE B DATA CLEANING")
print("=" * 65)


# -------------------------------------------------
# 1. LOAD SOURCE FILES
# -------------------------------------------------

train = pd.read_csv(TRAIN_PATH, dtype=str)
master = pd.read_csv(MASTER_PATH, dtype=str)
macro = pd.read_csv(MACRO_PATH)

original_rows = len(train)

print("\n1. SOURCE DATA")
print("Training rows:", original_rows)
print("Master rows:", len(master))
print("Macro rows:", len(macro))

required_train_columns = [
    "SKU_ID",
    "Region",
    "Product_Family",
    "SKU_Description",
    "Units_Sold",
    "Unit_Price_USD",
    "Revenue_USD",
]

missing_columns = [
    column for column in required_train_columns
    if column not in train.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required training columns: {missing_columns}"
    )


# -------------------------------------------------
# 2. NORMALIZE SKU IDS FOR MASTER MATCHING
# -------------------------------------------------

def normalize_sku(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    value = value.replace(" ", "").replace("_", "").replace("-", "")

    match = re.fullmatch(r"([A-Z]+)(\d+)", value)

    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"

    return value


train["SKU_Normalized"] = train["SKU_ID"].apply(normalize_sku)
master["SKU_Normalized"] = master["SKU_ID"].apply(normalize_sku)

master_lookup = (
    master.drop_duplicates("SKU_Normalized")
    .set_index("SKU_Normalized")
)


# -------------------------------------------------
# 3. RECOVER MISSING PRODUCT INFORMATION
# -------------------------------------------------

print("\n2. RECOVERING PRODUCT INFORMATION")

product_recovery_counts = {}

for column in ["Product_Family", "SKU_Description"]:

    master_column = train["SKU_Normalized"].map(
        master_lookup[column]
    )

    missing_mask = (
        train[column].isna()
        | train[column].str.strip().eq("")
    )

    recover_mask = missing_mask & master_column.notna()

    product_recovery_counts[column] = int(recover_mask.sum())

    train.loc[recover_mask, column] = master_column[recover_mask]

    print(
        f"{column}: filled {product_recovery_counts[column]} values"
    )


# -------------------------------------------------
# 4. STANDARDIZE REGION NAMES
# -------------------------------------------------

print("\n3. STANDARDIZING REGIONS")


def normalize_region(value):

    if pd.isna(value) or str(value).strip() == "":
        return pd.NA

    key = re.sub(r"[^a-z]", "", str(value).lower())

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
        "meanda": "Middle East & Africa",
        "middleeastafrica": "Middle East & Africa",
        "middleeastandafrica": "Middle East & Africa",
    }

    return mapping.get(key, str(value).strip())


train["Region"] = train["Region"].apply(normalize_region)

print(train["Region"].value_counts(dropna=False).to_string())


# -------------------------------------------------
# 5. CONVERT NUMERIC COLUMNS SAFELY
# -------------------------------------------------

print("\n4. CONVERTING NUMERIC COLUMNS")

numeric_columns = [
    "Units_Sold",
    "Unit_Price_USD",
    "Revenue_USD",
]

conversion_report = []

# Preserve the original target text for audit purposes.
original_units_text = train["Units_Sold"].copy()

for column in numeric_columns:

    original = train[column].copy()

    cleaned = (
        original.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({
            "-": pd.NA,
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "null": pd.NA,
        })
    )

    converted = pd.to_numeric(cleaned, errors="coerce")

    conversion_report.append({
        "Column": column,
        "Missing_Before": int(original.isna().sum()),
        "Missing_After": int(converted.isna().sum()),
        "Newly_Missing": int(
            (original.notna() & converted.isna()).sum()
        ),
    })

    train[column] = converted

    print(
        f"{column}: missing after conversion = "
        f"{converted.isna().sum()}"
    )

pd.DataFrame(conversion_report).to_csv(
    REPORT_DIR / "stage_b_numeric_conversion_report.csv",
    index=False
)


# -------------------------------------------------
# 6. RECOVER MISSING UNITS SOLD
# -------------------------------------------------

print("\n5. RECOVERING MISSING DEMAND VALUES")

# Track whether a target was recovered.
train["Units_Sold_Imputed"] = False

train["Units_Sold_Imputation_Method"] = "original"

# Identify missing targets after numeric conversion.
missing_units_mask = train["Units_Sold"].isna()

# Revenue / price recovery is permitted only when:
# 1. Units_Sold is missing.
# 2. Unit_Price_USD is positive.
# 3. Revenue_USD is available.
# 4. Estimated units are within 0.02 of a whole number.

eligible_mask = (
    missing_units_mask
    & train["Unit_Price_USD"].gt(0)
    & train["Revenue_USD"].notna()
)

estimated_units = pd.Series(
    np.nan,
    index=train.index,
    dtype="float64"
)

estimated_units.loc[eligible_mask] = (
    train.loc[eligible_mask, "Revenue_USD"]
    / train.loc[eligible_mask, "Unit_Price_USD"]
)

nearest_integer = estimated_units.round()

integer_match_mask = (
    eligible_mask
    & (estimated_units - nearest_integer).abs().le(0.02)
)

# Recover only validated integer estimates.
train.loc[integer_match_mask, "Units_Sold"] = (
    nearest_integer.loc[integer_match_mask]
)

train.loc[integer_match_mask, "Units_Sold_Imputed"] = True

train.loc[
    integer_match_mask,
    "Units_Sold_Imputation_Method"
] = "recovered_from_revenue_price"

# Identify unresolved target records.
unresolved_units_mask = train["Units_Sold"].isna()

train.loc[
    unresolved_units_mask,
    "Units_Sold_Imputation_Method"
] = "unresolved_missing"

recovered_count = int(integer_match_mask.sum())
unresolved_count = int(unresolved_units_mask.sum())

print("Missing units before recovery:", int(missing_units_mask.sum()))
print("Eligible for revenue/price recovery:", int(eligible_mask.sum()))
print("Successfully recovered:", recovered_count)
print("Still unresolved:", unresolved_count)
print(
    "Missing units after recovery:",
    int(train["Units_Sold"].isna().sum())
)

# Save recovered records for audit.
recovered_report = train.loc[integer_match_mask].copy()

recovered_report["Calculated_Units"] = (
    estimated_units.loc[integer_match_mask]
)

recovered_report["Rounded_Units"] = (
    nearest_integer.loc[integer_match_mask]
)

recovered_report.to_csv(
    REPORT_DIR / "stage_b_recovered_units_report.csv",
    index=False
)

# Save unresolved records for further investigation.
unresolved_report = train.loc[unresolved_units_mask].copy()

unresolved_report.to_csv(
    REPORT_DIR / "stage_b_unresolved_units_report.csv",
    index=False
)

# Save original target values that could not be parsed,
# for checking unexpected text formats.
invalid_target_mask = (
    original_units_text.notna()
    & train["Units_Sold"].isna()
    & ~train["Units_Sold_Imputed"]
)

invalid_target_report = train.loc[invalid_target_mask].copy()

invalid_target_report["Original_Units_Sold_Text"] = (
    original_units_text.loc[invalid_target_mask]
)

invalid_target_report.to_csv(
    REPORT_DIR / "stage_b_invalid_units_text_report.csv",
    index=False
)


# -------------------------------------------------
# 7. SAVE OTHER INVESTIGATION REPORTS
# -------------------------------------------------

print("\n6. SAVING INVESTIGATION REPORTS")

zero_price = train[
    train["Unit_Price_USD"].eq(0)
].copy()

negative_sales = train[
    train["Units_Sold"].lt(0)
    | train["Revenue_USD"].lt(0)
].copy()

missing_region = train[
    train["Region"].isna()
].copy()

zero_price.to_csv(
    REPORT_DIR / "stage_b_zero_price_review.csv",
    index=False
)

negative_sales.to_csv(
    REPORT_DIR / "stage_b_negative_sales_review.csv",
    index=False
)

missing_region.to_csv(
    REPORT_DIR / "stage_b_missing_region_review.csv",
    index=False
)


# -------------------------------------------------
# 8. SAVE CLEANED TRAINING DATA
# -------------------------------------------------

train_output = OUTPUT_DIR / "train_stage_b.csv"

train.to_csv(train_output, index=False)

print("\nTraining data saved:", train_output)


# -------------------------------------------------
# 9. CLEAN MACRO INDICATORS
# -------------------------------------------------

print("\n7. CLEANING MACRO INDICATORS")

macro["Date"] = pd.to_datetime(
    macro["Date"].astype(str),
    format="%Y-%m",
    errors="coerce"
)

if macro["Date"].isna().any():
    raise ValueError("Some macro dates could not be parsed.")

duplicate_macro = macro[
    macro.duplicated("Date", keep=False)
].copy()

duplicate_macro.to_csv(
    REPORT_DIR / "stage_b_macro_duplicate_source_rows.csv",
    index=False
)

# Average duplicate monthly observations.
macro_numeric_columns = macro.select_dtypes(
    include="number"
).columns.tolist()

macro_clean = (
    macro.groupby("Date", as_index=False)[macro_numeric_columns]
    .mean()
)

macro_clean = macro_clean.sort_values("Date")

# Set monthly date index and interpolate missing indicators.
macro_clean = macro_clean.set_index("Date")

indicator_columns = [
    column for column in macro_clean.columns
    if column not in ["Year", "Month"]
]

macro_clean[indicator_columns] = (
    macro_clean[indicator_columns]
    .interpolate(method="time", limit_area="inside")
)

macro_clean["Year"] = macro_clean.index.year
macro_clean["Month"] = macro_clean.index.month

macro_clean = macro_clean.reset_index()

macro_clean["Date"] = (
    macro_clean["Date"].dt.strftime("%Y-%m")
)

remaining_macro_missing = (
    macro_clean[indicator_columns].isna().sum()
)

print("Duplicate source rows:", len(duplicate_macro))
print("Macro rows after duplicate resolution:", len(macro_clean))
print("\nRemaining macro missing values:")
print(remaining_macro_missing.to_string())

macro_output = OUTPUT_DIR / "macro_stage_b.csv"

macro_clean.to_csv(
    macro_output,
    index=False
)


# -------------------------------------------------
# 10. STAGE B SUMMARY
# -------------------------------------------------

summary = [
    ("Training rows before cleaning", original_rows),
    ("Training rows after cleaning", len(train)),
    ("Training rows removed", original_rows - len(train)),
    (
        "Product families recovered",
        product_recovery_counts["Product_Family"]
    ),
    (
        "SKU descriptions recovered",
        product_recovery_counts["SKU_Description"]
    ),
    (
        "Missing units before recovery",
        int(missing_units_mask.sum())
    ),
    (
        "Units recovered using revenue/price",
        recovered_count
    ),
    (
        "Units still unresolved",
        unresolved_count
    ),
    (
        "Missing units after recovery",
        int(train["Units_Sold"].isna().sum())
    ),
    (
        "Missing revenue after conversion",
        int(train["Revenue_USD"].isna().sum())
    ),
    (
        "Missing regions",
        int(train["Region"].isna().sum())
    ),
    (
        "Negative sales records",
        len(negative_sales)
    ),
    (
        "Zero-price records",
        len(zero_price)
    ),
    (
        "Duplicate macro source rows",
        len(duplicate_macro)
    ),
    (
        "Macro rows after duplicate resolution",
        len(macro_clean)
    ),
]

pd.DataFrame(
    summary,
    columns=["Operation", "Count"]
).to_csv(
    REPORT_DIR / "stage_b_cleaning_summary.csv",
    index=False
)


# -------------------------------------------------
# 11. FINAL VALIDATION
# -------------------------------------------------

print("\n8. FINAL VALIDATION")

if len(train) != original_rows:
    raise ValueError(
        "Training row count changed unexpectedly."
    )

if train["Units_Sold"].isna().sum() != unresolved_count:
    raise ValueError(
        "Unresolved target count does not match validation."
    )

if not train.loc[
    train["Units_Sold_Imputed"],
    "Units_Sold"
].notna().all():
    raise ValueError(
        "Some imputed target records still contain missing values."
    )

print("Row count preserved:", len(train))
print("Imputed target records:", recovered_count)
print("Unresolved target records:", unresolved_count)
print(
    "Negative target records retained:",
    int(train["Units_Sold"].lt(0).sum())
)

print("\n" + "=" * 65)
print("STAGE B CLEANING COMPLETED")
print("=" * 65)
print("Training rows:", len(train))
print("Training output:", train_output)
print("Macro output:", macro_output)
print("Reports:", REPORT_DIR)
print("=" * 65)
