
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train_stage_a.csv"
MASTER_PATH = ROOT / "data" / "raw" / "sku_master.csv"
MACRO_PATH = ROOT / "data" / "raw" / "macro_indicators.csv"

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
master = pd.read_csv(MASTER_PATH)
macro = pd.read_csv(MACRO_PATH)

print("=" * 65)
print("CELERIS - STAGE B DATA QUALITY INVESTIGATION")
print("=" * 65)

# B1. Missing values
print("\nB1. MISSING VALUES")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    print(missing.to_string())

missing.to_csv(REPORT_DIR / "stage_b_missing_values.csv")

# B2. Region values
print("\nB2. REGION VALUES")
print(train["Region"].value_counts(dropna=False).to_string())

# B3. Sales channel values
print("\nB3. SALES CHANNEL VALUES")
print(train["Sales_Channel"].value_counts(dropna=False).to_string())

# B4. SKU reconciliation
def normalize_sku(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    value = value.replace(" ", "").replace("_", "").replace("-", "")

    import re
    match = re.fullmatch(r"([A-Z]+)(\d+)", value)

    if match:
        return f"{match.group(1)}-{int(match.group(2)):03d}"

    return value


train["SKU_Normalized"] = train["SKU_ID"].apply(normalize_sku)
master["SKU_Normalized"] = master["SKU_ID"].apply(normalize_sku)

train_skus = set(train["SKU_Normalized"].dropna().unique())
master_skus = set(master["SKU_Normalized"].dropna().unique())

unknown_skus = sorted(train_skus - master_skus)
unused_master_skus = sorted(master_skus - train_skus)

print("\nB4. SKU RECONCILIATION")
print("Unique training SKUs:", len(train_skus))
print("Unique master SKUs:", len(master_skus))
print("Training SKUs not in master:", len(unknown_skus))
print("Master SKUs not in training:", len(unused_master_skus))

pd.DataFrame({"Unknown_Training_SKU": unknown_skus}).to_csv(
    REPORT_DIR / "stage_b_unknown_skus.csv", index=False
)

pd.DataFrame({"Master_SKU_Not_In_Training": unused_master_skus}).to_csv(
    REPORT_DIR / "stage_b_unused_master_skus.csv", index=False
)

# B5. Numeric and financial checks
print("\nB5. NUMERIC AND FINANCIAL CHECKS")

for column in ["Units_Sold", "Unit_Price_USD", "Revenue_USD"]:
    train[column] = pd.to_numeric(train[column], errors="coerce")
    print(f"\n{column}")
    print(train[column].describe().to_string())
    print("Missing:", train[column].isna().sum())
    print("Negative:", (train[column] < 0).sum())
    print("Zero:", (train[column] == 0).sum())

negative_revenue = train[train["Revenue_USD"] < 0]
zero_price = train[train["Unit_Price_USD"] == 0]

negative_revenue.to_csv(
    REPORT_DIR / "stage_b_negative_revenue_rows.csv", index=False
)
zero_price.to_csv(
    REPORT_DIR / "stage_b_zero_price_rows.csv", index=False
)

# B6. Duplicate business keys
print("\nB6. DUPLICATE BUSINESS KEYS")

business_key = ["Date", "SKU_ID", "Region"]
duplicate_key_mask = train.duplicated(
    subset=business_key, keep=False
)

print("Rows involved in repeated Date-SKU-Region keys:",
      duplicate_key_mask.sum())
print("Number of repeated key groups:",
      train.loc[duplicate_key_mask, business_key]
      .drop_duplicates().shape[0])

train.loc[duplicate_key_mask].sort_values(
    business_key
).to_csv(REPORT_DIR / "stage_b_duplicate_business_keys.csv", index=False)

# B7. Macro indicator checks
print("\nB7. MACRO INDICATOR CHECKS")
print("Missing values:")
print(macro.isnull().sum().to_string())

macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce")
duplicate_macro_dates = macro[
    macro.duplicated(subset=["Date"], keep=False)
].sort_values("Date")

print("\nRepeated macro dates:")
if duplicate_macro_dates.empty:
    print("None")
else:
    print(duplicate_macro_dates.to_string(index=False))

duplicate_macro_dates.to_csv(
    REPORT_DIR / "stage_b_duplicate_macro_dates.csv", index=False
)

print("\n" + "=" * 65)
print("STAGE B INVESTIGATION COMPLETED")
print("Reports saved in:", REPORT_DIR)
print("=" * 65)
