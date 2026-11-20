
import pandas as pd
import os

RAW = "data/raw"
REPORTS = "outputs/reports"

os.makedirs(REPORTS, exist_ok=True)

train = pd.read_csv(
    f"{RAW}/train.csv",
    dtype=str,
    low_memory=False
)

test = pd.read_csv(f"{RAW}/test.csv", dtype=str)
master = pd.read_csv(f"{RAW}/sku_master.csv", dtype=str)
macro = pd.read_csv(f"{RAW}/macro_indicators.csv", dtype=str)


def save_report(df, filename):
    path = f"{REPORTS}/{filename}"
    df.to_csv(path, index=False)
    print(f"Saved: {path} | Rows: {len(df)}")


print("\n" + "=" * 65)
print("1. INVESTIGATING DATE ANOMALIES")
print("=" * 65)

# Use the raw date strings first. Do not silently correct them.
train["Parsed_Date"] = pd.to_datetime(
    train["Date"].str.strip(),
    errors="coerce",
    format="mixed",
    dayfirst=False
)

print("\nOriginal date examples:")
print(train["Date"].dropna().unique()[:30])

print("\nMost frequent date values:")
print(train["Date"].value_counts().head(20))

# Dates outside the expected historical window
bad_dates = train[
    (train["Parsed_Date"] < "2019-01-01") |
    (train["Parsed_Date"] > "2026-10-31") |
    (train["Parsed_Date"].isna())
].copy()

print("\nUnexpected date rows:", len(bad_dates))

print(
    bad_dates[
        ["Date", "Year", "Month", "SKU_ID", "Region"]
    ].head(30).to_string(index=False)
)

save_report(bad_dates, "suspicious_dates.csv")


# Compare Date with Year and Month
valid = train["Parsed_Date"].notna()

date_mismatch = train.loc[valid].copy()

date_mismatch["Parsed_Year"] = (
    date_mismatch["Parsed_Date"].dt.year.astype(str)
)

date_mismatch["Parsed_Month"] = (
    date_mismatch["Parsed_Date"].dt.month.astype(str)
)

date_mismatch["Year_Match"] = (
    date_mismatch["Year"].str.strip() ==
    date_mismatch["Parsed_Year"]
)

date_mismatch["Month_Match"] = (
    date_mismatch["Month"].str.strip() ==
    date_mismatch["Parsed_Month"]
)

date_mismatch = date_mismatch[
    (~date_mismatch["Year_Match"]) |
    (~date_mismatch["Month_Match"])
]

print("\nYear/Month mismatches:", len(date_mismatch))

save_report(date_mismatch, "date_year_month_mismatches.csv")


print("\n" + "=" * 65)
print("2. SKU MASTER RECONCILIATION")
print("=" * 65)

train_skus = set(train["SKU_ID"].dropna().str.strip())
master_skus = set(master["SKU_ID"].dropna().str.strip())
test_skus = set(test["SKU_ID"].dropna().str.strip())

unknown_skus = train_skus - master_skus

print("Historical unique SKUs:", len(train_skus))
print("Master unique SKUs:", len(master_skus))
print("Test unique SKUs:", len(test_skus))
print("Historical SKUs absent from master:", len(unknown_skus))

unknown_sku_df = pd.DataFrame({
    "SKU_ID": sorted(unknown_skus)
})

print("\nUnknown SKU examples:")
print(unknown_sku_df.head(30).to_string(index=False))

save_report(unknown_sku_df, "unknown_skus.csv")


print("\n" + "=" * 65)
print("3. REGION LABEL INVESTIGATION")
print("=" * 65)

region_counts = (
    train["Region"]
    .value_counts(dropna=False)
    .reset_index()
)

region_counts.columns = ["Region", "Row_Count"]

print(region_counts.to_string(index=False))

save_report(region_counts, "region_label_counts.csv")


print("\n" + "=" * 65)
print("4. SALES CHANNEL INVESTIGATION")
print("=" * 65)

channel_counts = (
    train["Sales_Channel"]
    .value_counts(dropna=False)
    .reset_index()
)

channel_counts.columns = ["Sales_Channel", "Row_Count"]

print(channel_counts.to_string(index=False))

save_report(channel_counts, "channel_label_counts.csv")


print("\n" + "=" * 65)
print("5. DUPLICATE INVESTIGATION")
print("=" * 65)

# Duplicate Record_ID values
record_duplicates = train[
    train.duplicated(subset=["Record_ID"], keep=False)
].sort_values(["Record_ID", "Load_Timestamp"])

print(
    "Rows involved in repeated Record_ID values:",
    len(record_duplicates)
)

save_report(record_duplicates, "duplicate_record_ids.csv")


# Duplicate business keys
business_duplicates = train[
    train.duplicated(
        subset=["Date", "SKU_ID", "Region"],
        keep=False
    )
].sort_values(["Date", "SKU_ID", "Region"])

print(
    "Rows involved in repeated business keys:",
    len(business_duplicates)
)

save_report(business_duplicates, "duplicate_business_keys.csv")


print("\n" + "=" * 65)
print("6. MISSING SALES QUANTITIES")
print("=" * 65)

missing_units = train[
    train["Units_Sold"].isna() |
    (train["Units_Sold"].str.strip() == "")
]

print("Missing Units_Sold rows:", len(missing_units))

print(
    missing_units[
        ["Date", "SKU_ID", "Region", "Units_Sold"]
    ].head(20).to_string(index=False)
)

save_report(missing_units, "missing_units_sold.csv")


print("\n" + "=" * 65)
print("7. PRICE AND REVENUE INVESTIGATION")
print("=" * 65)

for col in ["Units_Sold", "Unit_Price_USD", "Revenue_USD"]:
    train[col + "_Numeric"] = pd.to_numeric(
        train[col].str.replace(",", "", regex=False),
        errors="coerce"
    )

negative_revenue = train[
    train["Revenue_USD_Numeric"] < 0
]

zero_price = train[
    train["Unit_Price_USD_Numeric"] == 0
]

negative_price = train[
    train["Unit_Price_USD_Numeric"] < 0
]

print("Negative revenue rows:", len(negative_revenue))
print("Zero unit price rows:", len(zero_price))
print("Negative unit price rows:", len(negative_price))

save_report(negative_revenue, "negative_revenue.csv")
save_report(zero_price, "zero_price.csv")
save_report(negative_price, "negative_price.csv")


print("\n" + "=" * 65)
print("8. MACRO DATE DUPLICATES")
print("=" * 65)

macro_duplicates = macro[
    macro.duplicated(subset=["Date"], keep=False)
].sort_values("Date")

print("Rows with repeated macro dates:", len(macro_duplicates))

print(macro_duplicates.to_string(index=False))

save_report(macro_duplicates, "macro_duplicate_dates.csv")


print("\n" + "=" * 65)
print("INVESTIGATION COMPLETED")
print("Check outputs/reports for all generated files.")
print("=" * 65)
