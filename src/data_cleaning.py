
import pandas as pd
import re
import os

# ============================================================
# CELERIS DEMAND FORECASTING
# STAGE A: DATA STANDARDIZATION AND DUPLICATE RESOLUTION
# ============================================================

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
REPORT_PATH = "outputs/reports"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)


# ------------------------------------------------------------
# A1. DATE STANDARDIZATION
# ------------------------------------------------------------

def parse_date(value):

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%Y-%m",
        "%Y-%m-%d",
        "%Y/%m",
        "%m/%Y",
        "%m-%Y",
        "%b %Y",
        "%b-%Y",
        "%B %Y",
        "%B-%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y"
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue

    return pd.NaT


# ------------------------------------------------------------
# A2. SKU STANDARDIZATION
# ------------------------------------------------------------

def normalize_sku(value):

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    # Remove spaces and standardize separators
    value = re.sub(r"[\s_-]+", "", value)

    # Expected format: PREFIX + numeric identifier
    match = re.fullmatch(r"([A-Z]+)(\d+)", value)

    if not match:
        return value

    prefix = match.group(1)
    number = match.group(2)

    return f"{prefix}-{int(number):03d}"


# ------------------------------------------------------------
# A3. REGION STANDARDIZATION
# ------------------------------------------------------------

def normalize_region(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z]", "", value)

    region_mapping = {

        "india": "India",

        "apac": "Asia Pacific",
        "asiapacific": "Asia Pacific",

        "eu": "Europe",
        "europe": "Europe",

        "latam": "Latin America",
        "latinamerica": "Latin America",

        "na": "North America",
        "namerica": "North America",
        "northamerica": "North America",

        "mea": "Middle East & Africa",
        "meanda": "Middle East & Africa",
        "middleeastandafrica": "Middle East & Africa"
    }

    return region_mapping.get(value, str(value))


# ------------------------------------------------------------
# A4. SALES CHANNEL STANDARDIZATION
# ------------------------------------------------------------

def normalize_channel(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]", "", value)

    channel_mapping = {

        "directoem": "Direct OEM",
        "oemdirect": "Direct OEM",

        "tier1distributor": "Tier-1 Distributor",
        "t1distributor": "Tier-1 Distributor",

        "aftermarket": "Aftermarket",

        "export": "Export"
    }

    return channel_mapping.get(value, str(value))


# ------------------------------------------------------------
# A5. MAIN CLEANING PIPELINE
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("CELERIS - STAGE A DATA CLEANING")
    print("=" * 65)

    # Read original historical dataset
    df = pd.read_csv(
        f"{RAW_PATH}/train.csv",
        dtype=str,
        low_memory=False
    )

    print("\nOriginal rows:", len(df))

    # Preserve original values
    original_columns = [
        "Date",
        "SKU_ID",
        "Region",
        "Sales_Channel",
        "Year",
        "Month",
        "Load_Timestamp"
    ]

    for col in original_columns:
        df[f"Raw_{col}"] = df[col]

    audit = []

    # --------------------------------------------------------
    # STEP A1: STANDARDIZE DATES
    # --------------------------------------------------------

    print("\nA1. Standardizing dates...")

    df["Parsed_Date"] = df["Date"].apply(parse_date)

    invalid_dates = df["Parsed_Date"].isna()

    print("Unparseable dates:", invalid_dates.sum())

    # --------------------------------------------------------
    # STEP A2: REPAIR IMPOSSIBLE DATES
    # --------------------------------------------------------

    print("\nA2. Investigating impossible dates...")

    df["Year_Numeric"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df["Month_Numeric"] = pd.to_numeric(
        df["Month"],
        errors="coerce"
    )

    impossible_date = (
        df["Parsed_Date"].notna() &
        (
            (df["Parsed_Date"].dt.year < 2019) |
            (df["Parsed_Date"].dt.year > 2026)
        )
    )

    repair_mask = (
        impossible_date &
        df["Year_Numeric"].between(2019, 2026) &
        df["Month_Numeric"].between(1, 12) &
        (
            df["Parsed_Date"].dt.month ==
            df["Month_Numeric"]
        )
    )

    for idx in df.index[repair_mask]:

        old_date = df.at[idx, "Date"]

        year = int(df.at[idx, "Year_Numeric"])
        month = int(df.at[idx, "Month_Numeric"])

        corrected_date = pd.Timestamp(
            year=year,
            month=month,
            day=1
        )

        df.at[idx, "Parsed_Date"] = corrected_date

        audit.append({
            "Row_Index": idx,
            "Record_ID": df.at[idx, "Record_ID"],
            "Operation": "Date correction",
            "Original_Value": old_date,
            "Corrected_Value": corrected_date.strftime("%Y-%m"),
            "Reason": "Impossible date year; Year and Month agree"
        })

    print("Impossible dates found:", impossible_date.sum())
    print("Dates corrected:", repair_mask.sum())

    # Save dates that cannot be safely repaired
    unresolved_dates = df[
        df["Parsed_Date"].isna() |
        (
            (df["Parsed_Date"].dt.year < 2019) |
            (df["Parsed_Date"].dt.year > 2026)
        )
    ].copy()

    unresolved_dates.to_csv(
        f"{REPORT_PATH}/stage_a_unresolved_dates.csv",
        index=False
    )

    # Standardized monthly date
    df["Date"] = df["Parsed_Date"].dt.strftime("%Y-%m")

    # Recalculate Year and Month from authoritative Date
    df["Year"] = df["Parsed_Date"].dt.year.astype("Int64")
    df["Month"] = df["Parsed_Date"].dt.month.astype("Int64")

    # --------------------------------------------------------
    # STEP A3: STANDARDIZE SKU IDS
    # --------------------------------------------------------

    print("\nA3. Standardizing SKU identifiers...")

    df["SKU_ID"] = df["SKU_ID"].apply(normalize_sku)

    # --------------------------------------------------------
    # STEP A4: STANDARDIZE REGIONS
    # --------------------------------------------------------

    print("\nA4. Standardizing regions...")

    df["Region"] = df["Region"].apply(normalize_region)

    # --------------------------------------------------------
    # STEP A5: STANDARDIZE SALES CHANNELS
    # --------------------------------------------------------

    print("\nA5. Standardizing sales channels...")

    df["Sales_Channel"] = (
        df["Sales_Channel"].apply(normalize_channel)
    )

    # --------------------------------------------------------
    # STEP A6: DUPLICATE RECORD ID RESOLUTION
    # --------------------------------------------------------

    print("\nA6. Resolving duplicate Record_ID values...")

    before_dedup = len(df)

    # Convert timestamps for chronological comparison
    df["Parsed_Timestamp"] = pd.to_datetime(
        df["Load_Timestamp"],
        errors="coerce"
    )

    # Preserve original row order as a deterministic tie-breaker
    df["Original_Row_Index"] = df.index

    # Only non-null Record_IDs participate in ID deduplication
    has_record_id = df["Record_ID"].notna()

    with_record_id = df[has_record_id].copy()
    without_record_id = df[~has_record_id].copy()

    # Latest timestamp wins. If timestamps tie, later source row wins.
    with_record_id = with_record_id.sort_values(
        ["Parsed_Timestamp", "Original_Row_Index"],
        na_position="first"
    )

    duplicate_rows = with_record_id[
        with_record_id.duplicated(
            subset=["Record_ID"],
            keep="last"
        )
    ].copy()

    duplicate_rows["Audit_Operation"] = (
        "Removed repeated Record_ID; latest timestamp retained"
    )

    duplicate_rows.to_csv(
        f"{REPORT_PATH}/stage_a_removed_record_duplicates.csv",
        index=False
    )

    with_record_id = with_record_id.drop_duplicates(
        subset=["Record_ID"],
        keep="last"
    )

    # Missing Record_ID rows are retained for later investigation
    df = pd.concat(
        [with_record_id, without_record_id],
        ignore_index=True
    )

    after_dedup = len(df)

    print("Rows before deduplication:", before_dedup)
    print("Rows after deduplication:", after_dedup)
    print("Rows removed:", before_dedup - after_dedup)

    # --------------------------------------------------------
    # STEP A7: GENERATE STANDARDIZATION REPORT
    # --------------------------------------------------------

    print("\nA7. Generating reports...")

    # Standardization counts
    summary = pd.DataFrame([
        {
            "Operation": "Date corrections",
            "Count": int(repair_mask.sum())
        },
        {
            "Operation": "SKU standardization changes",
            "Count": int(
                (df["Raw_SKU_ID"] != df["SKU_ID"]).sum()
            )
        },
        {
            "Operation": "Region standardization changes",
            "Count": int(
                (
                    df["Raw_Region"].fillna("<MISSING>") !=
                    df["Region"].fillna("<MISSING>")
                ).sum()
            )
        },
        {
            "Operation": "Channel standardization changes",
            "Count": int(
                (
                    df["Raw_Sales_Channel"].fillna("<MISSING>") !=
                    df["Sales_Channel"].fillna("<MISSING>")
                ).sum()
            )
        },
        {
            "Operation": "Repeated Record_ID rows removed",
            "Count": before_dedup - after_dedup
        }
    ])

    summary.to_csv(
        f"{REPORT_PATH}/stage_a_summary.csv",
        index=False
    )

    pd.DataFrame(audit).to_csv(
        f"{REPORT_PATH}/stage_a_audit_log.csv",
        index=False
    )

    # --------------------------------------------------------
    # STEP A8: SAVE PROCESSED DATA
    # --------------------------------------------------------

    df = df.sort_values(
        ["Date", "SKU_ID", "Region"],
        na_position="last"
    )

    output_file = (
        f"{PROCESSED_PATH}/train_stage_a.csv"
    )

    df.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STAGE A COMPLETED")
    print("=" * 65)

    print("Original rows:", 248248)
    print("Final rows:", len(df))
    print("Removed Record_ID duplicates:", before_dedup - after_dedup)

    print("Unique SKUs:", df["SKU_ID"].nunique())
    print("Unique regions:", df["Region"].nunique())
    print("Unique channels:", df["Sales_Channel"].nunique())

    print("\nFinal dates:")
    print("Start:", df["Parsed_Date"].min())
    print("End:", df["Parsed_Date"].max())

    print("\nProcessed file:", output_file)
    print("Reports:", REPORT_PATH)

    print("\nStage A execution finished.")


if __name__ == "__main__":
    main()
