
import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw"
OUTPUT_PATH = "outputs/reports"

os.makedirs(OUTPUT_PATH, exist_ok=True)

FILES = [
    "train.csv",
    "test.csv",
    "macro_indicators.csv",
    "sku_master.csv",
    "events_calendar.csv",
    "inventory_stockouts.csv",
    "sample_submission.csv"
]


def parse_month(value):
    """Convert different date formats into a monthly timestamp."""

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%Y-%m",
        "%m-%Y",
        "%b-%Y",
        "%B-%Y",
        "%Y/%m",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y"
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue

    return pd.to_datetime(value, errors="coerce")


def profile_file(filename):

    path = os.path.join(RAW_PATH, filename)

    df = pd.read_csv(path, low_memory=False)

    print("\n" + "=" * 70)
    print(f"DATASET: {filename}")
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # 1. Column information
    print("\nCOLUMN INFORMATION")
    print(df.dtypes)

    # 2. Missing values
    missing = pd.DataFrame({
        "File": filename,
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    print("\nMISSING VALUES")
    print(missing[missing["Missing_Count"] > 0].to_string(index=False))

    # 3. Duplicate rows
    exact_duplicates = df.duplicated().sum()

    print("\nExact duplicate rows:", exact_duplicates)

    # 4. Unique values
    print("\nUNIQUE VALUES")

    for column in df.columns:
        print(f"{column}: {df[column].nunique(dropna=True)}")

    # 5. Date analysis
    date_report = []

    if "Date" in df.columns:

        parsed_dates = df["Date"].apply(parse_month)

        print("\nDATE ANALYSIS")
        print("Original date examples:")
        print(df["Date"].dropna().astype(str).unique()[:10])

        print("Valid dates:", parsed_dates.notna().sum())
        print("Invalid dates:", parsed_dates.isna().sum())

        print("Minimum date:", parsed_dates.min())
        print("Maximum date:", parsed_dates.max())

        date_report.append({
            "File": filename,
            "Valid_Dates": parsed_dates.notna().sum(),
            "Invalid_Dates": parsed_dates.isna().sum(),
            "Minimum_Date": parsed_dates.min(),
            "Maximum_Date": parsed_dates.max()
        })

    # 6. Business key duplicates
    duplicate_report = []

    if all(col in df.columns for col in ["Date", "SKU_ID", "Region"]):

        keys = df[["Date", "SKU_ID", "Region"]].copy()
        keys["Date"] = keys["Date"].apply(parse_month)

        duplicate_count = keys.duplicated(
            subset=["Date", "SKU_ID", "Region"]
        ).sum()

        print("\nDUPLICATE BUSINESS KEYS:", duplicate_count)

        duplicate_report.append({
            "File": filename,
            "Duplicate_Business_Keys": duplicate_count
        })

    # 7. Numeric statistics
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        print("\nNUMERICAL SUMMARY")
        print(numeric_df.describe().round(2).to_string())

    # Save reports
    return (
        df,
        missing,
        exact_duplicates,
        duplicate_report,
        date_report
    )


def main():

    all_missing = []
    all_dates = []
    all_duplicates = []
    dataset_summary = []

    for filename in FILES:

        path = os.path.join(RAW_PATH, filename)

        if not os.path.exists(path):
            print(f"WARNING: {filename} not found.")
            continue

        df, missing, exact_dup, dup_report, date_report = (
            profile_file(filename)
        )

        all_missing.append(missing)
        all_duplicates.extend(dup_report)
        all_dates.extend(date_report)

        dataset_summary.append({
            "File": filename,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Exact_Duplicates": exact_dup,
            "Missing_Values": df.isnull().sum().sum()
        })

    # Save consolidated reports
    pd.concat(all_missing).to_csv(
        f"{OUTPUT_PATH}/missing_values_report.csv",
        index=False
    )

    pd.DataFrame(dataset_summary).to_csv(
        f"{OUTPUT_PATH}/dataset_summary.csv",
        index=False
    )

    pd.DataFrame(all_dates).to_csv(
        f"{OUTPUT_PATH}/date_profile_report.csv",
        index=False
    )

    pd.DataFrame(all_duplicates).to_csv(
        f"{OUTPUT_PATH}/duplicate_key_report.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("DATA PROFILING COMPLETED")
    print("=" * 70)

    print("Reports saved in:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

