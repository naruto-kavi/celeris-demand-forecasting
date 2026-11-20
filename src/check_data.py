
import pandas as pd
import os

DATA_PATH = "data/raw"

files = [
    "train.csv",
    "test.csv",
    "macro_indicators.csv",
    "sku_master.csv",
    "events_calendar.csv",
    "inventory_stockouts.csv",
    "sample_submission.csv"
]

print("=" * 60)
print("CELERIS DEMAND FORECASTING - DATA CHECK")
print("=" * 60)

for file in files:
    path = os.path.join(DATA_PATH, file)

    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\nFile: {file}")
        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("Missing values:", df.isnull().sum().sum())
    else:
        print(f"\nMISSING FILE: {file}")

print("\nData verification completed.")
