
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

inventory_path = ROOT / "data" / "raw" / "inventory_stockouts.csv"
train_path = ROOT / "data" / "processed" / "train_stage_b.csv"
reports_dir = ROOT / "outputs" / "reports"

reports_dir.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CELERIS - STAGE C INVENTORY AND STOCKOUT INVESTIGATION")
print("=" * 65)

inventory = pd.read_csv(inventory_path)
train = pd.read_csv(train_path)

print("\n1. DATASET OVERVIEW")
print("Inventory rows:", len(inventory))
print("Training rows:", len(train))
print("\nInventory columns:")
print(inventory.columns.tolist())

print("\n2. MISSING VALUES")
print(inventory.isna().sum())

print("\n3. DATE RANGE")
inventory["Date"] = pd.to_datetime(inventory["Date"], errors="coerce")
print("Start:", inventory["Date"].min())
print("End:", inventory["Date"].max())

print("\n4. UNIQUE COUNTS")
for col in ["SKU_ID", "Region", "Reason"]:
    if col in inventory.columns:
        print(f"\n{col}:")
        print(inventory[col].value_counts(dropna=False))

print("\n5. NUMERIC SUMMARY")
numeric_columns = [
    "Requested_Units",
    "Carried_In_Units",
    "Shipped_Units",
    "Backordered_Units",
    "Lost_Units",
    "Fill_Rate_Pct",
    "Stockout_Days"
]

for col in numeric_columns:
    if col in inventory.columns:
        inventory[col] = pd.to_numeric(
            inventory[col], errors="coerce"
        )

print(inventory[numeric_columns].describe().T)

print("\n6. DEMAND FULFILLMENT COMPARISON")

if {"Requested_Units", "Shipped_Units"}.issubset(inventory.columns):
    inventory["Unshipped_Units"] = (
        inventory["Requested_Units"] - inventory["Shipped_Units"]
    )

    print("Total requested units:",
          inventory["Requested_Units"].sum())

    print("Total shipped units:",
          inventory["Shipped_Units"].sum())

    print("Total unshipped units:",
          inventory["Unshipped_Units"].sum())

    print("Records where requested > shipped:",
          (inventory["Requested_Units"] >
           inventory["Shipped_Units"]).sum())

print("\n7. STOCKOUT IMPACT")

if "Stockout_Days" in inventory.columns:
    print("Records with stockout days > 0:",
          (inventory["Stockout_Days"] > 0).sum())

    print("Total stockout days:",
          inventory["Stockout_Days"].sum())

if "Lost_Units" in inventory.columns:
    print("Total lost units:", inventory["Lost_Units"].sum())

if "Backordered_Units" in inventory.columns:
    print("Total backordered units:",
          inventory["Backordered_Units"].sum())

print("\n8. FILL RATE CHECK")

if "Fill_Rate_Pct" in inventory.columns:
    print("\nFill rate distribution:")
    print(inventory["Fill_Rate_Pct"].describe())

    print("\nFill rate below 100%:",
          (inventory["Fill_Rate_Pct"] < 100).sum())

print("\n9. SKU AND REGION COVERAGE")

print("Inventory unique SKUs:",
      inventory["SKU_ID"].nunique())

print("Training unique SKUs:",
      train["SKU_ID"].nunique())

print("\nInventory regions:")
print(inventory["Region"].value_counts(dropna=False))

print("\n10. SAVING REPORTS")

inventory.to_csv(
    reports_dir / "stage_c_inventory_review.csv",
    index=False
)

if "Unshipped_Units" in inventory.columns:
    inventory[inventory["Unshipped_Units"] > 0].to_csv(
        reports_dir / "stage_c_unshipped_records.csv",
        index=False
    )

if "Stockout_Days" in inventory.columns:
    inventory[inventory["Stockout_Days"] > 0].to_csv(
        reports_dir / "stage_c_stockout_records.csv",
        index=False
    )

print("Reports saved to:", reports_dir)

print("\n" + "=" * 65)
print("STAGE C INVESTIGATION COMPLETED")
print("=" * 65)
