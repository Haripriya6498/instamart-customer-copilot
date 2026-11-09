
import pandas as pd
from pathlib import Path

# Find the raw data folder
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Read the main datasets
customers = pd.read_csv(RAW_DIR / "customers.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

# Display dataset information
print("CUSTOMERS")
print(customers.shape)
print(customers.head())

print("\nORDERS")
print(orders.shape)
print(orders.head())

print("\nPRODUCTS")
print(products.shape)
print(products.head())


# Step 2: Audit the datasets

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Products": products
}

for name, df in datasets.items():
    print("\n==============================")
    print(name.upper(), "DATA AUDIT")
    print("==============================")

    print("Rows and columns:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)


# Step 3: Remove duplicate rows

print("\n--- Removing Duplicates ---")

customers = customers.drop_duplicates()
orders = orders.drop_duplicates()
products = products.drop_duplicates()

print("Customers after cleaning:", customers.shape)
print("Orders after cleaning:", orders.shape)
print("Products after cleaning:", products.shape)

print("\nDuplicate removal completed!")


# Step 4: Handle missing and invalid values

# Clean customer data
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

customers["age"] = customers["age"].fillna(
    customers["age"].median()
)

customers["gender"] = customers["gender"].fillna("Unknown")

# Clean order amounts
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Clean delivery times
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Negative delivery times are invalid
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Save cleaned datasets
CLEAN_DIR = BASE_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)

print("\nCleaned datasets saved successfully!")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Products:", products.shape)


# Step 5: Create SQLite database

import sqlite3

DB_PATH = BASE_DIR / "data" / "instamart.db"

connection = sqlite3.connect(DB_PATH)

customers.to_sql(
    "customers", connection, if_exists="replace", index=False
)

products.to_sql(
    "products", connection, if_exists="replace", index=False
)

orders.to_sql(
    "orders", connection, if_exists="replace", index=False
)

connection.close()

print("\nSQLite database created successfully!")
print("Database location:", DB_PATH)
