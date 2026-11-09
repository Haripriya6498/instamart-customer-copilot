
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "instamart.db"

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

for table in ["customers", "products", "orders"]:
    print("\nTABLE:", table)

    cursor.execute(f"PRAGMA table_info({table})")

    for column in cursor.fetchall():
        print(column[1])

connection.close()


# Step 2: Execute all 7 SQL queries

import pandas as pd

SQL_PATH = BASE_DIR / "sql" / "queries.sql"

with sqlite3.connect(DB_PATH) as connection:

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    queries = [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]

    for number, query in enumerate(queries, start=1):

        print(f"\n========== QUERY {number} ==========")

        result = pd.read_sql_query(query, connection)

        print(result.head(5))
        print("Total rows:", len(result))

print("\nAll SQL queries executed successfully!")


# Step 3: Order statistics and pandas validation

CLEAN_DIR = BASE_DIR / "data" / "clean"

orders_df = pd.read_csv(CLEAN_DIR / "orders.csv")

# Select completed orders
completed = orders_df[
    orders_df["status"].astype(str).str.lower() == "completed"
].copy()

# Calculate statistics
amount = pd.to_numeric(
    completed["order_amount"], errors="coerce"
).dropna()

print("\n===== ORDER STATISTICS =====")

print("Total completed orders:", len(completed))
print("Total revenue:", amount.sum())
print("Mean:", amount.mean())
print("Median:", amount.median())
print("Standard deviation:", amount.std())
print("Skewness:", amount.skew())
print("Q1:", amount.quantile(0.25))
print("Q3:", amount.quantile(0.75))

# Find high-value outliers
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr

outliers = amount[amount > upper_fence]

print("Upper fence:", upper_fence)
print("High-value outliers:", len(outliers))

# Correlation analysis
print("\n===== CORRELATION =====")

print(
    completed[["quantity", "discount_pct",
               "order_amount", "delivery_minutes"]]
    .corr(numeric_only=True)
)

# Pandas and SQLite revenue cross-check
with sqlite3.connect(DB_PATH) as connection:

    sql_revenue = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS revenue
        FROM orders
        WHERE LOWER(status) = 'completed'
        """,
        connection
    )["revenue"].iloc[0]

print("\n===== VALIDATION =====")
print("Pandas revenue:", amount.sum())
print("SQLite revenue:", sql_revenue)
print("Revenue matches:", abs(amount.sum() - sql_revenue) < 0.01)
