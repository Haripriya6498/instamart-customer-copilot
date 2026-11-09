
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned datasets
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Convert columns to correct data types
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Select completed orders
completed = orders[
    orders["status"].astype(str).str.lower() == "completed"
].copy()

completed = completed.dropna(
    subset=["customer_id", "order_ts", "order_amount"]
)

# Reference date
reference_date = (
    completed["order_ts"].max().normalize()
    + pd.Timedelta(days=1)
)

# Calculate RFM
rfm = completed.groupby("customer_id").agg(
    Recency=("order_ts",
             lambda x: (reference_date - x.max().normalize()).days),
    Frequency=("order_id", "nunique"),
    Monetary=("order_amount", "sum")
).reset_index()

# Save RFM results
rfm.to_csv(OUTPUT_DIR / "customer_rfm.csv", index=False)

print("RFM calculation completed!")
print(rfm.head(10))
print("\nTotal customers:", len(rfm))
print("\nRFM summary:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())


# Step 2: K-Means Customer Segmentation

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Select RFM features
features = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Log transform to reduce skewness
import numpy as np

features["Frequency"] = np.log1p(features["Frequency"])
features["Monetary"] = np.log1p(features["Monetary"])

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Test different K values from 2 to 8
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, labels)

    print(f"K={k}, Silhouette Score={score:.4f}")

# Select K=4 for final segmentation
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(scaled_features)

# Save customer segments
rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

print("\nCustomer segmentation completed!")
print(rfm["Cluster"].value_counts())


# Step 3: Name customer segments

# Calculate average RFM values for each cluster
cluster_summary = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    Customers=("customer_id", "count")
)

print("\n===== CLUSTER SUMMARY =====")
print(cluster_summary)

# Rank clusters based on RFM behaviour
cluster_summary["R_Score"] = cluster_summary["Recency"].rank(
    ascending=False
)
cluster_summary["F_Score"] = cluster_summary["Frequency"].rank(
    ascending=True
)
cluster_summary["M_Score"] = cluster_summary["Monetary"].rank(
    ascending=True
)

# Assign descriptive names based on cluster characteristics
segment_names = {}

for cluster, row in cluster_summary.iterrows():
    if row["Frequency"] >= cluster_summary["Frequency"].median() and \
       row["Monetary"] >= cluster_summary["Monetary"].median():
        name = "Loyal"
    elif row["Recency"] >= cluster_summary["Recency"].median():
        name = "At Risk"
    else:
        name = "Regular"

    segment_names[cluster] = name

rfm["Segment"] = rfm["Cluster"].map(segment_names)

# Save updated customer segments
rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

print("\n===== CUSTOMER SEGMENTS =====")
print(rfm["Segment"].value_counts())

print("\nCustomer segment names saved successfully!")


# Step 4: Principal Component Analysis (PCA)

from sklearn.decomposition import PCA

# Reduce RFM features to two dimensions
pca = PCA(n_components=2)

pca_result = pca.fit_transform(scaled_features)

# Add PCA columns
rfm["PCA1"] = pca_result[:, 0]
rfm["PCA2"] = pca_result[:, 1]

# Save updated customer segments
rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

print("\n===== PCA RESULTS =====")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:",
      pca.explained_variance_ratio_.sum())

print("\nPCA completed successfully!")


# Step 5: Cohort Analysis

# Use completed orders
cohort_data = completed.copy()

# Extract order month
cohort_data["OrderMonth"] = (
    cohort_data["order_ts"].dt.to_period("M")
)

# Find each customer's first purchase month
cohort_data["CohortMonth"] = (
    cohort_data.groupby("customer_id")["OrderMonth"]
    .transform("min")
)

# Calculate cohort index
cohort_data["CohortIndex"] = (
    (cohort_data["OrderMonth"].dt.year -
     cohort_data["CohortMonth"].dt.year) * 12
    +
    (cohort_data["OrderMonth"].dt.month -
     cohort_data["CohortMonth"].dt.month) + 1
)

# Calculate customer retention
cohort_counts = cohort_data.groupby(
    ["CohortMonth", "CohortIndex"]
)["customer_id"].nunique().unstack()

cohort_sizes = cohort_counts.iloc[:, 0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

# Save cohort results
retention.to_csv(OUTPUT_DIR / "cohort_analysis.csv")

print("\n===== COHORT ANALYSIS =====")
print(retention.round(3))

print("\nCohort analysis completed successfully!")


# Step 6: Prepare final customer segment output

# Identify the highest-monetary cluster as Premium
premium_cluster = cluster_summary["Monetary"].idxmax()

rfm["segment_id"] = rfm["Cluster"]
rfm["segment_name"] = rfm["Segment"]
rfm["is_premium"] = (
    rfm["segment_id"] == premium_cluster
)

# Rename RFM columns as required
rfm = rfm.rename(columns={
    "Recency": "recency",
    "Frequency": "frequency",
    "Monetary": "monetary"
})

# Save required columns
rfm[[
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]].to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nFinal customer segments saved!")
print("Premium customers:", rfm["is_premium"].sum())
