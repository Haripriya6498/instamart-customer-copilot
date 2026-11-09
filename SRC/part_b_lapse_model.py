
import pandas as pd
import numpy as np
import re
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")

# Find column names safely, regardless of capitalization
def find_column(df, wanted):
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }
    key = re.sub(r"[^a-z0-9]", "", wanted.lower())

    if key not in normalized:
        raise ValueError(
            f"Column '{wanted}' not found. Available columns: "
            f"{df.columns.tolist()}"
        )

    return normalized[key]


# --------------------------------------------------
# 3. STANDARDIZE IMPORTANT COLUMN NAMES
# --------------------------------------------------

signup_col = find_column(customers, "signUp_date")
order_time_col = find_column(orders, "order_ts")
status_col = find_column(orders, "status")
amount_col = find_column(orders, "order_amount")
delivery_col = find_column(orders, "delivery_minutes")
discount_col = find_column(orders, "discount_pct")

# Convert dates and numeric columns
cutoff = pd.Timestamp("2025-09-01")

orders[order_time_col] = pd.to_datetime(
    orders[order_time_col], errors="coerce"
)

customers[signup_col] = pd.to_datetime(
    customers[signup_col], errors="coerce"
)

orders[amount_col] = pd.to_numeric(
    orders[amount_col], errors="coerce"
)

orders[delivery_col] = pd.to_numeric(
    orders[delivery_col], errors="coerce"
)

orders[discount_col] = pd.to_numeric(
    orders[discount_col], errors="coerce"
)

# --------------------------------------------------
# 4. CREATE CUSTOMER COHORT
# --------------------------------------------------

completed = orders[
    orders[status_col].astype(str).str.lower() == "completed"
].copy()

before = orders[
    orders[order_time_col] < cutoff
].copy()

completed_before = completed[
    completed[order_time_col] < cutoff
].copy()

# Customers with at least one completed order before cutoff
cohort_ids = completed_before["customer_id"].unique()

# --------------------------------------------------
# 5. CREATE CUSTOMER FEATURES
# --------------------------------------------------

features = completed_before.groupby("customer_id").agg(
    recency=(
        order_time_col,
        lambda x: (cutoff - x.max()).days
    ),
    frequency=("order_id", "nunique"),
    monetary=(amount_col, "sum"),
    avg_discount=(discount_col, "mean"),
    avg_delivery=(delivery_col, "mean")
).reset_index()

# Cancellation rate: cancelled / all final-status orders
final_orders = before[
    before[status_col].astype(str).str.lower() != "pending"
].copy()

cancel_rate = final_orders.groupby("customer_id").agg(
    cancellation_rate=(
        status_col,
        lambda x: (
            x.astype(str).str.lower() == "cancelled"
        ).sum() / len(x)
    )
).reset_index()

features = features.merge(
    cancel_rate,
    on="customer_id",
    how="left"
)

# Most frequently ordered product category
category_orders = completed_before.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite = category_orders.groupby("customer_id")[
    "category"
].agg(
    lambda x: (
        x.mode().iloc[0]
        if not x.mode().empty
        else "Unknown"
    )
).reset_index(name="favourite_category")

features = features.merge(
    favourite,
    on="customer_id",
    how="left"
)

# Customer details and tenure
customer_info = customers.copy()

customer_info["tenure_days"] = (
    cutoff - customer_info[signup_col]
).dt.days

features = features.merge(
    customer_info[
        ["customer_id", "city", "age", "gender", "tenure_days"]
    ],
    on="customer_id",
    how="left"
)

# --------------------------------------------------
# 6. CREATE LAPSE LABEL
# --------------------------------------------------

september_customers = completed[
    (completed[order_time_col] >= cutoff) &
    (completed[order_time_col] < pd.Timestamp("2025-10-01"))
]["customer_id"].unique()

features["lapsed"] = (
    ~features["customer_id"].isin(september_customers)
).astype(int)

features["cancellation_rate"] = (
    features["cancellation_rate"].fillna(0)
)

# Save feature dataset
features.to_csv(
    OUTPUT_DIR / "lapse_features.csv",
    index=False
)

print("\nLAPSE FEATURES CREATED")
print("Cohort size:", len(features))
print("Lapsed customers:", features["lapsed"].sum())
print("Lapse percentage:", round(features["lapsed"].mean() * 100, 2))
print("Feature columns:", features.columns.tolist())

# --------------------------------------------------
# 7. TRAIN-TEST SPLIT
# --------------------------------------------------

X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 8. PREPROCESSING
# --------------------------------------------------

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery",
    "cancellation_rate",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nMissing feature values before preprocessing:")
print(X_train.isnull().sum())

# --------------------------------------------------
# 9. BASELINE MODEL
# --------------------------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBASELINE RESULTS")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print(
    "Lapsed F1:",
    f1_score(y_test, baseline_predictions, zero_division=0)
)

# --------------------------------------------------
# 10. LOGISTIC REGRESSION GRID SEARCH
# --------------------------------------------------

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION")
print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", lr_grid.best_score_)

# --------------------------------------------------
# 11. RANDOM FOREST GRID SEARCH
# --------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nRANDOM FOREST")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", rf_grid.best_score_)

# --------------------------------------------------
# 12. EVALUATE BOTH MODELS
# --------------------------------------------------

def evaluate_model(name, model):
    predictions = model.predict(X_test)

    print(f"\n{name} RESULTS")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

evaluate_model("Logistic Regression", lr_grid.best_estimator_)
evaluate_model("Random Forest", rf_grid.best_estimator_)

# Choose model by cross-validation F1, not test F1
if lr_grid.best_score_ >= rf_grid.best_score_:
    selected_model = lr_grid.best_estimator_
    selected_name = "Logistic Regression"
else:
    selected_model = rf_grid.best_estimator_
    selected_name = "Random Forest"

print("\nSELECTED MODEL:", selected_name)

# --------------------------------------------------
# 13. LOGISTIC REGRESSION COEFFICIENTS
# --------------------------------------------------

lr_model = lr_grid.best_estimator_
lr_preprocessor = lr_model.named_steps["preprocessor"]
lr_classifier = lr_model.named_steps["model"]

feature_names = lr_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_classifier.coef_[0]
})

print("\nTOP LOGISTIC REGRESSION COEFFICIENTS")
print(
    coefficients.reindex(
        coefficients["coefficient"].abs().sort_values(
            ascending=False
        ).index
    ).head(10)
)

# --------------------------------------------------
# 14. RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------

rf_model = rf_grid.best_estimator_
rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

rf_feature_names = rf_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP FIVE RANDOM FOREST FEATURES")
print(importance.head(5))

# --------------------------------------------------
# 15. OVERSAMPLING TRAINING DATA ONLY
# --------------------------------------------------

train_data = X_train.copy()
train_data["lapsed"] = y_train.values

majority_count = train_data["lapsed"].value_counts().max()

minority_data = train_data[
    train_data["lapsed"] == 1
]

if len(minority_data) == 0:
    raise ValueError("No lapsed customers available for oversampling.")

extra = minority_data.sample(
    n=majority_count - len(minority_data),
    replace=True,
    random_state=42
)

balanced_train = pd.concat(
    [train_data, extra],
    ignore_index=True
)

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

# Refit best Logistic Regression settings
balanced_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=lr_grid.best_params_["model__C"],
        max_iter=1000
    ))
])

balanced_lr.fit(X_balanced, y_balanced)

balanced_predictions = balanced_lr.predict(X_test)

print("\nOVERSAMPLED LOGISTIC REGRESSION")
print("Precision:", precision_score(y_test, balanced_predictions, zero_division=0))
print("Recall:", recall_score(y_test, balanced_predictions, zero_division=0))
print("F1:", f1_score(y_test, balanced_predictions, zero_division=0))

# --------------------------------------------------
# 16. SAVE AND RELOAD SELECTED MODEL
# --------------------------------------------------

model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

loaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

print("\nMODEL SAVE CHECK")
print("Saved model:", model_path)
print(
    "Predictions identical after reload:",
    np.array_equal(original_predictions, reloaded_predictions)
)

# --------------------------------------------------
# 17. PREMIUM CUSTOMER RETENTION LIST
# --------------------------------------------------

segments_path = OUTPUT_DIR / "customer_segments.csv"

if segments_path.exists():
    segments = pd.read_csv(segments_path)

    premium_ids = segments.loc[
        segments["is_premium"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ),
        "customer_id"
    ]

    test_results = pd.DataFrame({
        "customer_id": features.loc[
            X_test.index, "customer_id"
        ].values,
        "actually_lapsed": y_test.values,
        "lapse_probability": selected_model.predict_proba(X_test)[:, 1]
    })

    retention = test_results[
        test_results["customer_id"].isin(premium_ids)
    ].sort_values("lapse_probability", ascending=False)

    retention["actually_lapsed"] = retention[
        "actually_lapsed"
    ].astype(int)

    retention = retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    retention.to_csv(
        OUTPUT_DIR / "retention_list.csv",
        index=False
    )

    print("\nPREMIUM CUSTOMER RETENTION LIST")
    print(retention.head(10))
    print("Premium customers in test set:", len(retention))
    print("Actually lapsed:", retention["actually_lapsed"].sum())

else:
    print("\nCustomer segments file not found.")
    print("Retention list was not created.")

print("\nPART B LAPSE MODEL COMPLETED SUCCESSFULLY!")
