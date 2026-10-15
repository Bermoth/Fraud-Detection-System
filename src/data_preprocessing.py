
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "financial_fraud_detection_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TARGET = "is_fraud"

GRAPH_COLS = [
    "transaction_id",
    "sender_account",
    "receiver_account",
    "timestamp",
    "amount",
    TARGET,
]

ML_NUMERIC_RAW = [
    "amount",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]

BEHAVIORAL_COLS = [
    "sender_prior_transactions",
    "receiver_prior_transactions",
    "sender_time_since_previous",
]

ML_CATEGORICAL_COLS = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
]

NUMERIC_COLS = (
    ML_NUMERIC_RAW
    + BEHAVIORAL_COLS
    + ["hour", "day_of_week", "month"]
)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
SAMPLE_FRAC = None


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path=RAW_PATH, sample_frac=SAMPLE_FRAC):
    print("Loading dataset...")

    df = pd.read_csv(path)

    if sample_frac is not None:
        df = df.sample(
            frac=sample_frac,
            random_state=42,
        ).reset_index(drop=True)

    print(f"Dataset loaded: {df.shape}")

    return df


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):
    print("Cleaning data...")

    df = df.drop_duplicates(
        subset="transaction_id"
    ).copy()

    assert df["transaction_id"].is_unique, (
        "transaction_id must be unique."
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="ISO8601",
    )

    df[TARGET] = df[TARGET].astype("int8")

    print(f"Dataset after cleaning: {df.shape}")

    return df


# ============================================================
# 3. CREATE ACCOUNT-BEHAVIOR FEATURES
# ============================================================

def add_behavioral_features(df):
    print("Creating account-behavior features...")

    # Sort transactions chronologically so that the features
    # only describe activity occurring before each transaction.
    df = df.sort_values(
        "timestamp",
        kind="mergesort",
    ).copy()

    # Number of previous transactions for each sender.
    sender_groups = df.groupby(
        "sender_account",
        sort=False,
        dropna=False,
    )

    df["sender_prior_transactions"] = (
        sender_groups.cumcount()
    )

    # Number of previous transactions for each receiver.
    receiver_groups = df.groupby(
        "receiver_account",
        sort=False,
        dropna=False,
    )

    df["receiver_prior_transactions"] = (
        receiver_groups.cumcount()
    )

    # Seconds since the sender's previous transaction.
    df["sender_time_since_previous"] = (
        df.groupby(
            "sender_account",
            sort=False,
            dropna=False,
        )["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # -1 identifies a sender with no previous transaction.
    df["sender_time_since_previous"] = (
        df["sender_time_since_previous"]
        .fillna(-1)
    )

    # Use compact numeric types to reduce memory usage.
    df["sender_prior_transactions"] = (
        df["sender_prior_transactions"].astype("int32")
    )

    df["receiver_prior_transactions"] = (
        df["receiver_prior_transactions"].astype("int32")
    )

    df["sender_time_since_previous"] = (
        df["sender_time_since_previous"].astype("float32")
    )

    print("Behavioral features created:")
    for col in BEHAVIORAL_COLS:
        print(f"  - {col}")

    return df


# ============================================================
# 4. BUILD GRAPH DATASET
# ============================================================

def build_graph_dataframe(df):
    print("Building graph dataset...")

    graph_df = df[GRAPH_COLS].copy()

    return graph_df


# ============================================================
# 5. BUILD MACHINE-LEARNING DATASET
# ============================================================

def build_ml_dataframe(df):
    print("Building machine-learning dataset...")

    cols = (
        ["transaction_id", "timestamp", TARGET]
        + ML_NUMERIC_RAW
        + BEHAVIORAL_COLS
        + ML_CATEGORICAL_COLS
    )

    ml = df[cols].copy()

    # Extract temporal features.
    ml["hour"] = ml["timestamp"].dt.hour
    ml["day_of_week"] = ml["timestamp"].dt.dayofweek
    ml["month"] = ml["timestamp"].dt.month

    # Remove raw timestamp.
    ml = ml.drop(columns=["timestamp"])

    # Reduce the skew of transaction amounts.
    ml["amount"] = np.log1p(ml["amount"])

    # Ensure binary target.
    ml[TARGET] = ml[TARGET].astype("int8")

    # Keep transaction ID as the index, not a model feature.
    ml = ml.set_index("transaction_id")

    # One-hot encode categorical features.
    ml = pd.get_dummies(
        ml,
        columns=ML_CATEGORICAL_COLS,
        drop_first=True,
        dtype="int8",
    )

    print(f"ML dataset created: {ml.shape}")

    return ml


# ============================================================
# 6. SPLIT DATA AND SCALE NUMERIC FEATURES
# ============================================================


def split_and_scale(df):
    print("Splitting dataset...")

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train shape: {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")

    print("Scaling numeric features...")

    scaler = StandardScaler()

    # Convert selected columns to float32 BEFORE assigning
    # scaled values, avoiding pandas integer-dtype errors.
    x_train = x_train.copy()
    x_val = x_val.copy()
    x_test = x_test.copy()

    for split in (x_train, x_val, x_test):
        split[NUMERIC_COLS] = split[NUMERIC_COLS].astype("float32")

    x_train.loc[:, NUMERIC_COLS] = scaler.fit_transform(
        x_train[NUMERIC_COLS]
    ).astype("float32")

    x_val.loc[:, NUMERIC_COLS] = scaler.transform(
        x_val[NUMERIC_COLS]
    ).astype("float32")

    x_test.loc[:, NUMERIC_COLS] = scaler.transform(
        x_test[NUMERIC_COLS]
    ).astype("float32")

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        scaler,
    )



# ============================================================
# 7. SAVE PROCESSED DATA
# ============================================================

def main():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load and clean.
    df = load_data()
    df = clean_data(df)

    # Add behavioral features before creating either dataset.
    df = add_behavioral_features(df)

    # Save graph data, retaining sender and receiver IDs.
    graph_df = build_graph_dataframe(df)

    graph_path = PROCESSED_DIR / "graph_base.parquet"
    graph_df.to_parquet(
        graph_path,
        index=False,
    )

    print(f"Graph dataset saved: {graph_df.shape}")

    del graph_df
    gc.collect()

    # Build the ML dataset.
    ml_df = build_ml_dataframe(df)

    del df
    gc.collect()

    # Split and scale.
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = split_and_scale(ml_df)

    del ml_df
    gc.collect()

    # Save feature matrices.
    x_train.to_parquet(
        PROCESSED_DIR / "x_train.parquet"
    )

    x_val.to_parquet(
        PROCESSED_DIR / "x_val.parquet"
    )

    x_test.to_parquet(
        PROCESSED_DIR / "x_test.parquet"
    )

    # Save labels.
    y_train.to_frame().to_parquet(
        PROCESSED_DIR / "y_train.parquet"
    )

    y_val.to_frame().to_parquet(
        PROCESSED_DIR / "y_val.parquet"
    )

    y_test.to_frame().to_parquet(
        PROCESSED_DIR / "y_test.parquet"
    )

    # Save preprocessing artifacts.
    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.pkl",
    )

    joblib.dump(
        list(x_train.columns),
        MODELS_DIR / "feature_columns.pkl",
    )

    # Report results.
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)

    print(f"Training rows:   {len(x_train):,}")
    print(f"Validation rows: {len(x_val):,}")
    print(f"Test rows:       {len(x_test):,}")
    print(f"Total ML features: {x_train.shape[1]}")

    print(
        f"Training fraud rate: {y_train.mean():.4%}"
    )

    print(
        f"Validation fraud rate: {y_val.mean():.4%}"
    )

    print(
        f"Test fraud rate: {y_test.mean():.4%}"
    )

    print("\nBehavioral features included:")
    for col in BEHAVIORAL_COLS:
        print(f"  - {col}")

    print("\nProcessed files saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
