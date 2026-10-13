import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BAS_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = Path("data/raw/financial_fraud_detection_dataset.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = BAS_DIR/"models"
TARGET = "is_fraud"

#Columns to drop
COLS_TO_DROP = ["transaction_id", "sender_account", "receiver_account", "ip_address", "device_hash", "fraud_type"]
CATEGORICAL_COLS = ["transaction_type", "merchant_category", "location", "device_used", "payment_channel"]
#Numeric columns to normalize
NUMERIC_COLS = ["amount", "time_since_last_transaction", "spending_deviation_score", "velocity_score", "geo_anomaly_score", "hour", "day_of_week", "month"]
SAMPLE_FRAC = None

#Load the raw dataset
def load_data(path=RAW_PATH, sample_frac=SAMPLE_FRAC):
    df = pd.read_csv(path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df

#Remove duplicates and unnecessary columns
def clean_data(df):
    df = df.drop_duplicates()
    df = df.drop(columns=COLS_TO_DROP)
    return df

#Extract date and time features
def add_time_features(df, date_col="timestamp"):
    ts = pd.to_datetime(df[date_col], format="ISO8601")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    return df.drop(columns=[date_col])

#Handle missing values for time_since_last_transaction
def handle_missing_values(df):
    col = "time_since_last_transaction"
    df["is_first_transaction"] = df[col].isna().astype("int8")
    df[col] = df[col].fillna(0)
    return df

#Encode categorical text columns into numerical indicator columns
def encode_categorical(df):
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype="int8")

#Split into train/test sets and apply scaling
def split_and_scale(df):
    x = df.drop(columns=[TARGET])
    y = df[TARGET].astype("int8")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()
    x_train[NUMERIC_COLS] = scaler.fit_transform(x_train[NUMERIC_COLS]).astype("float32")
    x_test[NUMERIC_COLS] = scaler.transform(x_test[NUMERIC_COLS]).astype("float32")

    return x_train, x_test, y_train, y_test, scaler

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    df = add_time_features(df)
    df = handle_missing_values(df)
    df["amount"] = np.log1p(df["amount"])
    df[TARGET] = df[TARGET].astype("int8") 
    df = encode_categorical(df)

    x_train, x_test, y_train, y_test, scaler = split_and_scale(df)

    x_train.to_parquet(PROCESSED_DIR / "x_train.parquet")
    x_test.to_parquet(PROCESSED_DIR / "x_test.parquet")

    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(list(x_train.columns), MODELS_DIR / "feature_columns.pkl")

    print("Traitement terminé")
    print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")
    print(f"Taux de fraude train: {y_train.mean():.4%}")


if __name__ == "__main__":
    main()