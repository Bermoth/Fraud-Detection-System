
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

BEHAVIORAL_COLS = [
    "sender_prior_transactions",
    "receiver_prior_transactions",
    "sender_time_since_previous",
]


def load_split(name):
    x = pd.read_parquet(DATA_DIR / f"x_{name}.parquet")
    y = pd.read_parquet(DATA_DIR / f"y_{name}.parquet").iloc[:, 0]
    return x, y


def train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, label):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(x_val)[:, 1]
    test_probs = model.predict_proba(x_test)[:, 1]

    print(f"\n--- {label} ---")
    print(f"Validation PR-AUC: {average_precision_score(y_val, val_probs):.5f}")
    print(f"Test PR-AUC:       {average_precision_score(y_test, test_probs):.5f}")
    print(f"Test ROC-AUC:      {roc_auc_score(y_test, test_probs):.5f}")

    return model


def main():
    print("Loading existing processed data...")

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    # Experiment A: all existing features.
    train_and_evaluate(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        "ALL FEATURES",
    )

    # Experiment B: remove the three behavioral features.
    columns_to_remove = [
        col for col in BEHAVIORAL_COLS if col in x_train.columns
    ]

    x_train_reduced = x_train.drop(columns=columns_to_remove)
    x_val_reduced = x_val.drop(columns=columns_to_remove)
    x_test_reduced = x_test.drop(columns=columns_to_remove)

    train_and_evaluate(
        x_train_reduced, y_train,
        x_val_reduced, y_val,
        x_test_reduced, y_test,
        "WITHOUT BEHAVIORAL FEATURES",
    )


if __name__ == "__main__":
    main()
