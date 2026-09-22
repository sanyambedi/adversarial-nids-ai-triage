"""preprocess.py - Shared data loading, schema definitions, and preprocessing pipeline.

Single source of truth for NSL-KDD dataset columns, categorical encoding,
and inference alignment so training and the demo application never drift apart.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Canonical NSL-KDD column specifications
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

CAT_COLS = ["protocol_type", "service", "flag"]

# Features an attacker can plausibly manipulate in network connections
CONTROLLABLE = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]

FEATURE_COLUMNS_FILE = "feature_columns.joblib"


def get_numeric_columns():
    """Returns list of numeric feature column names excluding categorical and target metadata."""
    return [
        c for c in COLUMNS
        if c not in CAT_COLS and c not in ["label", "difficulty", "is_attack"]
    ]


def prepare_training_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encodes train and test datasets together to ensure zero category mismatch.

    Returns:
        X_train (pd.DataFrame): Processed training feature matrix
        y_train (np.ndarray): Binary targets (0: normal, 1: attack)
        X_test (pd.DataFrame): Processed test feature matrix
        y_test (np.ndarray): Binary targets (0: normal, 1: attack)
        feature_columns (list): Ordered list of all column names
    """
    # Create binary target (0 = normal, 1 = attack)
    y_train = (train_df["label"] != "normal").astype(int).values
    y_test = (test_df["label"] != "normal").astype(int).values

    # Combined dummy encoding for categorical features
    combined_cats = pd.concat([train_df[CAT_COLS], test_df[CAT_COLS]], axis=0)
    dummies = pd.get_dummies(combined_cats, columns=CAT_COLS, dtype=int)

    train_enc = dummies.iloc[:len(train_df)].reset_index(drop=True)
    test_enc = dummies.iloc[len(train_df):].reset_index(drop=True)

    num_cols = get_numeric_columns()

    X_train = pd.concat([train_df[num_cols].reset_index(drop=True), train_enc], axis=1)
    X_test = pd.concat([test_df[num_cols].reset_index(drop=True), test_enc], axis=1)

    feature_columns = list(X_train.columns)

    # Save feature columns to guarantee inference consistency
    joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)

    return X_train, y_train, X_test, y_test, feature_columns


def prepare_inference_data(df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    """Preprocesses arbitrary incoming CSV traffic rows for inference.

    Aligns columns strictly to feature_columns, filling unseen dummies with 0.
    """
    if feature_columns is None:
        if os.path.exists(FEATURE_COLUMNS_FILE):
            feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
        else:
            raise FileNotFoundError(
                f"Cannot find {FEATURE_COLUMNS_FILE}. Run train.py first to establish the feature schema."
            )

    df_copy = df.copy()

    # Drop target columns if present in evaluation CSV
    for col in ["label", "difficulty", "is_attack"]:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])

    # Ensure required categorical columns exist
    for col in CAT_COLS:
        if col not in df_copy.columns:
            df_copy[col] = "unknown"

    num_cols = get_numeric_columns()
    for col in num_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0

    # One-hot encode categoricals
    df_cats = pd.get_dummies(df_copy[CAT_COLS], columns=CAT_COLS, dtype=int)

    # Combine numerics and dummy columns
    processed = pd.concat([df_copy[num_cols].reset_index(drop=True), df_cats.reset_index(drop=True)], axis=1)

    # Align strictly to trained feature columns
    aligned = processed.reindex(columns=feature_columns, fill_value=0)
    return aligned
