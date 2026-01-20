from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import paths

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files. Expected: loans.csv and customers.csv."""
    loans = pd.read_csv(paths.loans_csv)
    customers = pd.read_csv(paths.customers_csv)
    return loans, customers

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, strip column names."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.drop_duplicates()
    return out

def merge_sources(loans: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Merge loans and customers on customer_id.
    Assumes both have column 'customer_id'.
    """
    if "customer_id" not in loans.columns or "customer_id" not in customers.columns:
        raise ValueError("Both loans and customers must contain 'customer_id' column.")

    df = loans.merge(customers, on="customer_id", how="left")
    return df

def make_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test keeping class distribution."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # stratify works only if y has at least 2 classes
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return X_train, X_test, y_train, y_test
