"""Тесты нормализации пропусков и заполнения без утечки."""
import numpy as np
import pandas as pd

from preprocessing import (
    normalize_missing,
    impute,
    apply_impute,
    drop_sparse_columns,
)


def test_normalize_tokens_to_nan():
    df = pd.DataFrame({"x": ["1", "2", "N/A", "unknown"]})
    res = normalize_missing(df, na_tokens={"N/A", "unknown"})
    assert res.df["x"].isna().sum() == 2


def test_normalize_coerces_numeric():
    df = pd.DataFrame({"x": ["1", "2", "3"]})
    res = normalize_missing(df, coerce_numeric=True)
    assert pd.api.types.is_integer_dtype(res.df["x"])


def test_impute_median_fills_all():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    res = impute(df, strategy="median")
    assert res.df["x"].isna().sum() == 0
    assert res.df["x"].iloc[3] == 2.0


def test_apply_impute_uses_train_value():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    imp = impute(train, strategy="median")
    test = pd.DataFrame({"x": [np.nan, np.nan]})
    out = apply_impute(test, imp)
    assert (out["x"] == 2.0).all()


def test_drop_sparse_removes_high_missing_column():
    df = pd.DataFrame({
        "keep": [1, 2, 3, 4],
        "drop": [1, np.nan, np.nan, np.nan],
    })
    res = drop_sparse_columns(df, threshold=0.5)
    assert "drop" not in res.df.columns
    assert "keep" in res.df.columns
