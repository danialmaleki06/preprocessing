"""Тесты масштабирования без утечки, логарифма и one-hot."""
import numpy as np
import pandas as pd

from preprocessing import (
    scale,
    apply_scale,
    log_transform,
    onehot_encode,
    to_numeric,
)


def test_scale_standard_centers_train():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    sc = scale(train, method="standard")
    assert abs(sc.df["x"].mean()) < 1e-9


def test_apply_scale_uses_train_params():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    sc = scale(train, method="standard")
    std = float(np.std([0, 1, 2, 3, 4]))
    test = pd.DataFrame({"x": [4.0, 0.0]})
    out = apply_scale(test, sc)
    assert abs(out["x"].iloc[0] - (4 - 2) / std) < 1e-9
    assert abs(out["x"].iloc[1] - (0 - 2) / std) < 1e-9


def test_log_transform_log1p():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    out = log_transform(df, columns=["x"], method="log1p").df
    assert np.allclose(out["x"].to_numpy(), np.log1p([0, 1, 2]))


def test_onehot_creates_dummies():
    df = pd.DataFrame({"c": ["a", "b", "a"]})
    out = onehot_encode(df, ["c"]).df
    assert "c" not in out.columns
    assert "c_a" in out.columns
    assert "c_b" in out.columns


def test_to_numeric_coerces_unparsable_to_nan():
    df = pd.DataFrame({"x": ["1", "2", "x"]})
    out = to_numeric(df, columns=["x"]).df
    assert out["x"].isna().sum() == 1
