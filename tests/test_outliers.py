"""Тесты обнаружения, обработки и переноса выбросов (без утечки)."""
import pandas as pd

from preprocessing import detect_outliers, handle_outliers, apply_outliers


def _frame():
    return pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})


def test_detect_iqr_flags_outlier():
    det = detect_outliers(_frame(), method="iqr", columns=["x"])
    assert det.masks["x"].iloc[-1]
    assert not det.masks["x"].iloc[0]
    assert "x" in det.bounds


def test_handle_clip_caps_value_and_keeps_size():
    df = _frame()
    det = detect_outliers(df, method="iqr", columns=["x"])
    out = handle_outliers(df, det, strategy="clip").df
    upper = det.bounds["x"][1]
    assert out["x"].max() <= upper + 1e-9
    assert out.shape[0] == df.shape[0]


def test_apply_outliers_uses_train_bounds():
    train = _frame()
    det = detect_outliers(train, method="iqr", columns=["x"])
    test = pd.DataFrame({"x": [12, 5000]})
    out = apply_outliers(test, det)
    upper = det.bounds["x"][1]
    assert out["x"].max() <= upper + 1e-9
