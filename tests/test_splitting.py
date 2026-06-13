"""Тесты стратифицированного разбиения на train/test."""
import pandas as pd
import pytest

from preprocessing import split_dataset


def _frame(n=100):
    return pd.DataFrame({"f": range(n), "target": [0, 1] * (n // 2)})


def test_part_sizes():
    df = _frame(100)
    sp = split_dataset(df, by="target", test_size=0.2)
    assert sp.report.n_part_b == 20
    assert sp.report.n_part_a == 80
    assert len(sp.part_a) + len(sp.part_b) == len(df)


def test_proportions_preserved():
    df = _frame(100)
    sp = split_dataset(df, by="target", test_size=0.3)
    assert abs(sp.part_a["target"].mean() - 0.5) < 0.05
    assert abs(sp.part_b["target"].mean() - 0.5) < 0.05


def test_missing_column_raises():
    with pytest.raises(KeyError):
        split_dataset(_frame(), by="nope")


def test_bad_test_size_raises():
    with pytest.raises(ValueError):
        split_dataset(_frame(), by="target", test_size=1.5)
