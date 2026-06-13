"""Внутренние утилиты для работы с dtype.

Решает проблему "колонка object, но внутри числа" — типичное состояние после
normalize_missing, где токены-пропуски заменены на NaN, но dtype не обновился.
"""
from __future__ import annotations

import warnings

import pandas as pd


def looks_numeric(s: pd.Series, threshold: float = 0.95) -> bool:
    """True если object-колонка содержит в основном числовые значения.

    Используется чтобы отличать "грязные числа в кавычках" (нужно
    to_numeric) от настоящих категориальных строк.
    """
    if s.dtype != object:
        return False
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    return numeric.notna().sum() / len(non_null) >= threshold


def looks_datetime(s: pd.Series, threshold: float = 0.8) -> bool:
    """True если object-колонка похожа на даты (≥threshold значений парсятся).

    Используется в parse_dates(columns=None), чтобы не пытаться парсить
    как дату каждую текстовую колонку. Чисто числовые колонки исключаются
    (их обрабатывает to_numeric, а pd.to_datetime трактует числа как timestamp).
    """
    if s.dtype != object:
        return False
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().mean() >= 0.5:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(non_null, errors="coerce")
    return parsed.notna().mean() >= threshold


def warn_numeric_objects(
    df: pd.DataFrame,
    skipped_cols: list[str],
    func_name: str,
) -> None:
    """Если в списке пропущенных колонок есть object-колонки с числовым
    содержимым — выдать warning. Используется в функциях которые работают
    только с числовыми колонками."""
    suspicious = [c for c in skipped_cols if c in df.columns and looks_numeric(df[c])]
    if suspicious:
        cols_str = ", ".join(repr(c) for c in suspicious)
        warnings.warn(
            f"{func_name}: пропущены колонки {cols_str} — dtype='object', "
            f"но содержимое выглядит числовым. "
            f"Примените to_numeric(df, columns=[...]) перед этой функцией.",
            UserWarning,
            stacklevel=3,
        )
