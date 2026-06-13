"""Разбиение датасета на две части с сохранением пропорций (стратификация).

split_dataset(df, by="churn") делит df на две части так, чтобы доли значений
колонки `by` совпадали в обеих частях и в исходном датасете.

Для категориальных/дискретных колонок стратифицируем напрямую. Для непрерывных
числовых — сначала режем на квантильные бины, потом стратифицируем по бинам
(иначе у каждого значения своя «категория» и стратификация бессмысленна).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitReport:
    """Отчёт split_dataset."""

    stratify_column: str
    test_size: float
    binned: bool
    n_total: int
    n_part_a: int
    n_part_b: int
    proportions: pd.DataFrame


@dataclass
class SplitResult:
    part_a: pd.DataFrame
    part_b: pd.DataFrame
    report: SplitReport


def split_dataset(
    df: pd.DataFrame,
    by: str,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    n_bins: int = 10,
) -> SplitResult:
    """Делит df на две части, сохраняя пропорции значений колонки `by`.

    Часть B получает долю `test_size` строк, часть A — остальное. В обеих
    частях доли значений `by` совпадают с исходными (с точностью до округления).

    Параметры:
        by:        колонка, пропорции которой сохранять.
        test_size: доля строк во второй части (0..1). Дефолт 0.2.
        seed:      зерно генератора для воспроизводимости.
        n_bins:    число квантильных бинов для числовой непрерывной колонки.

    Стратификация:
        - категориальная / дискретная колонка → по её значениям напрямую
        - непрерывная числовая (уникальных > n_bins) → по квантильным бинам
        - NaN в колонке `by` выделяется в отдельную группу "(NaN)"

    Бросает:
        KeyError:   колонки `by` нет в df.
        ValueError: какая-то группа содержит < 2 строк (стратификация
                    невозможна — её нельзя разделить на две части).

    Возвращает SplitResult с .part_a, .part_b и .report (с таблицей пропорций
    для проверки что доли сохранились).

    Пример::

        res = split_dataset(df, by="churn", test_size=0.25)
        train, test = res.part_a, res.part_b
        print(res.report.proportions)   # доли churn в full / train / test
    """
    if by not in df.columns:
        raise KeyError(f"Колонка {by!r} не найдена в df")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size должен быть в (0, 1), получено {test_size}")

    s = df[by]

    binned = False
    is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_numeric and s.nunique(dropna=True) > n_bins:
        strat = pd.qcut(s, q=n_bins, duplicates="drop").astype(str)
        binned = True
    else:
        strat = s.astype(str)

    strat = strat.where(s.notna(), "(NaN)")

    counts = strat.value_counts()
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        raise ValueError(
            f"Группы со слишком малым числом строк (< 2): "
            f"{too_small.to_dict()}. Стратификация невозможна — выберите "
            f"другую колонку или уменьшите n_bins."
        )

    from sklearn.model_selection import train_test_split

    part_a, part_b = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    prop = pd.DataFrame({
        "full": strat.value_counts(normalize=True),
        "part_a": strat.loc[part_a.index].value_counts(normalize=True),
        "part_b": strat.loc[part_b.index].value_counts(normalize=True),
    }).fillna(0.0).round(4).sort_index()

    part_a = part_a.reset_index(drop=True)
    part_b = part_b.reset_index(drop=True)

    return SplitResult(
        part_a=part_a,
        part_b=part_b,
        report=SplitReport(
            stratify_column=by,
            test_size=test_size,
            binned=binned,
            n_total=len(df),
            n_part_a=len(part_a),
            n_part_b=len(part_b),
            proportions=prop,
        ),
    )
