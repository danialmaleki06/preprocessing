"""Поиск, фильтрация и точечное редактирование значений в DataFrame.

- find_rows         — поиск строк по условию на одну колонку (read-only)
- find_coords       — поиск координат конкретных ячеек по условию (read-only)
- set_cells_to_nan  — точечно поставить выбранные ячейки в NaN; опционально
                      удалить строки целиком ставшие NaN
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def find_rows(
    df: pd.DataFrame,
    column: str,
    *,
    is_numeric: bool = False,
    is_non_numeric: bool = False,
    in_range: tuple[float, float] | None = None,
    equals: list | None = None,
    contains: str | None = None,
    pattern: str | None = None,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    """Возвращает строки где `column` удовлетворяет хотя бы одному критерию.

    Не изменяет исходный df. Все критерии объединяются через OR.

    Параметры:
        is_numeric:     значение приводится к числу
        is_non_numeric: значение НЕ приводится к числу (NaN тоже отсекается)
        in_range:       (lo, hi) — числовое значение в [lo, hi]
        equals:         значение из списка (точное совпадение)
        contains:       строковое значение содержит подстроку
        pattern:        строковое значение матчит regex
        case_sensitive: учитывать ли регистр для equals/contains/pattern
    """
    if column not in df.columns:
        raise KeyError(f"Колонка {column!r} не найдена")

    s = df[column]
    mask = pd.Series(False, index=df.index)

    if is_numeric:
        numeric = pd.to_numeric(s, errors="coerce")
        mask |= numeric.notna() & s.notna()

    if is_non_numeric:
        numeric = pd.to_numeric(s, errors="coerce")
        mask |= numeric.isna() & s.notna()

    if in_range is not None:
        lo, hi = in_range
        numeric = pd.to_numeric(s, errors="coerce")
        mask |= (numeric >= lo) & (numeric <= hi)

    if equals is not None:
        if case_sensitive:
            mask |= s.isin(equals)
        else:
            s_lower = s.astype(str).str.lower()
            eq_lower = [str(v).lower() for v in equals]
            mask |= s_lower.isin(eq_lower)

    if contains is not None:
        mask |= s.astype(str).str.contains(
            contains, case=case_sensitive, na=False, regex=False,
        )

    if pattern is not None:
        mask |= s.astype(str).str.contains(
            pattern, case=case_sensitive, na=False, regex=True,
        )

    return df[mask].copy()


def find_coords(
    df: pd.DataFrame,
    *,
    value: Any = None,
    is_numeric: bool = False,
    is_non_numeric: bool = False,
    in_range: tuple[float, float] | None = None,
    contains: str | None = None,
    pattern: str | None = None,
    case_sensitive: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Возвращает координаты всех ячеек удовлетворяющих условию.

    Возвращает DataFrame с колонками `row` (позиция строки 0..n), `column`
    (имя), `value` (содержимое ячейки). Если ничего не найдено — пустой df.

    Применяется ко всем колонкам df (или к подмножеству через `columns`).
    Удобно для:
    - найти все вхождения конкретного значения
    - найти все нечисловые ячейки в "числовых" колонках
    - найти все ячейки матчащие regex

    Параметры аналогичны find_rows, плюс:
        value:   найти ячейки с точным значением
        columns: ограничить поиск этими колонками. None = все колонки.
    """
    cols = columns if columns is not None else list(df.columns)
    cols = [c for c in cols if c in df.columns]

    rows: list[dict] = []
    for col in cols:
        s = df[col]
        mask = pd.Series(False, index=df.index)

        if value is not None:
            if case_sensitive or not isinstance(value, str):
                mask |= (s == value)
            else:
                mask |= s.astype(str).str.lower() == value.lower()

        if is_numeric:
            numeric = pd.to_numeric(s, errors="coerce")
            mask |= numeric.notna() & s.notna()

        if is_non_numeric:
            numeric = pd.to_numeric(s, errors="coerce")
            mask |= numeric.isna() & s.notna()

        if in_range is not None:
            lo, hi = in_range
            numeric = pd.to_numeric(s, errors="coerce")
            mask |= (numeric >= lo) & (numeric <= hi)

        if contains is not None:
            mask |= s.astype(str).str.contains(
                contains, case=case_sensitive, na=False, regex=False,
            )

        if pattern is not None:
            mask |= s.astype(str).str.contains(
                pattern, case=case_sensitive, na=False, regex=True,
            )

        positions = np.where(mask.values)[0]
        for pos in positions:
            rows.append({
                "row": int(pos),
                "column": str(col),
                "value": s.iloc[pos],
            })

    if not rows:
        return pd.DataFrame(columns=["row", "column", "value"])
    return pd.DataFrame(rows)


@dataclass
class SetCellsReport:
    """Отчёт set_cells_to_nan."""

    n_cells_set: int
    n_rows_dropped: int
    affected_columns: list[str]


@dataclass
class SetCellsResult:
    df: pd.DataFrame
    report: SetCellsReport


def set_cells_to_nan(
    df: pd.DataFrame,
    coords: list[tuple[int, int | str]],
    *,
    drop_empty_rows: bool = True,
) -> SetCellsResult:
    """Превращает выбранные ячейки в NaN. Опционально удаляет ставшие пустыми строки.

    Параметры:
        coords: список координат `(row, col)`:
                - `row` — позиция строки (int, 0..len-1)
                - `col` — имя колонки (str) ИЛИ позиция колонки (int)
        drop_empty_rows: True (дефолт) — удалить строки которые после операции
                стали целиком NaN. False — оставить пустые строки.

    Пример::

        # По именам колонок
        set_cells_to_nan(df, [(64, "Age"), (10, "Fare")])

        # По позициям
        set_cells_to_nan(df, [(64, 3), (10, 5)])

        # Смешано
        set_cells_to_nan(df, [(64, "Age"), (10, 5)])
    """
    df_out = df.copy()
    n_rows = len(df_out)
    affected: set[str] = set()
    n_set = 0

    for row_pos, col in coords:
        if not isinstance(row_pos, (int, np.integer)) or isinstance(row_pos, bool):
            raise TypeError(
                f"Позиция строки должна быть int, получено {type(row_pos).__name__}: {row_pos!r}"
            )
        if isinstance(col, int):
            if col < 0 or col >= len(df_out.columns):
                raise IndexError(
                    f"Позиция колонки {col} вне диапазона [0, {len(df_out.columns)})"
                )
            col_name = df_out.columns[col]
        else:
            col_name = col
            if col_name not in df_out.columns:
                raise KeyError(f"Колонка {col_name!r} не найдена")

        if row_pos < 0 or row_pos >= n_rows:
            raise IndexError(
                f"Позиция строки {row_pos} вне диапазона [0, {n_rows})"
            )

        col_loc = df_out.columns.get_loc(col_name)
        df_out.iat[row_pos, col_loc] = np.nan
        affected.add(str(col_name))
        n_set += 1

    n_dropped = 0
    if drop_empty_rows:
        before = len(df_out)
        df_out = df_out.dropna(how="all").reset_index(drop=True)
        n_dropped = before - len(df_out)

    return SetCellsResult(
        df=df_out,
        report=SetCellsReport(
            n_cells_set=n_set,
            n_rows_dropped=n_dropped,
            affected_columns=sorted(affected),
        ),
    )
