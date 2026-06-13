"""Pipeline для последовательной обработки датасета.

Pipeline собирает шаги через .step(name, func, **params) и .run(df) выполняет
их по очереди, собирая аудит-отчёт. Каждая step-функция должна:
    - принимать DataFrame первым аргументом
    - возвращать либо DataFrame, либо объект с атрибутами .df и .report
      (т.е. любой Result-dataclass из этого пакета)

Типичный сценарий:
    >>> from preprocessing import (
    ...     normalize_missing, drop_sparse_columns, impute, log_transform,
    ...     onehot_encode, Pipeline, detect_and_handle_outliers,
    ... )
    >>> pipe = (Pipeline()
    ...     .step("missing",     normalize_missing)
    ...     .step("drop_sparse", drop_sparse_columns, threshold=0.6)
    ...     .step("impute",      impute, strategy="median")
    ...     .step("encode",      onehot_encode, max_unique=20)
    ...     .step("log",         log_transform, columns=["Fare"])
    ...     .step("outliers",    detect_and_handle_outliers,
    ...           method="mad", strategy="clip")
    ... )
    >>> result = pipe.run(df)
    >>> print(result.df)
    >>> print(pipe.audit())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from preprocessing.outliers import detect_outliers, handle_outliers


@dataclass
class StepResult:
    """Что произошло на одном шаге пайплайна."""

    name: str
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    duration_seconds: float
    report: Any = None
    summary: str = ""


@dataclass
class PipelineResult:
    """Результат запуска всего пайплайна: финальный df + шаги."""

    df: pd.DataFrame
    steps: list[StepResult] = field(default_factory=list)


def _extract_df_and_report(output: Any) -> tuple[pd.DataFrame, Any]:
    """Из произвольного output функции достаёт df и report.

    Поддерживает два контракта:
        - функция вернула DataFrame напрямую -> report=None
        - функция вернула объект с .df и .report (наши Result-dataclass'ы)
    """
    if isinstance(output, pd.DataFrame):
        return output, None
    if hasattr(output, "df") and hasattr(output, "report"):
        return output.df, output.report
    raise TypeError(
        f"step-функция должна вернуть DataFrame или объект с .df и .report, "
        f"а вернула {type(output).__name__}"
    )


def _summarize_report(report: Any) -> str:
    """Делает короткую человекочитаемую сводку из произвольного report.

    Используется duck typing: смотрим какие у объекта есть атрибуты,
    не импортируя все Report-типы по одному.
    """
    if report is None:
        return ""

    if hasattr(report, "total_missing") and hasattr(report, "columns_with_missing"):
        return (
            f"NaN: {report.total_missing}, "
            f"колонок с NaN: {len(report.columns_with_missing)}"
        )

    if hasattr(report, "dropped_columns") and hasattr(report, "kept_columns"):
        names = [d.column for d in report.dropped_columns]
        joined = ", ".join(names) if names else "—"
        return f"удалено колонок: {len(report.dropped_columns)} ({joined})"

    if hasattr(report, "columns_imputed"):
        n_cells = sum(c.n_imputed for c in report.columns_imputed)
        return (
            f"заполнено ячеек: {n_cells} в {len(report.columns_imputed)} колонках"
        )

    if hasattr(report, "encoded_columns") and hasattr(report, "new_columns"):
        return (
            f"закодировано колонок: {len(report.encoded_columns)} "
            f"-> {len(report.new_columns)} dummy"
        )

    if hasattr(report, "parsed_columns") and hasattr(report, "extracted_features"):
        return (
            f"распарсено дат: {len(report.parsed_columns)} "
            f"-> {len(report.extracted_features)} feature"
        )

    if hasattr(report, "scaled_columns") and hasattr(report, "method"):
        return f"scale {report.method}: {len(report.scaled_columns)} колонок"

    if hasattr(report, "transformed_columns") and hasattr(report, "method"):
        return f"{report.method}: {len(report.transformed_columns)} колонок"

    if hasattr(report, "strategy") and hasattr(report, "cells_clipped"):
        if report.strategy == "clip":
            return f"clipped: {report.cells_clipped} ячеек"
        if report.strategy == "drop":
            return f"удалено строк: {report.rows_dropped}"
        if report.strategy == "mark_missing":
            return f"помечено NaN: {report.cells_marked_missing}"
        if report.strategy == "keep":
            return "без изменений"

    return type(report).__name__


def detect_and_handle_outliers(
    df: pd.DataFrame,
    *,
    method: str = "mad",
    strategy: str = "clip",
    columns: list[str] | None = None,
    threshold: float | None = None,
    percentile_bounds: tuple[float, float] = (1.0, 99.0),
    contamination: float = 0.1,
    min_unique_values: int = 1,
    n_neighbors: int = 20,
    gmm_components: int = 2,
):
    """Объединяет detect_outliers + handle_outliers в один шаг.

    Создаётся специально для Pipeline.step() — чтобы не разбивать
    обработку выбросов на два шага с прокидыванием detection-объекта.

    Параметры детекции (`method`, `columns`, `threshold`, ...) и параметр
    обработки (`strategy`) принимаются вместе.

    Возвращает OutlierHandlingResult (имеет .df и .report).
    """
    detection = detect_outliers(
        df,
        method=method,
        columns=columns,
        threshold=threshold,
        percentile_bounds=percentile_bounds,
        contamination=contamination,
        min_unique_values=min_unique_values,
        n_neighbors=n_neighbors,
        gmm_components=gmm_components,
    )
    return handle_outliers(df, detection, strategy=strategy)


class Pipeline:
    """Последовательный пайплайн преобразований DataFrame.

    Каждый шаг — пара (имя, функция, параметры). Pipeline хранит шаги в
    том порядке, в каком их добавили. .run(df) выполняет их по очереди,
    передавая выход одного шага на вход следующего.

    Все step-функции должны принимать df первым аргументом и возвращать
    либо DataFrame, либо объект с .df и .report (наши Result-dataclass'ы).
    """

    def __init__(self):
        self._steps: list[tuple[str, Callable, dict]] = []
        self._results: list[StepResult] = []

    def step(self, name: str, func: Callable, **params) -> "Pipeline":
        """Добавляет шаг и возвращает self (для chain-вызовов)."""
        self._steps.append((name, func, params))
        return self

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """Выполняет все шаги по очереди. Исходный df не модифицируется."""
        current = df.copy()
        results: list[StepResult] = []

        for name, func, params in self._steps:
            in_shape = current.shape
            t0 = time.time()
            output = func(current, **params)
            duration = time.time() - t0

            current, report = _extract_df_and_report(output)

            results.append(StepResult(
                name=name,
                input_shape=in_shape,
                output_shape=current.shape,
                duration_seconds=duration,
                report=report,
                summary=_summarize_report(report),
            ))

        self._results = results
        return PipelineResult(df=current, steps=results)

    def audit(self) -> pd.DataFrame:
        """Сводка по всем шагам: имя, форма до/после, время, summary."""
        return pd.DataFrame([{
            "step": r.name,
            "input": f"{r.input_shape[0]}x{r.input_shape[1]}",
            "output": f"{r.output_shape[0]}x{r.output_shape[1]}",
            "time_s": round(r.duration_seconds, 3),
            "summary": r.summary,
        } for r in self._results])

    def steps(self) -> list[str]:
        """Список имён шагов в порядке добавления."""
        return [name for name, _, _ in self._steps]

    def __repr__(self) -> str:
        return f"Pipeline(steps={self.steps()})"

    def __len__(self) -> int:
        return len(self._steps)
