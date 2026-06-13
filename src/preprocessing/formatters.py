"""Форматирование Report-объектов в человекочитаемый многострочный текст.

Главная функция — format_report(report). Она смотрит какие у объекта есть
атрибуты (duck typing) и подбирает шаблон под конкретный тип отчёта.

Использование:
    >>> from preprocessing import format_report
    >>> norm = normalize_missing(df)
    >>> print(format_report(norm.report))
    Пропуски — всего: 866
    Колонки с пропусками:
      Age: 177 (19.9%)
      Cabin: 687 (77.1%)
      Embarked: 2 (0.2%)
"""
from __future__ import annotations

from typing import Any


def format_report(report: Any) -> str:
    """Форматирует любой Report-объект в многострочный текст.

    Использует duck typing: проверяет наличие специфичных полей.
    Если ни один шаблон не подходит — возвращает repr().
    """
    if report is None:
        return "(нет отчёта)"

    if hasattr(report, "total_missing") and hasattr(report, "columns_with_missing"):
        return _format_missing(report)
    if hasattr(report, "dropped_columns") and hasattr(report, "kept_columns"):
        return _format_drop_sparse(report)
    if hasattr(report, "n_rows_before") and hasattr(report, "reasons"):
        return _format_drop_rows(report)
    if hasattr(report, "columns_imputed"):
        return _format_imputation(report)
    if hasattr(report, "encoded_columns") and hasattr(report, "new_columns"):
        return _format_onehot(report)
    if hasattr(report, "parsed_columns") and hasattr(report, "extracted_features"):
        return _format_dates(report)
    if hasattr(report, "scaled_columns") and hasattr(report, "method"):
        return _format_scale(report)
    if hasattr(report, "transformed_columns") and hasattr(report, "method"):
        return _format_transform(report)
    if hasattr(report, "column_stats") and hasattr(report, "total_outliers"):
        return _format_detect_outliers(report)
    if hasattr(report, "strategy") and hasattr(report, "cells_clipped"):
        return _format_handle_outliers(report)
    if hasattr(report, "stratify_column") and hasattr(report, "proportions"):
        return _format_split(report)

    return repr(report)


def _format_missing(r) -> str:
    lines = [f"Пропуски — всего: {r.total_missing}"]
    if r.columns_with_missing:
        lines.append("Колонки с пропусками:")
        for s in r.columns_with_missing:
            lines.append(f"  {s.column}: {s.n_missing} ({s.fraction_missing:.1%})")
    else:
        lines.append("(пропусков нет)")
    coerced = getattr(r, "coerced_numeric_columns", None)
    if coerced:
        lines.append(f"Авто-конвертировано в число: {', '.join(coerced)}")
    return "\n".join(lines)


def _format_drop_sparse(r) -> str:
    lines = [
        f"Drop sparse (threshold={r.threshold:.0%}): "
        f"удалено {len(r.dropped_columns)} колонок"
    ]
    for d in r.dropped_columns:
        lines.append(
            f"  {d.column}: {d.fraction_missing:.1%} пропусков ({d.n_missing} ячеек)"
        )
    return "\n".join(lines)


def _format_drop_rows(r) -> str:
    lines = [
        f"Drop rows по колонке {r.column!r}: "
        f"было {r.n_rows_before}, удалено {r.n_dropped}, осталось {r.n_rows_after}"
    ]
    if r.reasons:
        lines.append("Причины:")
        for reason, count in r.reasons.items():
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines)


def _format_imputation(r) -> str:
    n_total = sum(c.n_imputed for c in r.columns_imputed)
    lines = [
        f"Imputation (strategy={r.requested_strategy!r}): "
        f"заполнено {n_total} ячеек в {len(r.columns_imputed)} колонках"
    ]
    for c in r.columns_imputed:
        fill_str = repr(c.fill_value) if c.fill_value is not None else "—"
        lines.append(
            f"  {c.column}: {c.n_imputed} ячеек ({c.strategy_used}, fill={fill_str})"
        )
    if r.rows_dropped:
        lines.append(f"Удалено строк: {r.rows_dropped}")
    for w in r.warnings:
        lines.append(f"  [warn] {w}")
    return "\n".join(lines)


def _format_onehot(r) -> str:
    lines = [
        f"OneHot: закодировано {len(r.encoded_columns)} колонок "
        f"-> {len(r.new_columns)} dummy"
    ]
    if r.encoded_columns:
        lines.append(f"  Закодированы: {', '.join(r.encoded_columns)}")
    for col, reason in r.skipped_columns:
        lines.append(f"  [skipped] {col}: {reason}")
    return "\n".join(lines)


def _format_dates(r) -> str:
    lines = [
        f"Parse dates: распарсено {len(r.parsed_columns)} колонок "
        f"-> {len(r.extracted_features)} feature"
    ]
    if r.parsed_columns:
        lines.append(f"  Колонки: {', '.join(r.parsed_columns)}")
    for col, reason in r.skipped_columns:
        lines.append(f"  [skipped] {col}: {reason}")
    return "\n".join(lines)


def _format_transform(r) -> str:
    lines = [f"Transform ({r.method}): {len(r.transformed_columns)} колонок"]
    for ct in r.transformed_columns:
        lines.append(f"  {ct.column}")
    for col, reason in r.skipped_columns:
        lines.append(f"  [skipped] {col}: {reason}")
    return "\n".join(lines)


def _format_scale(r) -> str:
    lines = [f"Scale ({r.method}): масштабировано {len(r.scaled_columns)} колонок"]
    for cs in r.scaled_columns:
        lines.append(f"  {cs.column}: center={cs.center:.4g}, scale={cs.scale:.4g}")
    for col, reason in r.skipped_columns:
        lines.append(f"  [skipped] {col}: {reason}")
    return "\n".join(lines)


def _format_detect_outliers(r) -> str:
    th = f", threshold={r.threshold}" if r.threshold is not None else ""
    if getattr(r, "is_multivariate", False):
        lines = [
            f"Detect outliers (method={r.method}{th}): "
            f"найдено {r.total_outliers} строк-выбросов (по всей строке)"
        ]
        if r.columns_analyzed:
            lines.append(f"  Анализировались колонки: {', '.join(r.columns_analyzed)}")
    else:
        lines = [f"Detect outliers (method={r.method}{th}): найдено {r.total_outliers}"]
        for s in r.column_stats:
            lines.append(f"  {s.column}: {s.n_outliers} ({s.fraction_outliers:.1%})")
    for col, reason in r.columns_skipped:
        lines.append(f"  [skipped] {col}: {reason}")
    return "\n".join(lines)


def _format_handle_outliers(r) -> str:
    parts = [f"Handle outliers (strategy={r.strategy})"]
    if r.cells_clipped:
        parts.append(f"clipped: {r.cells_clipped}")
    if r.rows_dropped:
        parts.append(f"rows dropped: {r.rows_dropped}")
    if r.cells_marked_missing:
        parts.append(f"marked NaN: {r.cells_marked_missing}")
    lines = [", ".join(parts)]
    if r.columns_affected:
        lines.append(f"  Затронутые колонки: {', '.join(r.columns_affected)}")
    return "\n".join(lines)


def _format_split(r) -> str:
    bin_note = " (по квантильным бинам)" if r.binned else ""
    lines = [
        f"Split по {r.stratify_column!r}{bin_note}: "
        f"{r.n_total} -> part_a={r.n_part_a}, part_b={r.n_part_b} "
        f"(test_size={r.test_size})",
        "Пропорции (full / part_a / part_b):",
    ]
    for label, row in r.proportions.iterrows():
        lines.append(
            f"  {label}: {row['full']:.3f} / {row['part_a']:.3f} / {row['part_b']:.3f}"
        )
    return "\n".join(lines)
