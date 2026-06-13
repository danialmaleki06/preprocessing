"""Дополнительные графики: распределения, сравнения, корреляция с таргетом,
ранжирование выбросов.

Дополняют heatmap.py (overview-карты) и scatter.py (2D-поиск выбросов)
более специализированными форматами под конкретные задачи.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_histogram(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Гистограмма с KDE-кривой для каждой числовой колонки отдельным графиком.

    Параметры:
        columns: список числовых колонок или "all" — все числовые.
        bins:    число корзин гистограммы.
        kde:     накладывать ли KDE-кривую.
    """
    cols = _resolve_numeric(df, columns, "plot_histogram")
    for col in cols:
        data = df[col].dropna()
        if data.empty:
            print(f"[!] {col}: пустая колонка, пропущена")
            continue
        plt.figure(figsize=figsize)
        sns.histplot(data, bins=bins, kde=kde, color="#4878cf", alpha=0.6)
        plt.title(
            f"Распределение: {col}  "
            f"(n={len(data):,}, mean={data.mean():.2f}, median={data.median():.2f})"
        )
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()
        plt.show()


def plot_violin(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    figsize: tuple[int, int] = (4, 6),
) -> None:
    """Violin-plot для каждой числовой колонки. Показывает форму распределения
    через ширину; лучше box-plot для оценки модальности.
    """
    cols = _resolve_numeric(df, columns, "plot_violin")
    for col in cols:
        data = df[col].dropna()
        if data.empty:
            print(f"[!] {col}: пустая колонка, пропущена")
            continue
        plt.figure(figsize=figsize)
        sns.violinplot(y=data, color="#4878cf", inner="quartile")
        plt.title(f"{col}\n(n={len(data):,})")
        plt.tight_layout()
        plt.show()


def plot_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Overlay KDE «до» (синий) и «после» (красный) для каждой колонки.

    Удобно после log_transform / handle_outliers / impute — сразу видишь
    как изменилась форма распределения.
    """
    from preprocessing._typing_utils import warn_numeric_objects
    if columns == "all":
        numeric_before = df_before.select_dtypes(include="number").columns
        cols = [
            c for c in numeric_before
            if c in df_after.columns and pd.api.types.is_numeric_dtype(df_after[c])
        ]
        skipped = [c for c in df_before.columns if c not in numeric_before]
        if skipped:
            warn_numeric_objects(df_before, skipped, "plot_before_after")
    else:
        cols = [
            c for c in columns
            if c in df_before.columns and c in df_after.columns
        ]

    for col in cols:
        b = pd.to_numeric(df_before[col], errors="coerce").dropna()
        a = pd.to_numeric(df_after[col], errors="coerce").dropna()
        if len(b) < 2 or len(a) < 2:
            print(f"[!] {col}: одна из выборок < 2 числовых точек, пропущено")
            continue
        plt.figure(figsize=figsize)
        sns.kdeplot(b, label=f"До (n={len(b):,})", color="#4878cf", fill=True, alpha=0.3)
        sns.kdeplot(a, label=f"После (n={len(a):,})", color="#d62728", fill=True, alpha=0.3)
        plt.title(f"До / после: {col}")
        plt.xlabel(col)
        plt.ylabel("Плотность")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_value_counts(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    top: int = 20,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Bar-plot частоты значений для каждой категориальной колонки.

    Показывает топ-N самых частых значений. Удобно для:
    - дисбаланса классов
    - поиска редких категорий-выбросов
    - оценки кардинальности
    """
    if columns == "all":
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        cols = [c for c in columns if c in df.columns]

    for col in cols:
        counts = df[col].value_counts(dropna=False).head(top)
        if counts.empty:
            print(f"[!] {col}: нет данных")
            continue
        plt.figure(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
        plt.barh(range(len(counts)), counts.values, color=colors)
        plt.yticks(range(len(counts)), [str(v) for v in counts.index])
        plt.xlabel("Количество")
        plt.title(
            f"{col} — топ-{len(counts)} значений "
            f"(всего уникальных: {df[col].nunique()})"
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


def plot_target_correlation(
    df: pd.DataFrame,
    target: str,
    *,
    method: str = "pearson",
    top: int | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Bar-plot: корреляция каждой числовой фичи с таргетом.

    Сортировка по убыванию |корреляции|. Зелёный — положительная связь,
    красный — отрицательная. Подходит для feature ranking перед ML.

    Параметры:
        target: имя колонки-таргета. Должна быть числовой.
        method: "pearson" (линейная), "spearman" (ранговая), "kendall".
        top:    показать только N сильнейших связей. None = все.
    """
    if target not in df.columns:
        raise KeyError(f"Колонка {target!r} не найдена в df")

    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        raise ValueError(f"Колонка {target!r} должна быть числовой")

    corr = numeric.corr(method=method)[target].drop(target)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    if top is not None:
        corr = corr.head(top)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in corr.values]
    plt.figure(figsize=figsize)
    plt.barh(range(len(corr)), corr.values, color=colors)
    plt.yticks(range(len(corr)), corr.index)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel(f"Корреляция с {target} ({method})")
    plt.title(f"Корреляция фич с таргетом {target!r}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_outlier_consensus(
    df: pd.DataFrame,
    pair: tuple[str, str],
    methods: list[str] | None = None,
    *,
    contamination: float = 0.05,
    figsize: tuple[int, int] = (9, 7),
) -> None:
    """Scatter пары колонок: цвет точки = сколько методов считают её выбросом.

    Точка с цветом ближе к красному — единогласный выброс, ближе к жёлтому —
    спорный, жёлто-белый — норма.

    Параметры:
        pair:    кортеж имён колонок `(x, y)`.
        methods: список методов из detect_outliers. По умолчанию 4 разнотипных.

    Пример::

        plot_outlier_consensus(
            df, ("Age", "Fare"),
            methods=["mad", "isolation_forest", "lof", "ecod"],
        )
    """
    from preprocessing.outliers import detect_outliers

    if methods is None:
        methods = ["mad", "isolation_forest", "lof", "ecod"]

    x, y = pair
    sub = df[[x, y]].dropna().reset_index(drop=True)
    if len(sub) < 10:
        print(f"[!] {x} vs {y}: меньше 10 точек, пропущено")
        return

    votes = pd.Series(0, index=sub.index)
    for m in methods:
        det = detect_outliers(sub, method=m, contamination=contamination)
        if not det.masks:
            continue
        row_mask = pd.Series(False, index=sub.index)
        for col_mask in det.masks.values():
            row_mask = row_mask | col_mask.reindex(sub.index, fill_value=False)
        votes = votes + row_mask.astype(int)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        sub[x], sub[y], c=votes.values,
        cmap="YlOrRd", s=40, alpha=0.7,
        edgecolors="black", linewidth=0.3,
        vmin=0, vmax=len(methods),
    )
    cbar = plt.colorbar(scatter, ticks=range(len(methods) + 1))
    cbar.set_label(f"Кол-во методов считающих выбросом (из {len(methods)})")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Outlier consensus: {x} vs {y}\n{', '.join(methods)}")
    plt.tight_layout()
    plt.show()


def plot_outlier_ranking(
    detection,
    *,
    top: int = 30,
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Bar-plot: сколько выбросов нашлось в каждой колонке.

    Сортировка по убыванию количества. С процентом от общего числа строк.
    Альтернатива plot_outlier_heatmap в виде столбчатой диаграммы.

    Параметры:
        detection: результат detect_outliers().
        top:       показать только N топ колонок.
    """
    column_stats = detection.report.column_stats
    if not column_stats:
        print("[!] В detection нет column_stats")
        return

    if getattr(detection.report, "is_multivariate", False):
        print(
            f"[!] Метод '{detection.report.method}' многомерный — выброс это "
            f"вся строка ({detection.report.total_outliers} строк), а не колонка. "
            f"Ранжирование по колонкам неприменимо. "
            f"Используйте plot_outlier_ranking с одномерным методом (iqr/mad/zscore)."
        )
        return

    sorted_stats = sorted(
        column_stats, key=lambda s: s.n_outliers, reverse=True
    )[:top]

    cols = [s.column for s in sorted_stats]
    counts = [s.n_outliers for s in sorted_stats]
    fractions = [s.fraction_outliers * 100 for s in sorted_stats]

    plt.figure(figsize=figsize)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(cols)))
    plt.barh(range(len(cols)), counts, color=colors)
    plt.yticks(range(len(cols)), cols)
    plt.xlabel("Количество выбросов")
    plt.title(
        f"Ранжирование выбросов по колонкам "
        f"(method={detection.report.method}, всего {detection.report.total_outliers})"
    )
    plt.gca().invert_yaxis()

    for i, (count, pct) in enumerate(zip(counts, fractions)):
        plt.text(count, i, f"  {pct:.1f}%", va="center", fontsize=9, color="#444")

    plt.tight_layout()
    plt.show()


def _resolve_numeric(df: pd.DataFrame, columns, func_name: str = "plot") -> list[str]:
    """Разрешает 'all'/список в реальный список числовых колонок."""
    from preprocessing._typing_utils import warn_numeric_objects

    numeric = df.select_dtypes(include="number").columns.tolist()
    if columns == "all":
        skipped = [c for c in df.columns if c not in numeric]
        if skipped:
            warn_numeric_objects(df, skipped, func_name)
        return numeric
    result = [c for c in columns if c in df.columns and c in numeric]
    skipped = [c for c in columns if c in df.columns and c not in numeric]
    if skipped:
        warn_numeric_objects(df, skipped, func_name)
    return result
