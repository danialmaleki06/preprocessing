"""Тепловые карты для визуального анализа датасета."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing._typing_utils import warn_numeric_objects


def _warn_numeric_objects_for_plot(df: pd.DataFrame, numeric_df: pd.DataFrame, func_name: str) -> None:
    skipped = [c for c in df.columns if c not in numeric_df.columns]
    if skipped:
        warn_numeric_objects(df, skipped, func_name)


def plot_missing_heatmap(
    df: pd.DataFrame,
    *,
    figsize: tuple[int, int] | None = None,
    title: str = "Пропуски по колонкам (%)",
) -> None:
    """Тепловая карта процента пропусков по каждой колонке.

    Каждая ячейка = % пропусков в этой колонке (0–100).
    Колонки отсортированы по убыванию % пропусков — самые «дырявые» слева.
    Цвет: белый = 0%, тёмно-красный = 100%.
    В каждой ячейке подписан точный процент.

    Пример::

        plot_missing_heatmap(df)
        plot_missing_heatmap(df, title="Мой датасет — пропуски")
    """
    miss_pct = (df.isnull().sum() / len(df) * 100).round(2)

    if miss_pct.sum() == 0:
        print("В датасете нет пропусков.")
        return

    miss_sorted = miss_pct.sort_values(ascending=False)

    matrix = pd.DataFrame([miss_sorted.values], columns=miss_sorted.index)

    if figsize is None:
        w = max(8, min(len(df.columns) * 0.8 + 2, 28))
        figsize = (w, 2.4)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Reds",
        vmin=0,
        vmax=100,
        annot=False,
        linewidths=1.2,
        linecolor="white",
        cbar=False,
    )

    miss_count = df.isnull().sum()[miss_sorted.index]
    font_size = max(7, 10 - len(df.columns) // 8)
    for i, (col, val) in enumerate(miss_sorted.items()):
        color = "#aaa" if val == 0 else ("white" if val > 40 else "#333")
        if val == 0:
            text = "0%\n0"
        else:
            text = f"{val:.1f}%\n{int(miss_count[col]):,}"
        ax.text(
            i + 0.5, 0.5, text,
            ha="center", va="center",
            fontsize=font_size, color=color, fontweight="bold",
            linespacing=1.6,
        )

    ax.set_yticks([])
    ax.set_xticks(np.arange(len(miss_sorted)) + 0.5)
    ax.set_xticklabels(
        miss_sorted.index,
        rotation=40,
        ha="right",
        fontsize=font_size,
        color="#444",
    )

    ax.set_title(
        f"{title}  ·  {len(df):,} строк  ·  {len(df.columns)} колонок",
        fontsize=12,
        pad=10,
        color="#333",
        fontweight="bold",
    )

    cols_with_missing = int((miss_pct > 0).sum())
    fig.text(
        0.01, -0.04,
        f"Колонок с пропусками: {cols_with_missing} / {len(df.columns)}   "
        f"Всего пропущено: {int(df.isnull().sum().sum()):,} ячеек",
        fontsize=8,
        color="#888",
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_minmax_heatmap(
    df: pd.DataFrame,
    *,
    n: int = 20,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> None:
    """Два графика: топ-N минимальных и топ-N максимальных значений по каждому столбцу.

    Строки = ранг (1 = самое крайнее значение).
    Цвет ячейки = позиция значения в диапазоне столбца по всему датасету.
    В ячейках подписаны фактические значения (если столбцов ≤ 15).
    """
    numeric_df = df.select_dtypes(include="number")
    _warn_numeric_objects_for_plot(df, numeric_df, "plot_minmax_heatmap")
    if numeric_df.empty:
        print("Нет числовых столбцов для отображения.")
        return

    cols = list(numeric_df.columns)

    min_actual = pd.DataFrame(
        {col: numeric_df[col].dropna().nsmallest(n).values for col in cols}
    )
    max_actual = pd.DataFrame(
        {col: numeric_df[col].dropna().nlargest(n).values for col in cols}
    )
    min_actual.index = [f"#{i + 1}" for i in range(len(min_actual))]
    max_actual.index = [f"#{i + 1}" for i in range(len(max_actual))]

    min_bg = pd.DataFrame(0.25, index=min_actual.index, columns=min_actual.columns)
    max_bg = pd.DataFrame(0.25, index=max_actual.index, columns=max_actual.columns)

    font_size = max(5, 9 - len(cols) // 4)
    annot_kws = {"size": font_size}

    def _fmt_annot(val: float) -> str:
        if pd.isna(val):
            return ""
        if val == int(val):
            return f"{int(val):,}"
        abs_val = abs(val)
        if abs_val >= 100:
            return f"{val:,.1f}"
        if abs_val >= 1:
            return f"{val:.2f}"
        return f"{val:.4f}"

    def _annot_array(frame: pd.DataFrame) -> np.ndarray:
        return np.vectorize(_fmt_annot)(frame.values)

    if figsize is None:
        w = max(8, min(len(cols) * 0.9 + 2, 26))
        figsize = (w, 14)

    fig, (ax_min, ax_max) = plt.subplots(2, 1, figsize=figsize)

    sns.heatmap(
        min_bg,
        ax=ax_min,
        cmap="Blues",
        vmin=0, vmax=1,
        annot=_annot_array(min_actual),
        fmt="",
        annot_kws=annot_kws,
        linewidths=0.5,
        linecolor="#ddd",
        cbar=False,
    )
    ax_min.set_title(f"Топ-{n} минимальных значений по каждому столбцу", fontsize=12, pad=8)
    ax_min.set_xlabel("")
    ax_min.set_ylabel("Ранг")
    ax_min.set_xticklabels(ax_min.get_xticklabels(), rotation=45, ha="right", fontsize=font_size + 1)

    sns.heatmap(
        max_bg,
        ax=ax_max,
        cmap="Reds",
        vmin=0, vmax=1,
        annot=_annot_array(max_actual),
        fmt="",
        annot_kws=annot_kws,
        linewidths=0.5,
        linecolor="#ddd",
        cbar=False,
    )
    ax_max.set_title(f"Топ-{n} максимальных значений по каждому столбцу", fontsize=12, pad=8)
    ax_max.set_xlabel("")
    ax_max.set_ylabel("Ранг")
    ax_max.set_xticklabels(ax_max.get_xticklabels(), rotation=45, ha="right", fontsize=font_size + 1)

    display_title = title or f"Min / Max значения датасета  (топ-{n} по каждому столбцу)"
    fig.suptitle(display_title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    annot_threshold: int = 20,
) -> None:
    """Корреляционная матрица числовых столбцов.

    Показывает, насколько сильно столбцы связаны друг с другом.
    Значения от -1 (обратная связь) до +1 (прямая связь).
    Ячейки аннотируются числами, если столбцов ≤ annot_threshold.

    method: 'pearson' (линейная), 'spearman' (ранговая), 'kendall'.
    """
    numeric_df = df.select_dtypes(include="number")
    _warn_numeric_objects_for_plot(df, numeric_df, "plot_correlation_heatmap")
    if numeric_df.shape[1] < 2:
        print("Нужно минимум 2 числовых столбца для корреляции.")
        return

    corr = numeric_df.corr(method=method)
    n = len(corr)
    annot = n <= annot_threshold

    if figsize is None:
        side = max(6, min(n * 0.65 + 2, 20))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        annot=annot,
        fmt=".2f" if annot else "",
        annot_kws={"size": max(6, 10 - n // 5)},
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Коэффициент корреляции", "shrink": 0.6},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    display_title = title or f"Корреляционная матрица  (метод: {method})"
    ax.set_title(display_title, fontsize=13, pad=10)

    plt.tight_layout()
    plt.show()


def rank_correlations(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    min_abs: float = 0.0,
) -> pd.DataFrame:
    """Возвращает все пары колонок отсортированные по |корреляции| (по убыванию).

    Каждая пара появляется один раз (верхний треугольник матрицы), диагональ
    исключена. Удобно для быстрого поиска сильно связанных колонок без
    разглядывания heatmap.

    Параметры:
        method:  "pearson" (линейная), "spearman" (ранговая), "kendall".
        min_abs: фильтр по |корреляции|. Например, 0.3 отбросит слабые пары.

    Возвращает DataFrame с колонками: col_a, col_b, corr, abs_corr.

    Пример::

        ranked = rank_correlations(df)
        print(ranked.head(10))
        strong = rank_correlations(df, min_abs=0.5)
    """
    corr = df.corr(method=method, numeric_only=True)
    mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask_upper)
    pairs = upper.stack().reset_index()
    pairs.columns = ["col_a", "col_b", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs[pairs["abs_corr"] >= min_abs]
    pairs = pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return pairs


def plot_outlier_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "mad",
    threshold: float = 3.5,
    iqr_k: float = 1.5,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    top_n: int = 10,
) -> None:
    """Карта выбросов: одна строка — % выбросов в каждой колонке по всему датасету.

    method:
        "mad"    — Modified Z-score: 0.6745*(x−median)/MAD > threshold (робастный, по умолчанию).
        "iqr"    — правило Тьюки: значение < Q1 − k·IQR или > Q3 + k·IQR.
        "zscore" — выброс если |z-score| > threshold (чувствителен к самим выбросам).

    Каждая ячейка = % строк-выбросов в этой колонке (0–100).
    Столбцы отсортированы по убыванию % выбросов — самые «выбросистые» слева.
    Цвет: белый = 0%, тёмно-красный = высокий %.
    В конце печатается топ-top_n столбцов — кандидатов для 2D-анализа.

    Пример::

        plot_outlier_heatmap(df)
        plot_outlier_heatmap(df, method="iqr")
        plot_outlier_heatmap(df, method="mad", threshold=3.5)
    """
    numeric_df = df.select_dtypes(include="number")
    _warn_numeric_objects_for_plot(df, numeric_df, "plot_outlier_heatmap")
    if numeric_df.empty:
        print("Нет числовых столбцов для отображения.")
        return

    if method == "mad":
        median = numeric_df.median()
        mad = (numeric_df - median).abs().median().replace(0, np.nan)
        score = 0.6745 * (numeric_df - median) / mad
        is_outlier = score.abs() > threshold
        method_label = f"Modified Z-score (MAD) > {threshold}"
    elif method == "zscore":
        mean = numeric_df.mean()
        std = numeric_df.std().replace(0, np.nan)
        score = (numeric_df - mean) / std
        is_outlier = score.abs() > threshold
        method_label = f"|z-score| > {threshold}"
    elif method == "iqr":
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        lo_bound = q1 - iqr_k * iqr
        hi_bound = q3 + iqr_k * iqr
        is_outlier = (numeric_df < lo_bound) | (numeric_df > hi_bound)
        method_label = f"Tukey IQR (k={iqr_k})"
    else:
        raise ValueError(
            f"Неизвестный method: {method!r}. Используй 'mad', 'zscore' или 'iqr'."
        )

    outlier_pct = (is_outlier.sum() / len(numeric_df) * 100).round(2)
    n_outliers_total = int(is_outlier.sum().sum())

    if n_outliers_total == 0:
        print(f"Выбросов не найдено ({method_label}).")
        return

    outlier_sorted = outlier_pct.sort_values(ascending=False)
    matrix = pd.DataFrame([outlier_sorted.values], columns=outlier_sorted.index)

    if figsize is None:
        w = max(8, min(len(outlier_sorted) * 0.8 + 2, 28))
        figsize = (w, 2.4)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Reds",
        vmin=0,
        vmax=100,
        annot=False,
        linewidths=1.2,
        linecolor="white",
        cbar=False,
    )

    outlier_count = is_outlier.sum()[outlier_sorted.index]
    font_size = max(7, 10 - len(outlier_sorted) // 8)
    for i, (col, val) in enumerate(outlier_sorted.items()):
        color = "#aaa" if val == 0 else ("white" if val > 40 else "#333")
        if val == 0:
            text = "0%\n0"
        else:
            text = f"{val:.1f}%\n{int(outlier_count[col]):,}"
        ax.text(
            i + 0.5, 0.5, text,
            ha="center", va="center",
            fontsize=font_size, color=color, fontweight="bold",
            linespacing=1.6,
        )

    ax.set_yticks([])
    ax.set_xticks(np.arange(len(outlier_sorted)) + 0.5)
    ax.set_xticklabels(
        outlier_sorted.index,
        rotation=40,
        ha="right",
        fontsize=font_size,
        color="#444",
    )

    display_title = title or f"Выбросы по колонкам  ({method_label})"
    ax.set_title(
        f"{display_title}  ·  {len(df):,} строк  ·  {len(outlier_sorted)} колонок",
        fontsize=12, pad=10, color="#333", fontweight="bold",
    )

    cols_with_outliers = int((outlier_pct > 0).sum())
    fig.text(
        0.01, -0.04,
        f"Колонок с выбросами: {cols_with_outliers} / {len(outlier_sorted)}   "
        f"Всего выбросов: {n_outliers_total:,} ячеек",
        fontsize=8, color="#888",
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    candidates = outlier_sorted[outlier_sorted > 0].head(top_n)
    if len(candidates):
        print(
            f"\nТоп-{len(candidates)} столбцов с наибольшим % выбросов "
        )
        for i, (col, pct) in enumerate(candidates.items(), 1):
            print(f"  {i:>2}. {col:<30} {pct:>6.2f}%")
