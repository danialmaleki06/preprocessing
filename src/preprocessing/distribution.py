"""Профилирование колонок датасета.

profile_columns(df) анализирует каждую колонку и возвращает DataFrame с метриками:
тип переменной, zero-inflation, skewness/kurtosis, тяжесть хвоста, число пиков KDE,
модальность и рекомендуемый метод detect_outliers.

Типичный сценарий:
    >>> from preprocessing.distribution import profile_columns
    >>> profile = profile_columns(df)
    >>> print(profile[["var_type", "tail_score", "modality", "recommended_method"]])
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


VarType = Literal["continuous", "discrete", "nominal", "ordinal"]
Modality = Literal["unimodal", "bimodal", "multimodal"]

_TAIL_MODERATE = 0.83
_TAIL_HEAVY    = 2.0
_TAIL_EXTREME  = 4.33


def _is_mixed_dtype(s: pd.Series) -> bool:
    """True если в object-колонке есть и числовые, и нечисловые значения.

    Сигнал баг загрузки или грязных данных: например, "yes"/"no" попали в
    колонку с числами, или наоборот. Чистая object-колонка (только строки)
    или чистая числовая (object с числами в кавычках) — НЕ mixed.
    """
    if s.dtype != object:
        return False
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    n_numeric = int(numeric.notna().sum())
    n_non_numeric = int(numeric.isna().sum())
    return n_numeric > 0 and n_non_numeric > 0


def _dtype_with_mixed_marker(s: pd.Series) -> str:
    """Возвращает str(dtype) с пометкой ' (!)' для смешанных типов."""
    return f"{s.dtype} (!)" if _is_mixed_dtype(s) else str(s.dtype)


def _classify_var_type(s: pd.Series, n_unique_discrete_threshold: int) -> VarType:
    """Определяет тип переменной по dtype и кардинальности."""
    if pd.api.types.is_bool_dtype(s):
        return "ordinal"
    if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
        return "nominal"
    if pd.api.types.is_integer_dtype(s):
        return "discrete" if s.nunique() < n_unique_discrete_threshold else "continuous"
    clean = s.dropna()
    if len(clean) > 0 and (clean == clean.round()).all() and s.nunique() < n_unique_discrete_threshold:
        return "discrete"
    return "continuous"


def _count_kde_peaks(s: pd.Series, bandwidth: float | None) -> int:
    """Считает число локальных максимумов в KDE-кривой (512 точек)."""
    clean = s.dropna().to_numpy(dtype=float)
    if len(clean) < 10:
        return 1
    try:
        kde = stats.gaussian_kde(clean, bw_method=bandwidth)
    except Exception:
        return 1
    x = np.linspace(clean.min(), clean.max(), 512)
    y = kde(x)
    dy = np.diff(y)
    sign_changes = np.diff(np.sign(dy))
    n_peaks = int((sign_changes < 0).sum())
    return max(n_peaks, 1)


def _classify_modality(n_peaks: int) -> Modality:
    if n_peaks == 1:
        return "unimodal"
    if n_peaks == 2:
        return "bimodal"
    return "multimodal"


def _compute_tail_score(skewness: float, excess_kurtosis: float) -> float:
    """Непрерывная оценка тяжести хвоста.

    score = abs(skewness) + max(0, excess_kurtosis) / 3

    Интерпретация через пороги _TAIL_*:
        score < 0.83  → лёгкий хвост (light)
        score < 2.0   → умеренный    (moderate)
        score < 4.33  → тяжёлый      (heavy)
        score ≥ 4.33  → экстремальный (extreme)
    """
    return round(abs(skewness) + max(0.0, excess_kurtosis) / 3.0, 3)


def _test_normality(s: pd.Series, alpha: float) -> bool | None:
    """Shapiro-Wilk только для n < 5000. None если выборка вне диапазона."""
    clean = s.dropna()
    n = len(clean)
    if n < 8 or n > 5000:
        return None
    try:
        _, p = stats.shapiro(clean.sample(min(n, 5000), random_state=0))
        return bool(p > alpha)
    except Exception:
        return None


def _recommend_method(
    var_type: VarType,
    tail_score: float | None,
    is_normal: bool | None,
    modality: Modality | None,
) -> str | None:
    """Рекомендует метод для detect_outliers по профилю колонки.

    Роутинг:
        nominal / ordinal        → None
        discrete                 → iqr
        continuous + multimodal  → None (нужен LOF/GMM)
        continuous:
            score < _TAIL_MODERATE и нормальное → zscore
            score < _TAIL_HEAVY                 → mad
            score < _TAIL_EXTREME               → mad
            score ≥ _TAIL_EXTREME               → log+mad
    """
    if var_type in ("nominal", "ordinal"):
        return None
    if var_type == "discrete":
        return "iqr"
    if modality in ("bimodal", "multimodal"):
        return None
    if tail_score is None:
        return "mad"
    if is_normal and tail_score < _TAIL_MODERATE:
        return "zscore"
    if tail_score < _TAIL_EXTREME:
        return "mad"
    return "log+mad"


def profile_columns(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    normality_alpha: float = 0.05,
    n_unique_discrete_threshold: int = 20,
    kde_bandwidth: float | None = None,
) -> pd.DataFrame:
    """Профилирует каждую колонку датасета и возвращает сводный DataFrame.

    Для каждой колонки вычисляет:
        var_type            — тип переменной: continuous/discrete/nominal/ordinal
        n_unique            — число уникальных значений (без NaN)
        missing_pct         — доля пропусков (0–100)
        zero_pct            — доля нулей среди non-NaN (zero-inflation, 0–100)
        skewness            — коэффициент асимметрии (только числовые)
        kurtosis            — избыточный эксцесс (только числовые)
        tail_score          — тяжесть хвоста: abs(skew) + max(0, kurtosis)/3
        is_normal           — Shapiro-Wilk при n < 5000: True/False/None
        n_kde_peaks         — число мод по KDE (только continuous)
        modality            — unimodal/bimodal/multimodal (только continuous)
        recommended_method  — рекомендуемый метод для detect_outliers

    Параметры:
        columns: список колонок. None → все колонки df.
        normality_alpha: уровень значимости для Shapiro-Wilk (по умолчанию 0.05).
        n_unique_discrete_threshold: граница уникальных для discrete/continuous.
        kde_bandwidth: bandwidth для KDE. None → scott rule (автоматически).

    Пример::

        profile = profile_columns(df)
        profile = profile_columns(df, columns=["age", "fare"])
        heavy = profile[profile["tail_score"] > 2.0].index.tolist()
    """
    if df.columns.has_duplicates:
        raise ValueError(
            f"В df есть дублирующиеся имена колонок: "
            f"{df.columns[df.columns.duplicated()].tolist()}. "
            f"Переименуйте их перед использованием."
        )
    target_cols = list(columns) if columns is not None else df.columns.tolist()
    rows: list[dict] = []

    for col in target_cols:
        if col not in df.columns:
            continue

        s = df[col]
        n = len(s)
        n_missing = int(s.isna().sum())
        missing_pct = round(n_missing / n * 100, 2) if n > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        var_type = _classify_var_type(s, n_unique_discrete_threshold)

        row: dict = {
            "dtype": _dtype_with_mixed_marker(s),
            "var_type": var_type,
            "n_unique": n_unique,
            "missing_pct": missing_pct,
            "zero_pct": None,
            "skewness": None,
            "kurtosis": None,
            "tail_score": None,
            "is_normal": None,
            "n_kde_peaks": None,
            "modality": None,
            "recommended_method": None,
        }

        if var_type in ("continuous", "discrete"):
            numeric_vals = pd.to_numeric(s.dropna(), errors="coerce").dropna()

            if len(numeric_vals) > 0:
                row["zero_pct"] = round((numeric_vals == 0).sum() / len(numeric_vals) * 100, 2)

                skewness = float(numeric_vals.skew())
                excess_kurtosis = float(numeric_vals.kurtosis())
                row["skewness"] = round(skewness, 4)
                row["kurtosis"] = round(excess_kurtosis, 4)

                tail_score = _compute_tail_score(skewness, excess_kurtosis)
                row["tail_score"] = tail_score
                row["is_normal"] = _test_normality(numeric_vals, alpha=normality_alpha)

                modality = None
                if var_type == "continuous":
                    n_peaks = _count_kde_peaks(numeric_vals, bandwidth=kde_bandwidth)
                    row["n_kde_peaks"] = n_peaks
                    modality = _classify_modality(n_peaks)
                    row["modality"] = modality

                row["recommended_method"] = _recommend_method(
                    var_type=var_type,
                    tail_score=tail_score,
                    is_normal=row["is_normal"],
                    modality=modality,
                )

        rows.append({"column": col, **row})

    from preprocessing._typing_utils import warn_numeric_objects
    suspicious = [r["column"] for r in rows if r.get("var_type") == "nominal"]
    if suspicious:
        warn_numeric_objects(df, suspicious, "profile_columns")
    if not rows:
        return pd.DataFrame(columns=[
            "dtype", "var_type", "n_unique", "missing_pct", "zero_pct",
            "skewness", "kurtosis", "tail_score", "is_normal",
            "n_kde_peaks", "modality", "recommended_method",
        ])
    return pd.DataFrame(rows).set_index("column")


def count_variations(
    df: pd.DataFrame,
    *,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Краткая сводка кардинальности каждой колонки.

    На каждую колонку возвращает: dtype, число уникальных значений, % от не-NaN,
    число пропусков и несколько примеров значений. Удобно для быстрой разведки:

    - Колонка с n_unique=1 — константа, можно удалить
    - Колонка с unique_pct ≈ 100% и dtype object — скорее всего ID или текст
    - Низкий n_unique у object — потенциальный кандидат на OneHot
    - Колонка с unique_pct < 5% — дискретная, для неё IQR надёжнее MAD

    Параметры:
        n_samples: сколько примеров значений показать в колонке samples.

    Возвращает DataFrame со строками-колонками датасета.

    Пример::

        var = count_variations(df)
        constants = var[var["n_unique"] == 1].index.tolist()
        ids = var[(var["unique_pct"] > 99) & (var["dtype"] == "object")].index.tolist()
    """
    if df.columns.has_duplicates:
        raise ValueError(
            f"В df есть дублирующиеся имена колонок: "
            f"{df.columns[df.columns.duplicated()].tolist()}. "
            f"Переименуйте их перед использованием."
        )
    rows: list[dict] = []
    n_total = len(df)
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_non_missing = max(n_total - n_missing, 1)
        n_unique = int(s.nunique(dropna=True))
        unique_pct = round(n_unique / n_non_missing * 100, 2)
        samples = s.dropna().unique()[:n_samples].tolist()
        rows.append({
            "column": str(col),
            "dtype": _dtype_with_mixed_marker(s),
            "n_unique": n_unique,
            "unique_pct": unique_pct,
            "n_missing": n_missing,
            "samples": samples,
        })
    if not rows:
        return pd.DataFrame(columns=["dtype", "n_unique", "unique_pct", "n_missing", "samples"])
    return pd.DataFrame(rows).set_index("column")


def _column_stats(s: pd.Series | None, total_rows: int) -> dict:
    """Статистика по одной колонке. None → колонка отсутствует в df."""
    if s is None:
        return {
            "dtype": "—",
            "n_unique": None,
            "missing_pct": None,
            "skew": None,
            "min": None,
            "max": None,
        }
    n_missing = int(s.isna().sum())
    missing_pct = round(n_missing / total_rows * 100, 2) if total_rows > 0 else 0.0
    result = {
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "missing_pct": missing_pct,
        "skew": None,
        "min": None,
        "max": None,
    }
    if pd.api.types.is_numeric_dtype(s) and s.notna().sum() >= 2:
        result["skew"] = round(float(s.skew()), 4)
        result["min"] = float(s.min())
        result["max"] = float(s.max())
    return result


def compare_datasets(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> pd.DataFrame:
    """Сводное сравнение двух DataFrame: что изменилось по каждой колонке.

    Для каждой колонки показывает before/after по: dtype, n_unique, % пропусков,
    skewness, min, max. Колонка `status` показывает: kept / dropped / added.

    Использование:
        - Проверить эффект log_transform: skew_before vs skew_after
        - Убедиться что impute убрал пропуски: missing_pct_after = 0
        - Увидеть какие колонки выкинул drop_sparse_columns: status="dropped"
        - Увидеть какие колонки добавил onehot_encode / parse_dates: status="added"

    Параметры:
        df_before: датасет до обработки.
        df_after:  датасет после обработки.

    Возвращает DataFrame со строкой на каждую колонку из объединения.

    Пример::

        df_clean = impute(df_raw, strategy="median").df
        diff = compare_datasets(df_raw, df_clean)
        print(diff[diff["status"] != "kept"])              # что выкинули/добавили
        print(diff[diff["missing_pct_before"] > 0])         # где были пропуски
    """
    cols_b = list(df_before.columns)
    cols_a = list(df_after.columns)
    all_cols = list(dict.fromkeys(cols_b + cols_a))

    n_b = len(df_before)
    n_a = len(df_after)
    rows: list[dict] = []

    for col in all_cols:
        s_b = df_before[col] if col in cols_b else None
        s_a = df_after[col] if col in cols_a else None

        if s_b is not None and s_a is None:
            status = "dropped"
        elif s_b is None and s_a is not None:
            status = "added"
        else:
            status = "kept"

        stats_b = _column_stats(s_b, n_b)
        stats_a = _column_stats(s_a, n_a)

        row = {"column": str(col), "status": status}
        for k in ["dtype", "n_unique", "missing_pct", "skew", "min", "max"]:
            row[f"{k}_before"] = stats_b[k]
            row[f"{k}_after"] = stats_a[k]
        rows.append(row)

    return pd.DataFrame(rows).set_index("column")
