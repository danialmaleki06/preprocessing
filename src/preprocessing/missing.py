"""Поиск, нормализация и заполнение пропущенных значений."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from preprocessing._typing_utils import warn_numeric_objects

DEFAULT_NA_TOKENS: frozenset[str] = frozenset({
    "",
    "na", "n/a", "#n/a", "#n/a n/a",
    "null", "none", "nil",
    "nan", "-nan",
    "?", "-", "--", "---",
    "#null!", "#div/0!", "#value!", "#ref!", "#name?",
    "не указано", "неизвестно", "н/д", "нд",
    "unknown", "undefined", "missing"
})


@dataclass
class ColumnMissingStats:
    """Статистика пропусков по одной колонке после нормализации."""

    column: str
    n_missing: int
    fraction_missing: float
    found_tokens: dict[str, int]


@dataclass
class MissingReport:
    """Отчёт о пропусках после normalize_missing."""

    n_rows: int
    n_columns: int
    total_missing: int
    columns_with_missing: list[ColumnMissingStats]
    na_tokens_used: frozenset[str]
    coerced_numeric_columns: list[str] = field(default_factory=list)


@dataclass
class NormalizeMissingResult:
    """Результат normalize_missing: новый DataFrame + отчёт."""

    df: pd.DataFrame
    report: MissingReport


def _normalize_token(value: str) -> str:
    """Регистронезависимое сравнение после strip — 'NA  ' и 'na' это один токен."""
    return value.strip().lower()


def normalize_missing(
    df: pd.DataFrame,
    na_tokens: frozenset[str] | set[str] | None = None,
    *,
    coerce_numeric: bool = True,
    coerce_threshold: float = 0.95,
) -> NormalizeMissingResult:
    """Превращает sentinel-строки в np.nan и собирает отчёт о пропусках.

    Sentinel ищем только в object-колонках. В числовых колонках строки уже
    бы превратились в NaN при чтении CSV, либо колонка стала бы object из-за
    смешения типов — этот случай мы тоже обрабатываем здесь.

    Реальные NaN не считаются совпадением с токеном "nan": проверка
    isinstance(x, str) внутри карты пропускает их.

    Параметры:
        df: исходный DataFrame.
        na_tokens: набор строк, которые надо считать пропусками.
                   None → DEFAULT_NA_TOKENS. Сравнение регистронезависимое,
                   после strip.
        coerce_numeric: True (дефолт) — после замены токенов на NaN, для каждой
                   object-колонки попробовать pd.to_numeric. Если ≥coerce_threshold
                   значений парсятся в число — колонка конвертируется в Int64
                   (если все целые) или float64. Это лечит ситуацию когда числа
                   лежат как строки из-за грязи в CSV. False — оставить как есть.
        coerce_threshold: доля non-null значений которые должны парситься в число
                   чтобы запустить авто-конвертацию. Дефолт 0.95.

    Возвращает NormalizeMissingResult с нормализованной копией df и отчётом.
    """
    raw_tokens = na_tokens if na_tokens is not None else DEFAULT_NA_TOKENS
    tokens = frozenset(_normalize_token(t) for t in raw_tokens)

    df_out = df.copy()
    column_stats: list[ColumnMissingStats] = []

    for col in df_out.columns:
        series = df_out[col]
        token_hits: dict[str, int] = {}

        if series.dtype == object:
            normalized = series.map(
                lambda x: _normalize_token(x) if isinstance(x, str) else None
            )
            mask = normalized.isin(tokens)
            if mask.any():
                token_hits = {
                    str(k): int(v)
                    for k, v in normalized[mask].value_counts().to_dict().items()
                }
                df_out.loc[mask, col] = np.nan

        n_missing = int(df_out[col].isna().sum())
        if n_missing > 0:
            column_stats.append(ColumnMissingStats(
                column=str(col),
                n_missing=n_missing,
                fraction_missing=n_missing / len(df_out) if len(df_out) > 0 else 0.0,
                found_tokens=token_hits,
            ))

    coerced: list[str] = []
    if coerce_numeric:
        for col in df_out.columns:
            if df_out[col].dtype != object:
                continue
            non_null = df_out[col].dropna()
            if len(non_null) == 0:
                continue
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().sum() / len(non_null) >= coerce_threshold:
                converted = pd.to_numeric(df_out[col], errors="coerce")
                clean = converted.dropna()
                if len(clean) > 0 and (clean == clean.astype(int)).all():
                    df_out[col] = converted.astype("Int64")
                else:
                    df_out[col] = converted
                coerced.append(str(col))

    report = MissingReport(
        n_rows=len(df_out),
        n_columns=len(df_out.columns),
        total_missing=int(df_out.isna().sum().sum()),
        columns_with_missing=column_stats,
        na_tokens_used=tokens,
        coerced_numeric_columns=coerced,
    )
    return NormalizeMissingResult(df=df_out, report=report)


@dataclass
class DroppedColumn:
    """Информация об одной удалённой колонке."""

    column: str
    n_missing: int
    fraction_missing: float


@dataclass
class DropSparseReport:
    """Отчёт о drop_sparse_columns."""

    threshold: float
    dropped_columns: list[DroppedColumn]
    kept_columns: list[str]


@dataclass
class DropSparseResult:
    df: pd.DataFrame
    report: DropSparseReport


def drop_sparse_columns(
    df: pd.DataFrame,
    threshold: float = 0.5,
    columns: list[str] | None = None,
) -> DropSparseResult:
    """Удаляет колонки, в которых доля пропусков >= threshold.

    Колонка с большим числом пропусков несёт мало информации и при заполнении
    модой/медианой создаёт фейковый сигнал в большинстве строк. Например,
    в Titanic колонка `body` пуста на 90.8% — заполнение её медианой даст
    одно и то же значение в ~1180 строках из 1309.

    Запускается ПОСЛЕ normalize_missing, чтобы sentinel-строки уже были
    превращены в NaN и попали в подсчёт. И ДО detect_outliers, чтобы не
    тратить время на анализ колонок, которые мы всё равно выкинем.

    Параметры:
        df: DataFrame.
        threshold: порог в [0, 1]. Колонка удаляется при
                   n_missing / n_rows >= threshold.
                   Дефолт 0.5 — удалять при половине и более пропусков.
        columns: ограничить рассмотрение этими колонками. None → все колонки df.

    Возвращает DropSparseResult с уменьшенным df и отчётом, включая полный
    список удалённых колонок с их статистикой.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold должен быть в [0, 1], получено: {threshold}")

    candidates = (
        list(df.columns) if columns is None
        else [c for c in columns if c in df.columns]
    )
    n_rows = max(len(df), 1)

    dropped: list[DroppedColumn] = []
    drop_names: set[str] = set()
    for col in candidates:
        n_missing = int(df[col].isna().sum())
        frac = n_missing / n_rows
        if frac >= threshold:
            dropped.append(DroppedColumn(
                column=str(col),
                n_missing=n_missing,
                fraction_missing=frac,
            ))
            drop_names.add(col)

    kept = [c for c in df.columns if c not in drop_names]
    df_out = df[kept].copy() if dropped else df.copy()

    return DropSparseResult(
        df=df_out,
        report=DropSparseReport(
            threshold=threshold,
            dropped_columns=dropped,
            kept_columns=[str(c) for c in kept],
        ),
    )


@dataclass
class DropRowsReport:
    """Отчёт об удалении строк через drop_rows."""

    column: str
    n_rows_before: int
    n_rows_after: int
    n_dropped: int
    reasons: dict[str, int]


@dataclass
class DropRowsResult:
    df: pd.DataFrame
    report: DropRowsReport


def drop_rows(
    df: pd.DataFrame,
    column: str,
    *,
    drop_values: list | None = None,
    keep_values: list | None = None,
    keep_range: tuple[float, float] | None = None,
    drop_numeric: bool = False,
    drop_non_numeric: bool = False,
    drop_null: bool = False,
) -> DropRowsResult:
    """Удаляет строки по условию на одну колонку.

    Все критерии можно комбинировать; строка удаляется если попала хотя бы
    под одно условие. В отчёте видно сколько строк зацепил каждый критерий
    (одна и та же строка может попасть под несколько — в reasons считаются
    независимо).

    Критерии:
        drop_values:       список значений — удалить строки где column в нём
        keep_values:       обратное — оставить только строки где column в этом
                           списке; остальное удалить (NaN не трогаем если не
                           включён drop_null)
        keep_range:        (lo, hi) — оставить только строки где значение
                           числовое и lo ≤ value ≤ hi; остальное удалить
        drop_numeric:      удалить строки где column читается как число
                           (т.е. value — не категория). Полезно когда в
                           категориальной колонке встречаются цифры-загрязнения.
        drop_non_numeric:  обратное — удалить строки где column НЕ читается
                           как число. Полезно когда в числовой колонке
                           встречаются "yes"/"no" из-за бага загрузки.
        drop_null:         удалить строки где column = NaN

    Пример::

        # Оставить только Age в разумном диапазоне
        drop_rows(df, "Age", keep_range=(0, 100))

        # Выкинуть строки где expenditure не получилось распарсить как число
        drop_rows(df, "expenditure", drop_non_numeric=True)

        # Оставить только пассажиров 1 и 2 класса
        drop_rows(df, "Pclass", keep_values=[1, 2])
    """
    if column not in df.columns:
        raise KeyError(f"Колонка {column!r} не найдена в df")

    n_before = len(df)
    s = df[column]
    drop_mask = pd.Series(False, index=df.index)
    reasons: dict[str, int] = {}

    if drop_values is not None:
        m = s.isin(drop_values)
        if m.any():
            reasons["drop_values"] = int(m.sum())
        drop_mask |= m

    if keep_values is not None:
        m = ~s.isin(keep_values) & s.notna()
        if m.any():
            reasons["not_in_keep_values"] = int(m.sum())
        drop_mask |= m

    if keep_range is not None:
        lo, hi = keep_range
        numeric = pd.to_numeric(s, errors="coerce")
        m = ((numeric < lo) | (numeric > hi) | numeric.isna()) & s.notna()
        if m.any():
            reasons["outside_range"] = int(m.sum())
        drop_mask |= m

    if drop_numeric:
        numeric = pd.to_numeric(s, errors="coerce")
        m = numeric.notna() & s.notna()
        if m.any():
            reasons["is_numeric"] = int(m.sum())
        drop_mask |= m

    if drop_non_numeric:
        numeric = pd.to_numeric(s, errors="coerce")
        m = numeric.isna() & s.notna()
        if m.any():
            reasons["is_non_numeric"] = int(m.sum())
        drop_mask |= m

    if drop_null:
        m = s.isna()
        if m.any():
            reasons["is_null"] = int(m.sum())
        drop_mask |= m

    df_out = df.loc[~drop_mask].reset_index(drop=True)
    n_after = len(df_out)

    return DropRowsResult(
        df=df_out,
        report=DropRowsReport(
            column=column,
            n_rows_before=n_before,
            n_rows_after=n_after,
            n_dropped=n_before - n_after,
            reasons=reasons,
        ),
    )


ImputeStrategy = Literal[
    "auto", "mean", "median", "mode", "constant", "knn", "iterative", "drop"
]


@dataclass
class ColumnImputation:
    """Что было сделано с одной колонкой при заполнении."""

    column: str
    strategy_used: str
    fill_value: object | None
    n_imputed: int


@dataclass
class ImputationReport:
    """Отчёт о заполнении пропусков."""

    requested_strategy: str
    columns_imputed: list[ColumnImputation]
    rows_dropped: int = 0
    indicator_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImputationResult:
    df: pd.DataFrame
    report: ImputationReport


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _safe_fillna(df: pd.DataFrame, col: str, fill) -> None:
    """fillna со страховкой от TypeError на nullable Int64 + дробный fill.

    Если колонка int (включая Int64) и fill — нецелое число — расширяем
    до float64. Иначе pandas падает: "Invalid value '3.5' for dtype 'Int64'".
    """
    s = df[col]
    if (
        pd.api.types.is_integer_dtype(s)
        and isinstance(fill, (int, float, np.integer, np.floating))
        and not float(fill).is_integer()
    ):
        df[col] = s.astype("Float64" if str(s.dtype).startswith("Int") else "float64")
    df[col] = df[col].fillna(fill)


def _select_target_columns(
    df: pd.DataFrame, columns: list[str] | None
) -> list[str]:
    """Возвращает список колонок для обработки: только те, где реально есть пропуски."""
    if columns is None:
        return [c for c in df.columns if df[c].isna().any()]
    return [c for c in columns if c in df.columns and df[c].isna().any()]


def _fill_with_mode(
    df_out: pd.DataFrame,
    col: str,
    column_imputations: list[ColumnImputation],
    warnings: list[str],
) -> None:
    """Заполняет колонку самым частым значением. Используется как fallback
    для категориальных колонок при стратегиях knn/iterative."""
    n_missing = int(df_out[col].isna().sum())
    modes = df_out[col].mode(dropna=True)
    if len(modes) == 0:
        warnings.append(f"Колонка {col!r}: нет значений для вычисления моды, пропущена.")
        return
    fill = modes.iloc[0]
    _safe_fillna(df_out, col, fill)
    column_imputations.append(ColumnImputation(
        column=str(col), strategy_used="mode", fill_value=fill, n_imputed=n_missing,
    ))


def impute(
    df: pd.DataFrame,
    strategy: ImputeStrategy = "auto",
    columns: list[str] | None = None,
    constant_value: object = None,
    knn_neighbors: int = 5,
    knn_weights: Literal["uniform", "distance"] = "uniform",
    scale_for_knn: bool = True,
    add_indicator: bool = False,
) -> ImputationResult:
    """Заполняет пропуски выбранной стратегией.

    Стратегии:
        "auto":      median для числовых колонок, mode для категориальных.
                     Безопасный дефолт — медиана робастна к выбросам.
        "mean":      среднее (только числовые; категориальные пропускаются с warning).
        "median":    медиана (только числовые).
        "mode":      самое частое значение (любые типы).
        "constant":  заполнить значением constant_value (требуется параметр).
        "knn":       KNNImputer от sklearn на числовых колонках; категориальные
                     заполняются модой. Расстояние считается по ВСЕМ числовым
                     колонкам (даже не входящим в `columns`), чтобы найти соседей.
        "iterative": IterativeImputer (MICE) на числовых, mode на категориальных.
        "drop":      удалить строки, где есть пропуск в `columns` (или в любой
                     колонке, если columns=None).

    Параметры:
        columns: какие колонки заполнять. None → все с пропусками.
                 Колонки без пропусков игнорируются автоматически.
        constant_value: значение для strategy="constant". Обязательно.
        knn_neighbors: k для KNN.
        knn_weights: "uniform" — равные веса соседей; "distance" — обратно
                     пропорционально расстоянию (ближние влияют сильнее).
        scale_for_knn: нормировать колонки (z-score, игнорируя NaN) перед KNN
                       и обращать после. Без этого признак с большой амплитудой
                       подавит остальные в евклидовой метрике. Рекомендуется True.
        add_indicator: добавить колонку <col>_was_missing (тип int8: 1 — было
                       пропущено, 0 — было заполнено) для каждой обрабатываемой.

    Возвращает ImputationResult с заполненным DataFrame и отчётом.
    """
    df_out = df.copy()
    target_cols = _select_target_columns(df_out, columns)
    column_imputations: list[ColumnImputation] = []
    warnings: list[str] = []
    rows_dropped = 0
    indicator_cols: list[str] = []

    if strategy == "constant" and constant_value is None:
        raise ValueError(
            "Для strategy='constant' необходимо передать constant_value."
        )

    if not target_cols and strategy != "drop":
        return ImputationResult(
            df=df_out,
            report=ImputationReport(
                requested_strategy=strategy,
                columns_imputed=[],
            ),
        )

    if add_indicator:
        for col in target_cols:
            ind_name = f"{col}_was_missing"
            df_out[ind_name] = df_out[col].isna().astype("int8")
            indicator_cols.append(ind_name)

    if strategy == "drop":
        cols_to_check = (
            target_cols if columns is not None
            else [c for c in df_out.columns if c not in indicator_cols]
        )
        before = len(df_out)
        df_out = df_out.dropna(subset=cols_to_check).reset_index(drop=True)
        rows_dropped = before - len(df_out)

    elif strategy == "auto":
        for col in target_cols:
            n_missing = int(df_out[col].isna().sum())
            series = df_out[col]
            if _is_numeric(series):
                fill = series.median()
                used = "median"
            else:
                modes = series.mode(dropna=True)
                if len(modes) == 0:
                    warnings.append(f"Колонка {col!r}: нет значений для моды, пропущена.")
                    continue
                fill = modes.iloc[0]
                used = "mode"
            _safe_fillna(df_out, col, fill)
            column_imputations.append(ColumnImputation(
                column=str(col), strategy_used=used, fill_value=fill, n_imputed=n_missing,
            ))

    elif strategy in ("mean", "median", "mode", "constant"):
        for col in target_cols:
            n_missing = int(df_out[col].isna().sum())
            series = df_out[col]

            if strategy == "mean":
                if not _is_numeric(series):
                    warnings.append(f"Колонка {col!r} не числовая, пропущена для strategy='mean'.")
                    continue
                fill = series.mean()
            elif strategy == "median":
                if not _is_numeric(series):
                    warnings.append(f"Колонка {col!r} не числовая, пропущена для strategy='median'.")
                    continue
                fill = series.median()
            elif strategy == "mode":
                modes = series.mode(dropna=True)
                if len(modes) == 0:
                    warnings.append(f"Колонка {col!r}: нет значений для моды, пропущена.")
                    continue
                fill = modes.iloc[0]
            else:
                if constant_value is None:
                    raise ValueError(
                        "Для strategy='constant' необходимо передать constant_value."
                    )
                fill = constant_value

            _safe_fillna(df_out, col, fill)
            column_imputations.append(ColumnImputation(
                column=str(col), strategy_used=strategy, fill_value=fill, n_imputed=n_missing,
            ))

    elif strategy == "knn":
        try:
            from sklearn.impute import KNNImputer
        except ImportError as exc:
            raise ImportError(
                "Для strategy='knn' требуется scikit-learn. "
                "Установите: pip install scikit-learn"
            ) from exc

        numeric_targets = [c for c in target_cols if _is_numeric(df_out[c])]
        non_numeric_targets = [c for c in target_cols if not _is_numeric(df_out[c])]
        warn_numeric_objects(df_out, non_numeric_targets, "impute(strategy='knn')")

        if numeric_targets:
            all_numeric = df_out.select_dtypes(include=[np.number]).columns.tolist()
            n_missing_per_col = {c: int(df_out[c].isna().sum()) for c in numeric_targets}

            X = df_out[all_numeric].to_numpy(dtype=float)

            if scale_for_knn:
                col_means = np.nanmean(X, axis=0)
                col_stds = np.nanstd(X, axis=0)
                col_stds = np.where(col_stds == 0, 1.0, col_stds)
                X_proc = (X - col_means) / col_stds
            else:
                X_proc = X

            imputer = KNNImputer(n_neighbors=knn_neighbors, weights=knn_weights)
            X_imp = imputer.fit_transform(X_proc)

            if scale_for_knn:
                X_imp = X_imp * col_stds + col_means

            col_to_idx = {c: i for i, c in enumerate(all_numeric)}
            for col in numeric_targets:
                df_out[col] = X_imp[:, col_to_idx[col]]
                column_imputations.append(ColumnImputation(
                    column=str(col), strategy_used="knn", fill_value=None,
                    n_imputed=n_missing_per_col[col],
                ))

        for col in non_numeric_targets:
            _fill_with_mode(df_out, col, column_imputations, warnings)

    elif strategy == "iterative":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
        except ImportError as exc:
            raise ImportError(
                "Для strategy='iterative' требуется scikit-learn."
            ) from exc

        numeric_targets = [c for c in target_cols if _is_numeric(df_out[c])]
        non_numeric_targets = [c for c in target_cols if not _is_numeric(df_out[c])]
        warn_numeric_objects(df_out, non_numeric_targets, "impute(strategy='iterative')")

        if numeric_targets:
            all_numeric = df_out.select_dtypes(include=[np.number]).columns.tolist()
            n_missing_per_col = {c: int(df_out[c].isna().sum()) for c in numeric_targets}

            X = df_out[all_numeric].to_numpy(dtype=float)
            imputer = IterativeImputer(random_state=0)
            X_imp = imputer.fit_transform(X)

            col_to_idx = {c: i for i, c in enumerate(all_numeric)}
            for col in numeric_targets:
                df_out[col] = X_imp[:, col_to_idx[col]]
                column_imputations.append(ColumnImputation(
                    column=str(col), strategy_used="iterative", fill_value=None,
                    n_imputed=n_missing_per_col[col],
                ))

        for col in non_numeric_targets:
            _fill_with_mode(df_out, col, column_imputations, warnings)

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy!r}")

    return ImputationResult(
        df=df_out,
        report=ImputationReport(
            requested_strategy=strategy,
            columns_imputed=column_imputations,
            rows_dropped=rows_dropped,
            indicator_columns=indicator_cols,
            warnings=warnings,
        ),
    )


def apply_impute(
    df: pd.DataFrame,
    imputation: ImputationResult | ImputationReport,
) -> pd.DataFrame:
    """Заполняет пропуски значениями, посчитанными на train (без утечки).

    Берёт fill_value из impute(train).report и заполняет ими этот df (обычно
    test) — те же числа, что и на train. Так test не влияет на свои же
    медианы/моды, и пайплайн применим к одиночному объекту на инференсе.

    Симметрична apply_scale: impute(train) → fit, apply_impute(test) → apply.

    Переносятся только стратегии с единым значением замены
    (mean/median/mode/constant и auto). Колонки со стратегией без fill_value
    (knn/iterative/drop — там fill_value=None) пропускаются: переносить
    нечего, для них нужен отдельный fit на train. Колонки, которых нет в df,
    тоже пропускаются. Если в train у колонки не было пропусков, она не попала
    в отчёт — её пропуски в test останутся (нет train-значения для переноса).

    Параметры:
        df:         выборка для заполнения (обычно test).
        imputation: результат impute(train) — ImputationResult или его .report.

    Возвращает новый DataFrame (исходный не модифицируется).

    Пример::

        imp   = impute(train, strategy="median", columns=["age", "income"])
        train = imp.df
        test  = apply_impute(test, imp)
    """
    report = imputation.report if isinstance(imputation, ImputationResult) else imputation
    df_out = df.copy()
    for ci in report.columns_imputed:
        if ci.fill_value is None:
            continue
        if ci.column not in df_out.columns:
            continue
        _safe_fillna(df_out, ci.column, ci.fill_value)
    return df_out
