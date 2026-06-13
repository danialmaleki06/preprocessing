"""Преобразования значений колонок (log и пр.).

Выделено в отдельный модуль, потому что трансформации — это самостоятельная
концепция: они применяются и до outlier-детекции (для тяжелохвостых
распределений), и до ML-моделей (нормализация распределения), и просто для
EDA. Не привязаны исключительно к outlier-обработке.

Типичный сценарий:
    >>> from preprocessing.transforms import log_transform, inverse_log_transform
    >>> from preprocessing.outliers import detect_outliers, handle_outliers
    >>>
    >>> tr = log_transform(df, columns=["fare"])
    >>> det = detect_outliers(tr.df, method="iqr", columns=["fare"])
    >>> hand = handle_outliers(tr.df, det, strategy="clip")
    >>> df_back = inverse_log_transform(hand.df, tr)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from preprocessing._typing_utils import looks_datetime, looks_numeric, warn_numeric_objects


LogMethod = Literal["log1p", "signed_log"]


@dataclass
class ColumnTransform:
    """Информация об одной преобразованной колонке."""

    column: str
    method: str


@dataclass
class TransformReport:
    """Отчёт о применённых преобразованиях."""

    method: str
    transformed_columns: list[ColumnTransform]
    skipped_columns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class TransformResult:
    df: pd.DataFrame
    report: TransformReport


def log_transform(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: LogMethod = "log1p",
) -> TransformResult:
    """Применяет логарифмическое преобразование к указанным колонкам.

    Зачем: тяжелохвостые положительные распределения (доходы, цены, длительности,
    население) после log-трансформации становятся ближе к нормальному, и
    outlier-методы (IQR, z-score) перестают помечать легитимные хвосты как
    ошибки. Например, на Titanic `fare` варьируется от 0 до 512$, и IQR находит
    171 «выброс» в первом классе. После log1p диапазон становится 0..6.2,
    и IQR находит реальных аномалий гораздо меньше.

    Методы:
        "log1p":      np.log1p(x) = ln(1 + x). Требует x >= -1.
                      Идеально для неотрицательных значений (включая нули).
                      Обратная: np.expm1.
        "signed_log": np.sign(x) * np.log1p(|x|). Работает для любых значений,
                      сохраняя знак. Полезно когда есть отрицательные
                      (например, изменения цен).

    Параметры:
        columns: какие колонки преобразовать. None → все числовые.
        method: тип логарифма.

    Колонки, которые нельзя преобразовать (нечисловые / log1p со значениями
    < -1), попадают в `report.skipped_columns` с указанием причины и НЕ
    меняются в выходном df.

    Возвращает TransformResult с преобразованным df и отчётом, по которому
    можно сделать обратное преобразование через inverse_log_transform.
    """
    if columns is None:
        target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        target_cols = list(columns)

    df_out = df.copy()
    transformed: list[ColumnTransform] = []
    skipped: list[tuple[str, str]] = []

    for col in target_cols:
        if col not in df_out.columns:
            skipped.append((str(col), "колонка отсутствует в df"))
            continue
        s = df_out[col]
        if not pd.api.types.is_numeric_dtype(s):
            skipped.append((str(col), "не числовая колонка"))
            continue

        if method == "log1p":
            min_val = s.min(skipna=True)
            if pd.notna(min_val) and min_val < -1:
                skipped.append((
                    str(col),
                    f"log1p требует x >= -1, но min={min_val:.4f}; "
                    f"используйте method='signed_log'",
                ))
                continue
            df_out[col] = np.log1p(s)
        elif method == "signed_log":
            df_out[col] = np.sign(s) * np.log1p(s.abs())
        else:
            raise ValueError(f"Неизвестный method: {method!r}")

        transformed.append(ColumnTransform(column=str(col), method=method))

    _skipped_as_non_numeric = [c for c, r in skipped if "не числовая" in r]
    if _skipped_as_non_numeric:
        warn_numeric_objects(df_out, _skipped_as_non_numeric, "log_transform")

    return TransformResult(
        df=df_out,
        report=TransformReport(
            method=method,
            transformed_columns=transformed,
            skipped_columns=skipped,
        ),
    )


def inverse_log_transform(
    df: pd.DataFrame,
    transform_result: TransformResult,
) -> pd.DataFrame:
    """Обращает log_transform по информации из transform_result.report.

    Применяет обратную функцию к тем колонкам, которые были преобразованы:
        log1p:      x = np.expm1(y) = exp(y) - 1
        signed_log: x = sign(y) * (exp(|y|) - 1)

    Колонки, упомянутые в transformed_columns, но отсутствующие в df,
    тихо пропускаются (это нормально: пользователь мог удалить колонку
    через drop_sparse_columns между трансформациями).

    Возвращает копию df с обращёнными колонками.
    """
    df_out = df.copy()
    method = transform_result.report.method

    for ct in transform_result.report.transformed_columns:
        col = ct.column
        if col not in df_out.columns:
            continue
        s = df_out[col]
        if method == "log1p":
            df_out[col] = np.expm1(s)
        elif method == "signed_log":
            df_out[col] = np.sign(s) * np.expm1(s.abs())
        else:
            raise ValueError(f"Неизвестный method в TransformResult: {method!r}")

    return df_out


@dataclass
class OneHotReport:
    """Отчёт о применённом OneHot encoding."""
    encoded_columns: list[str]
    new_columns: list[str]
    skipped_columns: list[tuple[str, str]]


@dataclass
class OneHotResult:
    df: pd.DataFrame
    report: OneHotReport


def onehot_encode(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    max_unique: int = 50,
    drop_first: bool = False,
) -> OneHotResult:
    """OneHot encoding категориальных колонок.

    Каждое уникальное значение колонки становится отдельной бинарной колонкой
    вида `колонка_значение`. Оригинальная колонка удаляется.

    Параметры:
        columns:    список колонок для кодирования. None → все object/category.
        max_unique: пропустить колонку если уникальных значений больше этого
                    порога (защита от случайного кодирования ID или текста).
        drop_first: удалять первую dummy-колонку чтобы избежать мультиколлинеарности
                    (нужно для линейных моделей).

    Пример::

        result = onehot_encode(df)
        result = onehot_encode(df, ["gender", "city"], drop_first=True)
        df_encoded = result.df
    """
    if columns is None:
        target_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        target_cols = list(columns)

    df_out = df.copy()
    encoded: list[str] = []
    new_cols: list[str] = []
    skipped: list[tuple[str, str]] = []

    for col in target_cols:
        if col not in df_out.columns:
            skipped.append((col, "колонка отсутствует в df"))
            continue
        n_unique = df_out[col].nunique(dropna=True)
        if n_unique > max_unique:
            skipped.append((col, f"слишком много уникальных значений ({n_unique} > {max_unique})"))
            continue
        if n_unique == 0:
            skipped.append((col, "все значения NaN"))
            continue

        dummies = pd.get_dummies(df_out[col], prefix=col, drop_first=drop_first, dtype=int)
        new_cols.extend(dummies.columns.tolist())
        df_out = pd.concat([df_out.drop(columns=[col]), dummies], axis=1)
        encoded.append(col)

    return OneHotResult(
        df=df_out,
        report=OneHotReport(
            encoded_columns=encoded,
            new_columns=new_cols,
            skipped_columns=skipped,
        ),
    )


@dataclass
class DateParseReport:
    """Отчёт о разборе дат."""
    parsed_columns: list[str]
    extracted_features: list[str]
    skipped_columns: list[tuple[str, str]]


@dataclass
class DateParseResult:
    df: pd.DataFrame
    report: DateParseReport


def parse_dates(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    features: list[str] | None = None,
    drop_original: bool = True,
) -> DateParseResult:
    """Разбирает колонки с датами на числовые признаки.

    Из каждой колонки с датой извлекаются отдельные числовые колонки.
    Оригинальная колонка по умолчанию удаляется.

    Доступные признаки (features):
        "year"        — год (2024)
        "month"       — месяц (1–12)
        "day"         — день месяца (1–31)
        "weekday"     — день недели (0=пн, 6=вс)
        "quarter"     — квартал (1–4)
        "hour"        — час (0–23), если есть время
        "is_weekend"  — 1 если суббота или воскресенье, иначе 0

    Параметры:
        columns:       список колонок. None → все object-колонки похожие на дату.
        features:      какие признаки извлекать. None → все семь выше.
        drop_original: удалять исходную колонку (по умолчанию True).

    Пример::

        result = parse_dates(df, ["birth_date", "order_date"])
        result = parse_dates(df, ["created_at"], features=["year", "month", "weekday"])
        df_parsed = result.df
    """
    _all_features = ["year", "month", "day", "weekday", "quarter", "hour", "is_weekend"]
    active_features = features if features is not None else _all_features

    if columns is None:
        target_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
            or (df[c].dtype == object and looks_datetime(df[c]))
        ]
    else:
        target_cols = list(columns)

    df_out = df.copy()
    parsed: list[str] = []
    extracted: list[str] = []
    skipped: list[tuple[str, str]] = []

    for col in target_cols:
        if col not in df_out.columns:
            skipped.append((col, "колонка отсутствует в df"))
            continue

        if not pd.api.types.is_datetime64_any_dtype(df_out[col]):
            converted = pd.to_datetime(df_out[col], errors="coerce")
            n_failed = converted.isna().sum() - df_out[col].isna().sum()
            if converted.isna().all():
                skipped.append((col, "не удалось распарсить ни одну дату"))
                continue
            if n_failed > 0:
                skipped.append((col, f"частично распарсено ({n_failed} значений не удалось)"))
            df_out[col] = converted

        dt = df_out[col].dt
        feature_map = {
            "year":       (f"{col}_year",       dt.year),
            "month":      (f"{col}_month",      dt.month),
            "day":        (f"{col}_day",         dt.day),
            "weekday":    (f"{col}_weekday",     dt.weekday),
            "quarter":    (f"{col}_quarter",     dt.quarter),
            "hour":       (f"{col}_hour",        dt.hour),
            "is_weekend": (f"{col}_is_weekend",  dt.weekday.isin([5, 6]).astype(int)),
        }

        for feat in active_features:
            if feat not in feature_map:
                continue
            new_col, values = feature_map[feat]
            df_out[new_col] = values
            extracted.append(new_col)

        if drop_original:
            df_out = df_out.drop(columns=[col])

        parsed.append(col)

    return DateParseResult(
        df=df_out,
        report=DateParseReport(
            parsed_columns=parsed,
            extracted_features=extracted,
            skipped_columns=skipped,
        ),
    )


@dataclass
class ConvertedColumn:
    """Информация об одной приведённой колонке."""

    column: str
    dtype_before: str
    dtype_after: str
    n_coerced_to_nan: int


@dataclass
class ToNumericReport:
    """Отчёт to_numeric."""

    converted_columns: list[ConvertedColumn]
    skipped_columns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ToNumericResult:
    df: pd.DataFrame
    report: ToNumericReport


def to_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    as_int: bool = False,
) -> ToNumericResult:
    """Приводит указанные колонки к числовому типу.

    Что не парсится → NaN. Значения NaN сохраняются (через nullable Int64
    или обычный float64). Удобно после ``normalize_missing`` для колонок
    которые остались ``object`` из-за смешанных типов.

    Параметры:
        columns: список колонок для конвертации.
                 None → авто: только object-колонки которые ВЫГЛЯДЯТ числовыми
                 (≥95% значений парсятся в число). Категориальный текст
                 ("M"/"F", названия городов) пропускается — иначе он
                 превратился бы в сплошные NaN. Чтобы принудительно
                 сконвертировать конкретную колонку, передайте её явно в columns.
        as_int: True → результат в Int64 (nullable int). Если есть дробные
                значения после конвертации — колонка пропускается с warning,
                чтобы не потерять данные. False (дефолт) → float64.

    Пример::

        tn = to_numeric(df, columns=["age"], as_int=True)
        df = tn.df
        # age теперь Int64, NaN сохранены как <NA>
    """
    df_out = df.copy()
    converted: list[ConvertedColumn] = []
    skipped: list[tuple[str, str]] = []

    if columns is None:
        target = []
        for c in df_out.columns:
            if df_out[c].dtype != object:
                continue
            if looks_numeric(df_out[c]):
                target.append(c)
            else:
                skipped.append((str(c), "не выглядит числовой (columns=None пропускает)"))
    else:
        target = []
        for c in columns:
            if c not in df_out.columns:
                raise KeyError(f"Колонка {c!r} не найдена")
            target.append(c)

    for col in target:
        s = df_out[col]
        dtype_before = str(s.dtype)
        n_non_null_before = int(s.notna().sum())

        numeric = pd.to_numeric(s, errors="coerce")
        n_non_null_after = int(numeric.notna().sum())
        n_coerced = n_non_null_before - n_non_null_after

        if as_int:
            clean = numeric.dropna()
            if len(clean) > 0 and not np.all(clean == clean.astype(int)):
                skipped.append((col, "as_int=True, но в колонке есть дробные значения"))
                continue
            df_out[col] = numeric.astype("Int64")
        else:
            df_out[col] = numeric

        converted.append(ConvertedColumn(
            column=str(col),
            dtype_before=dtype_before,
            dtype_after=str(df_out[col].dtype),
            n_coerced_to_nan=n_coerced,
        ))

    return ToNumericResult(
        df=df_out,
        report=ToNumericReport(
            converted_columns=converted,
            skipped_columns=skipped,
        ),
    )


ScaleMethod = Literal["standard", "minmax", "robust", "maxabs"]


@dataclass
class ColumnScaling:
    """Параметры масштабирования одной колонки: scaled = (x - center) / scale.

    Хранятся чтобы применить ТЕ ЖЕ числа к test через apply_scale (без утечки).
    """

    column: str
    center: float
    scale: float


@dataclass
class ScaleReport:
    """Отчёт scale. scaled_columns содержит параметры для apply_scale."""

    method: str
    scaled_columns: list[ColumnScaling]
    skipped_columns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ScaleResult:
    df: pd.DataFrame
    report: ScaleReport


def _scale_params(clean: pd.Series, method: ScaleMethod) -> tuple[float, float]:
    """Считает (center, scale) для колонки по выбранному методу."""
    if method == "standard":
        return float(clean.mean()), float(clean.std(ddof=0))
    if method == "minmax":
        return float(clean.min()), float(clean.max() - clean.min())
    if method == "robust":
        return float(clean.median()), float(clean.quantile(0.75) - clean.quantile(0.25))
    if method == "maxabs":
        return 0.0, float(clean.abs().max())
    raise ValueError(f"Неизвестный метод: {method!r}")


def scale(
    df: pd.DataFrame,
    method: ScaleMethod = "standard",
    columns: list[str] | None = None,
) -> ScaleResult:
    """Масштабирует числовые колонки. Считает параметры по ЭТОМУ df (train).

    Все методы приводят к виду ``(x - center) / scale``:
        "standard": center=mean, scale=std — z-score, среднее 0, дисперсия 1.
        "minmax":   center=min, scale=(max-min) — в диапазон [0, 1].
        "robust":   center=median, scale=IQR (Q3-Q1) — устойчив к выбросам.
        "maxabs":   center=0, scale=max(|x|) — в [-1, 1], сохраняет нули.

    Параметры (center/scale) каждой колонки сохраняются в report.scaled_columns,
    чтобы потом применить ТЕ ЖЕ числа к тестовой выборке через apply_scale —
    это исключает утечку (test не должен влиять на свои же mean/std).

    Колонка пропускается (skipped_columns с причиной), если она:
        - не числовая (object с числами → warning с советом to_numeric)
        - bool
        - все значения NaN
        - вырожденная (scale=0: все значения одинаковы)

    Параметры:
        columns: какие колонки масштабировать. None → все числовые.
        method:  метод из списка выше.

    Пример (правильный workflow без утечки)::

        sp = split_dataset(df, by="target")
        sc = scale(sp.part_a, method="standard")        # fit на train
        train = sc.df
        test  = apply_scale(sp.part_b, sc)              # те же mean/std на test
    """
    df_out = df.copy()
    scaled_cols: list[ColumnScaling] = []
    skipped: list[tuple[str, str]] = []

    if columns is None:
        target = df_out.select_dtypes(include=[np.number]).columns.tolist()
        target = [c for c in target if not pd.api.types.is_bool_dtype(df_out[c])]
        skipped_object = [
            c for c in df_out.columns
            if c not in target and not pd.api.types.is_bool_dtype(df_out[c])
        ]
        warn_numeric_objects(df_out, skipped_object, "scale")
    else:
        target = []
        for c in columns:
            if c not in df_out.columns:
                raise KeyError(f"Колонка {c!r} не найдена")
            target.append(c)

    for col in target:
        s = df_out[col]
        if pd.api.types.is_bool_dtype(s):
            skipped.append((str(col), "bool-колонка"))
            continue
        if not pd.api.types.is_numeric_dtype(s):
            skipped.append((str(col), "не числовая колонка"))
            warn_numeric_objects(df_out, [col], "scale")
            continue
        clean = s.dropna()
        if clean.empty:
            skipped.append((str(col), "все значения NaN"))
            continue

        center, scale_val = _scale_params(clean, method)
        if scale_val == 0:
            skipped.append((str(col), f"вырожденная колонка (scale=0 для '{method}')"))
            continue

        df_out[col] = (pd.to_numeric(s, errors="coerce") - center) / scale_val
        scaled_cols.append(ColumnScaling(column=str(col), center=center, scale=scale_val))

    return ScaleResult(
        df=df_out,
        report=ScaleReport(method=method, scaled_columns=scaled_cols, skipped_columns=skipped),
    )


def apply_scale(
    df: pd.DataFrame,
    scaling: ScaleResult | ScaleReport,
) -> pd.DataFrame:
    """Применяет к df параметры масштабирования, посчитанные на train.

    Берёт center/scale из scale(train) и применяет ТЕ ЖЕ числа к этому df
    (обычно test). Так test не влияет на свою нормализацию — нет утечки.

    Параметры:
        df:      выборка для масштабирования (обычно test).
        scaling: результат scale(train) — ScaleResult или его .report.

    Колонки, которых нет в df, пропускаются. Возвращает новый DataFrame.

    Пример::

        sc = scale(train, method="standard")
        test_scaled = apply_scale(test, sc)
    """
    report = scaling.report if isinstance(scaling, ScaleResult) else scaling
    df_out = df.copy()
    for cs in report.scaled_columns:
        if cs.column not in df_out.columns:
            continue
        s = pd.to_numeric(df_out[cs.column], errors="coerce")
        df_out[cs.column] = (s - cs.center) / cs.scale
    return df_out
