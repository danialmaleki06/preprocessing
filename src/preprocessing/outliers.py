"""Поиск и обработка выбросов в числовых колонках."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from preprocessing._typing_utils import warn_numeric_objects

DetectMethod = Literal[
    "mad", "iqr", "zscore", "percentile",
    "isolation_forest", "lof", "one_class_svm", "mahalanobis", "gmm",
    "ecod", "copod",
]
HandleStrategy = Literal["clip", "drop", "mark_missing", "keep"]

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "mad": 3.5,
    "iqr": 1.5,
    "zscore": 3.0,
    "mahalanobis": 0.975,
}


@dataclass
class ColumnOutlierStats:
    """Результат поиска выбросов в одной колонке."""

    column: str
    n_outliers: int
    fraction_outliers: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass
class OutlierReport:
    """Отчёт о найденных выбросах."""

    method: str
    threshold: float | None
    columns_analyzed: list[str]
    columns_skipped: list[tuple[str, str]]
    column_stats: list[ColumnOutlierStats]
    total_outliers: int
    is_multivariate: bool = False


@dataclass
class OutlierDetectionResult:
    """Результат detect_outliers: маски выбросов + границы для clip."""

    df: pd.DataFrame
    report: OutlierReport
    masks: dict[str, pd.Series]
    bounds: dict[str, tuple[float, float]]


_DegenerateReason = str
_DetectionResult = tuple[pd.Series, float, float]


def _detect_mad(
    s: pd.Series, threshold: float
) -> _DetectionResult | _DegenerateReason:
    """Modified Z-score через MAD (медианное абсолютное отклонение).

    Робастная версия z-score: использует median и MAD вместо mean и std,
    поэтому сами выбросы не смещают оценку центра/разброса.

    Формула: 0.6745 * (x - median) / MAD; |z_mod| > threshold → выброс.
    Константа 0.6745 ≈ обратное Φ(0.75) и приводит MAD к "сравнимой" со std
    оценке для нормального распределения.

    Возвращает строку-причину, если MAD=0 (вырожденное распределение).
    """
    s_clean = s.dropna()
    median = s_clean.median()
    mad = (s_clean - median).abs().median()
    if mad == 0:
        return "вырожденное распределение (MAD=0): большинство значений совпадают с медианой"
    modified_z = 0.6745 * (s - median) / mad
    mask = modified_z.abs() > threshold
    bound = threshold * mad / 0.6745
    return mask.fillna(False), float(median - bound), float(median + bound)


def _detect_iqr(
    s: pd.Series, k: float
) -> _DetectionResult | _DegenerateReason:
    """IQR-метод (Тьюки): x вне [Q1 - k·IQR, Q3 + k·IQR] — выброс.

    Возвращает строку-причину, если IQR=0 (Q1=Q3): на дискретных счётчиках
    вроде parch/sibsp медиана и квартили часто совпадают, и метод вырождается
    в «всё ненулевое — выброс».
    """
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return "вырожденное распределение (IQR=0): Q1=Q3, метод неприменим"
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (s < lower) | (s > upper)
    return mask.fillna(False), float(lower), float(upper)


def _detect_zscore(
    s: pd.Series, threshold: float
) -> _DetectionResult | _DegenerateReason:
    """Классический z-score. Чувствителен к самим выбросам и предполагает
    нормальность — используйте, только если в этом уверены.

    Возвращает строку-причину, если std=0 (все значения одинаковые).
    """
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return "вырожденное распределение (std=0): все значения одинаковые"
    z = (s - mean) / std
    mask = z.abs() > threshold
    bound = threshold * std
    return mask.fillna(False), float(mean - bound), float(mean + bound)


def _detect_percentile(
    s: pd.Series, lower_pct: float, upper_pct: float
) -> _DetectionResult:
    """Перцентильное отсечение: x < P_lower или x > P_upper — выброс.

    Не вырождается: даже на константных данных квантили равны константе,
    маска получается пустой, что является корректным ответом «выбросов нет».
    """
    lower = s.quantile(lower_pct / 100)
    upper = s.quantile(upper_pct / 100)
    mask = (s < lower) | (s > upper)
    return mask.fillna(False), float(lower), float(upper)


def _filter_columns(
    df: pd.DataFrame,
    columns: list[str],
    min_unique_values: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Возвращает (валидные числовые колонки, список skipped с причинами).

    Применяется и для одномерных методов, и для isolation_forest, чтобы
    оба пути вели одинаковую фильтрацию.
    """
    valid: list[str] = []
    skipped: list[tuple[str, str]] = []
    for col in columns:
        if col not in df.columns:
            skipped.append((str(col), "колонка отсутствует в df"))
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            skipped.append((str(col), "не числовая колонка"))
            continue
        if pd.api.types.is_bool_dtype(s):
            skipped.append((str(col), "bool-колонка (выбросы не определены для 2 значений)"))
            continue
        clean = s.dropna()
        if clean.empty:
            skipped.append((str(col), "все значения NaN"))
            continue
        n_unique = clean.nunique()
        if n_unique < min_unique_values:
            skipped.append((
                str(col),
                f"низкая кардинальность ({n_unique} уник. значений < {min_unique_values})",
            ))
            continue
        valid.append(str(col))
    return valid, skipped


def detect_outliers(
    df: pd.DataFrame,
    method: DetectMethod = "mad",
    columns: list[str] | None = None,
    threshold: float | None = None,
    percentile_bounds: tuple[float, float] = (1.0, 99.0),
    contamination: float = 0.1,
    min_unique_values: int = 1,
    n_neighbors: int = 20,
    gmm_components: int = 2,
) -> OutlierDetectionResult:
    """Ищет выбросы в числовых колонках выбранным методом.

    Одномерные методы (по каждой колонке отдельно):
        "mad":        Modified Z-score через MAD. Threshold по умолчанию 3.5.
        "iqr":        Tukey's IQR. Threshold = коэффициент (по умолчанию 1.5).
        "zscore":     Классический z-score. Threshold = 3.0.
        "percentile": Отсечение по percentile_bounds.

    Многомерные методы (смотрят на всю строку как точку в n-мерном пространстве;
    одномерных границ нет → стратегия "clip" неприменима):
        "isolation_forest": Случайные разбиения дерева. Использует contamination.
        "lof":              Local Outlier Factor. Локальные выбросы внутри
                            кластеров. contamination + n_neighbors.
        "one_class_svm":    SVM с одним классом. Использует contamination как nu.
                            Медленный на больших датасетах.
        "mahalanobis":      Расстояние от центра с учётом ковариации.
                            threshold = перцентиль chi-square (по умолчанию 0.975).
        "gmm":              Gaussian Mixture Model. Точки с низкой log-likelihood
                            под смесью. contamination + gmm_components.
        "ecod":             Empirical CDF (PyOD). Без параметров кроме contamination,
                            быстрый, хорошее значение по умолчанию для табличных данных.
        "copod":            Copula-based (PyOD). Учитывает зависимости между колонками
                            через копулы. Тоже без параметров.

    Параметры:
        method: метод обнаружения.
        columns: список колонок. None → все числовые.
        threshold: порог для mad/iqr/zscore/mahalanobis. None → дефолт по методу.
        percentile_bounds: (lower_pct, upper_pct) для method="percentile".
        contamination: ожидаемая доля выбросов (0..0.5) для многомерных методов.
        min_unique_values: пропускать колонки с числом уникальных значений меньше
                           этого. Полезно поднять до 5–10, чтобы исключить
                           дискретные счётчики и dummy-колонки.
        n_neighbors: число соседей для LOF (по умолчанию 20).
        gmm_components: число гауссиан в смеси для GMM (по умолчанию 2).

    Колонки, не прошедшие проверки (не числовая / все NaN / низкая
    кардинальность / вырожденное распределение для выбранного метода),
    попадают в `report.columns_skipped` с указанием причины и НЕ
    участвуют в масках/границах.

    NaN не считаются выбросами; для одномерных методов они игнорируются
    при подсчёте границ. Для isolation_forest NaN временно заполняется
    медианой только для подачи в модель — исходный df не меняется.

    Возвращает OutlierDetectionResult с масками и границами.
    """
    if columns is None:
        analyzed = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        analyzed = list(columns)

    valid_cols, skipped = _filter_columns(df, analyzed, min_unique_values)
    _skipped_as_non_numeric = [c for c, r in skipped if "не числовая" in r]
    if _skipped_as_non_numeric:
        warn_numeric_objects(df, _skipped_as_non_numeric, "detect_outliers")

    masks: dict[str, pd.Series] = {}
    bounds: dict[str, tuple[float, float]] = {}
    column_stats: list[ColumnOutlierStats] = []

    if method == "isolation_forest":
        return _detect_isolation_forest(df, valid_cols, skipped, contamination)
    if method == "lof":
        return _detect_lof(df, valid_cols, skipped, contamination, n_neighbors)
    if method == "one_class_svm":
        return _detect_one_class_svm(df, valid_cols, skipped, contamination)
    if method == "mahalanobis":
        chi2_q = threshold if threshold is not None else _DEFAULT_THRESHOLDS["mahalanobis"]
        return _detect_mahalanobis(df, valid_cols, skipped, chi2_q)
    if method == "gmm":
        return _detect_gmm(df, valid_cols, skipped, contamination, gmm_components)
    if method == "ecod":
        return _detect_ecod(df, valid_cols, skipped, contamination)
    if method == "copod":
        return _detect_copod(df, valid_cols, skipped, contamination)

    eff_threshold = threshold if threshold is not None else _DEFAULT_THRESHOLDS.get(method)

    for col in valid_cols:
        s = df[col]

        if method == "mad":
            outcome = _detect_mad(s, eff_threshold)
        elif method == "iqr":
            outcome = _detect_iqr(s, eff_threshold)
        elif method == "zscore":
            outcome = _detect_zscore(s, eff_threshold)
        elif method == "percentile":
            outcome = _detect_percentile(s, *percentile_bounds)
        else:
            raise ValueError(f"Неизвестный метод: {method!r}")

        if isinstance(outcome, str):
            skipped.append((str(col), outcome))
            continue
        mask, lower, upper = outcome

        masks[str(col)] = mask
        bounds[str(col)] = (lower, upper)
        n_out = int(mask.sum())
        column_stats.append(ColumnOutlierStats(
            column=str(col),
            n_outliers=n_out,
            fraction_outliers=n_out / len(df) if len(df) > 0 else 0.0,
            lower_bound=lower,
            upper_bound=upper,
        ))

    total = sum(stat.n_outliers for stat in column_stats)
    return OutlierDetectionResult(
        df=df,
        report=OutlierReport(
            method=method,
            threshold=eff_threshold if method != "percentile" else None,
            columns_analyzed=[stat.column for stat in column_stats],
            columns_skipped=skipped,
            column_stats=column_stats,
            total_outliers=total,
        ),
        masks=masks,
        bounds=bounds,
    )


def _run_multivariate(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    method_name: str,
    threshold_for_report: float | None,
    fit_predict,
) -> OutlierDetectionResult:
    """Общая обвязка для многомерных методов.

    Все многомерные методы (isolation_forest, lof, one_class_svm, mahalanobis, gmm)
    работают по одной схеме: матрица из выбранных колонок (NaN временно заполнены
    медианой) → bool-маска по строкам. fit_predict — функция np.ndarray → np.ndarray[bool].

    Возвращает OutlierDetectionResult без одномерных границ (bounds={}) —
    стратегия clip к многомерным методам неприменима.
    """
    column_stats: list[ColumnOutlierStats] = []
    masks: dict[str, pd.Series] = {}

    if not valid_cols:
        return OutlierDetectionResult(
            df=df,
            report=OutlierReport(
                method=method_name,
                threshold=threshold_for_report,
                columns_analyzed=[],
                columns_skipped=skipped,
                column_stats=[],
                total_outliers=0,
            ),
            masks={},
            bounds={},
        )

    X = df[valid_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    is_outlier = fit_predict(X.to_numpy())
    global_mask = pd.Series(is_outlier, index=df.index)

    for c in valid_cols:
        col_mask = global_mask & df[c].notna()
        masks[str(c)] = col_mask
        n_out = int(col_mask.sum())
        column_stats.append(ColumnOutlierStats(
            column=str(c),
            n_outliers=n_out,
            fraction_outliers=n_out / len(df) if len(df) > 0 else 0.0,
            lower_bound=None,
            upper_bound=None,
        ))

    total = int(global_mask.sum())
    return OutlierDetectionResult(
        df=df,
        report=OutlierReport(
            method=method_name,
            threshold=threshold_for_report,
            columns_analyzed=valid_cols,
            columns_skipped=skipped,
            column_stats=column_stats,
            total_outliers=total,
            is_multivariate=True,
        ),
        masks=masks,
        bounds={},
    )


def _detect_isolation_forest(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    contamination: float,
) -> OutlierDetectionResult:
    """Изоляционный лес: редкие сочетания значений изолируются за меньше шагов."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ImportError("Для method='isolation_forest' требуется scikit-learn.") from exc

    def fit_predict(X):
        clf = IsolationForest(contamination=contamination, random_state=0)
        return clf.fit_predict(X) == -1

    return _run_multivariate(
        df, valid_cols, skipped, "isolation_forest", contamination, fit_predict,
    )


def _detect_lof(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    contamination: float,
    n_neighbors: int,
) -> OutlierDetectionResult:
    """Local Outlier Factor: точки с пониженной локальной плотностью.

    Считает плотность вокруг каждой точки и сравнивает с плотностью её
    n_neighbors соседей. Находит локальные выбросы — точки нормальные глобально,
    но аномальные внутри своего кластера.
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError as exc:
        raise ImportError("Для method='lof' требуется scikit-learn.") from exc

    def fit_predict(X):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        return clf.fit_predict(X) == -1

    return _run_multivariate(df, valid_cols, skipped, "lof", contamination, fit_predict)


def _detect_one_class_svm(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    nu: float,
) -> OutlierDetectionResult:
    """One-Class SVM: учит гиперплоскость отделяющую "нормальную" зону от всего.

    nu — верхняя граница доли выбросов (≈ contamination). Подходит для
    нелинейных границ. Медленный на больших датасетах (> 50k строк).
    """
    try:
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise ImportError("Для method='one_class_svm' требуется scikit-learn.") from exc

    def fit_predict(X):
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        return clf.fit_predict(X) == -1

    return _run_multivariate(df, valid_cols, skipped, "one_class_svm", nu, fit_predict)


def _detect_mahalanobis(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    chi2_quantile: float,
) -> OutlierDetectionResult:
    """Махаланобисово расстояние от центра облака с учётом ковариации.

    d² = (x - μ)ᵀ Σ⁻¹ (x - μ), где μ — среднее, Σ — ковариационная матрица.
    При нормальности d² распределено как χ² с df=k степенями свободы.
    Порог — chi2.ppf(chi2_quantile, df=k). По умолчанию 0.975.
    """
    try:
        from scipy.stats import chi2
    except ImportError as exc:
        raise ImportError("Для method='mahalanobis' требуется scipy.") from exc

    def fit_predict(X):
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        diff = X - mean
        dists_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        cutoff = chi2.ppf(chi2_quantile, df=X.shape[1])
        return dists_sq > cutoff

    return _run_multivariate(
        df, valid_cols, skipped, "mahalanobis", chi2_quantile, fit_predict,
    )


def _detect_gmm(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    contamination: float,
    n_components: int,
) -> OutlierDetectionResult:
    """Gaussian Mixture Model: смесь n_components гауссиан под данные.

    Точки с самой низкой log-likelihood (нижние contamination% от scores) —
    выбросы. Работает на мультимодальных данных где обычные методы ломаются.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("Для method='gmm' требуется scikit-learn.") from exc

    def fit_predict(X):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(X)
        scores = gmm.score_samples(X)
        cutoff = np.percentile(scores, contamination * 100)
        return scores < cutoff

    return _run_multivariate(df, valid_cols, skipped, "gmm", contamination, fit_predict)


def _detect_ecod(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    contamination: float,
) -> OutlierDetectionResult:
    """ECOD (Empirical CDF Outlier Detection) из PyOD.

    Для каждой колонки строит эмпирическую CDF, считает хвостовую вероятность
    значения. Итоговый score — сумма log(1/p) по всем колонкам. Без параметров
    (кроме contamination), очень быстрый, хорошо работает по умолчанию.
    """
    try:
        from pyod.models.ecod import ECOD
    except ImportError as exc:
        raise ImportError("Для method='ecod' требуется pyod.") from exc

    def fit_predict(X):
        clf = ECOD(contamination=contamination)
        clf.fit(X)
        return clf.labels_ == 1

    return _run_multivariate(df, valid_cols, skipped, "ecod", contamination, fit_predict)


def _detect_copod(
    df: pd.DataFrame,
    valid_cols: list[str],
    skipped: list[tuple[str, str]],
    contamination: float,
) -> OutlierDetectionResult:
    """COPOD (Copula-Based Outlier Detection) из PyOD.

    Строит копулу — модель совместного распределения через индивидуальные CDF.
    Score — хвостовая вероятность под копулой. Похож на ECOD но учитывает
    зависимости между колонками. Тоже без параметров, очень быстрый.
    """
    try:
        from pyod.models.copod import COPOD
    except ImportError as exc:
        raise ImportError("Для method='copod' требуется pyod.") from exc

    def fit_predict(X):
        clf = COPOD(contamination=contamination)
        clf.fit(X)
        return clf.labels_ == 1

    return _run_multivariate(df, valid_cols, skipped, "copod", contamination, fit_predict)


@dataclass
class OutlierHandlingReport:
    """Отчёт о применённой обработке выбросов."""

    strategy: str
    rows_dropped: int = 0
    cells_clipped: int = 0
    cells_marked_missing: int = 0
    columns_affected: list[str] = field(default_factory=list)


@dataclass
class OutlierHandlingResult:
    df: pd.DataFrame
    report: OutlierHandlingReport


def handle_outliers(
    df: pd.DataFrame,
    detection: OutlierDetectionResult,
    strategy: HandleStrategy = "clip",
) -> OutlierHandlingResult:
    """Обрабатывает выбросы согласно отчёту detect_outliers.

    Стратегии:
        "clip":         winsorize — выбросы заменяются на границы метода
                        (lower_bound / upper_bound из detection.bounds).
                        Размер выборки сохраняется. Неприменимо к
                        isolation_forest (у него нет одномерных границ).
        "drop":         удалить строки, в которых хоть одна колонка помечена
                        как выброс. Гарантированно убирает выбросы, но теряет
                        данные.
        "mark_missing": заменить выбросы на np.nan. Удобно перед impute() —
                        тогда выбросы и пропуски обрабатываются единым проходом.
        "keep":         ничего не делать (df возвращается копией без изменений).

    Параметры:
        df: DataFrame для обработки. Обычно тот же, что подавался в detect_outliers.
        detection: результат detect_outliers (содержит маски и границы).
        strategy: какую стратегию применить.

    Возвращает OutlierHandlingResult с обработанным df и отчётом.
    """
    df_out = df.copy()
    report = OutlierHandlingReport(strategy=strategy)
    affected: set[str] = set()

    if strategy == "keep":
        return OutlierHandlingResult(df=df_out, report=report)

    valid_mask_cols = {c for c in detection.masks.keys() if c in df_out.columns}

    if strategy == "drop":
        rows_to_drop = pd.Series(False, index=df_out.index)
        for col, mask in detection.masks.items():
            if col not in valid_mask_cols:
                continue
            if len(mask) != len(df_out):
                continue
            if mask.any():
                rows_to_drop = rows_to_drop | mask
                affected.add(col)
        before = len(df_out)
        df_out = df_out.loc[~rows_to_drop].reset_index(drop=True)
        report.rows_dropped = before - len(df_out)
        report.columns_affected = sorted(affected)
        return OutlierHandlingResult(df=df_out, report=report)

    if strategy == "mark_missing":
        if getattr(detection.report, "is_multivariate", False):
            raise ValueError(
                f"Стратегия 'mark_missing' неприменима к многомерному методу "
                f"'{detection.report.method}': выброс — это вся строка, а не "
                f"отдельная ячейка. Используйте strategy='drop' (удалить строки) "
                f"или 'keep'."
            )
        for col, mask in detection.masks.items():
            if col not in valid_mask_cols or len(mask) != len(df_out):
                continue
            n = int(mask.sum())
            if n > 0:
                df_out.loc[mask, col] = np.nan
                report.cells_marked_missing += n
                affected.add(col)
        report.columns_affected = sorted(affected)
        return OutlierHandlingResult(df=df_out, report=report)

    if strategy == "clip":
        if not detection.bounds:
            raise ValueError(
                "Стратегия 'clip' неприменима для метода без одномерных "
                "границ (например, isolation_forest). Используйте 'drop' "
                "или 'mark_missing'."
            )
        for col, (lower, upper) in detection.bounds.items():
            if col not in df_out.columns:
                continue
            mask = detection.masks.get(col)
            if mask is None or len(mask) != len(df_out) or not mask.any():
                continue
            n = int(mask.sum())
            if pd.api.types.is_integer_dtype(df_out[col]) and (
                not float(lower).is_integer() or not float(upper).is_integer()
            ):
                df_out[col] = df_out[col].astype(
                    "Float64" if str(df_out[col].dtype).startswith("Int") else "float64"
                )
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
            report.cells_clipped += n
            affected.add(col)
        report.columns_affected = sorted(affected)
        return OutlierHandlingResult(df=df_out, report=report)

    raise ValueError(f"Неизвестная стратегия: {strategy!r}")


def apply_outliers(
    df: pd.DataFrame,
    detection: OutlierDetectionResult,
) -> pd.DataFrame:
    """Клипирует df по границам, посчитанным на train (без утечки).

    Винзоризация test по train-лимитам: берёт (lower, upper) из
    detect_outliers(train).bounds и обрезает ими значения этого df (обычно
    test). Значения test за пределами train-границ заменяются на границу.
    Так test не влияет на свои же лимиты — нет утечки, и шаг применим к
    одиночному объекту на инференсе.

    В отличие от handle_outliers(strategy="clip"), НЕ использует маски
    выбросов (они посчитаны на train и привязаны к его строкам): обрезка идёт
    строго по границам, поэтому корректно применяется к данным любого размера.

    Подходит только для одномерных clip-методов (mad/iqr/zscore/percentile).
    Для многомерных методов без границ (isolation_forest и т.п.) bounds пуст —
    обрезать нечем, df возвращается без изменений. Колонки, которых нет в df,
    пропускаются.

    Параметры:
        df:        выборка для обрезки (обычно test).
        detection: результат detect_outliers(train) — содержит bounds.

    Возвращает новый DataFrame (исходный не модифицируется).

    Пример::

        det   = detect_outliers(train, method="iqr", columns=["session_minutes"])
        train = handle_outliers(train, det, strategy="clip").df
        test  = apply_outliers(test, det)
    """
    df_out = df.copy()
    for col, (lower, upper) in detection.bounds.items():
        if col not in df_out.columns:
            continue
        if pd.api.types.is_integer_dtype(df_out[col]) and (
            not float(lower).is_integer() or not float(upper).is_integer()
        ):
            df_out[col] = df_out[col].astype(
                "Float64" if str(df_out[col].dtype).startswith("Int") else "float64"
            )
        df_out[col] = df_out[col].clip(lower=lower, upper=upper)
    return df_out
