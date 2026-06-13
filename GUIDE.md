# 📖 Руководство по `preprocessing`

Подробная инструкция: как устроена библиотека, как собрать пайплайн предобработки
и полный справочник функций с параметрами.

## Содержание

- [Философия: почему «без утечки»](#философия)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Как устроен пайплайн](#как-устроен-пайплайн)
- [Сборка пайплайна классом `Pipeline`](#сборка-пайплайна)
- [Справочник функций](#справочник-функций)
  - [Загрузка](#загрузка)
  - [Пропуски](#пропуски)
  - [Преобразования](#преобразования)
  - [Выбросы](#выбросы)
  - [Разбиение](#разбиение)
  - [Анализ и профилирование](#анализ-и-профилирование)
  - [Фильтры](#фильтры)
  - [Визуализация](#визуализация)
  - [Отчёты](#отчёты)
- [Сквозной пример](#сквозной-пример)
- [Частые ошибки](#частые-ошибки)

---

<a id="философия"></a>

## Философия: почему «без утечки»

**Утечка данных (data leakage)** — когда информация из тестовой выборки незаметно
попадает в обучение. Самый частый источник — предобработка, посчитанная по всему
датасету до разбиения:

- заполнили пропуски **медианой всего датасета** → медиана «знает» про тест;
- отмасштабировали по **среднему и std всего датасета** → масштаб «знает» про тест;
- обрезали выбросы по **границам всего датасета** → границы «знают» про тест.

Результат — завышенная оценка на тесте, которая разваливается в проде.

Решение в этой библиотеке — **разделение на `fit` и `apply`**:

| fit (считает параметры по train) | apply (применяет к test) |
|---|---|
| `impute(train, ...)` | `apply_impute(test, result)` |
| `scale(train, ...)` | `apply_scale(test, result)` |
| `detect_outliers(train, ...)` | `apply_outliers(test, detection)` |

Параметры (медианы, средние, границы) сохраняются в `.report` и переиспользуются —
поэтому пайплайн воспроизводимо применяется хоть к одному объекту на инференсе.

---

<a id="установка"></a>

## Установка

```bash
git clone https://github.com/danialmaleki06/preprocessing.git
cd preprocessing
pip install -e .
```

Для запуска тестов:

```bash
pip install -e ".[dev]"
pytest
```

---

<a id="быстрый-старт"></a>

## Быстрый старт

```python
import preprocessing as pp

# 1. Загрузка (кодировка и разделитель определяются автоматически)
df = pp.load_csv("data/titanic.csv").df

# 2. Очистка структуры (на всём датасете — это безопасно, утечки нет)
df = pp.normalize_missing(df).df
df = pp.drop_sparse_columns(df, threshold=0.5).df

# 3. Разбиение
sp = pp.split_dataset(df, by="survived")
train, test = sp.part_a, sp.part_b

# 4. Статистические шаги: fit на train → apply на test
imp = pp.impute(train, strategy="median")
train, test = imp.df, pp.apply_impute(test, imp)

sc = pp.scale(train, method="standard")
train, test = sc.df, pp.apply_scale(test, sc)
```

Почти каждая функция возвращает **Result-объект** с двумя полями:

- `.df` — преобразованный `DataFrame`;
- `.report` — отчёт о том, что произошло (для аудита и для `apply_*`).

---

<a id="как-устроен-пайплайн"></a>

## Как устроен пайплайн

Порядок шагов важен. Рекомендованный конвейер:

```
load_csv
   ↓
normalize_missing          ← привести заглушки к NaN, типы
   ↓
drop_sparse_columns        ← убрать почти пустые колонки
   ↓
detect_outliers + handle   ← обработать выбросы ДО импутации
   ↓
impute                     ← заполнить пропуски
   ↓
─────── split_dataset ───────   (граница утечки)
   ↓
onehot_encode / scale      ← fit на train, apply на test
```

**Два правила:**

1. **Выбросы — до импутации.** KNN/MICE-импьютеры считают расстояния между строками;
   невырезанные выбросы исказят эти расстояния.
2. **Статистические шаги (`impute`, `scale`, `detect_outliers`) — после `split`,**
   через пары `fit`/`apply`. Структурные шаги (`normalize_missing`,
   `drop_sparse_columns`, `parse_dates`) можно делать до split — они не «подсматривают».

---

<a id="сборка-пайплайна"></a>

## Сборка пайплайна классом `Pipeline`

`Pipeline` — последовательный исполнитель шагов с авто-аудитом. Он удобен для
**дослит-овой части** (структурная очистка), где утечки нет.

```python
from preprocessing import (
    Pipeline, normalize_missing, drop_sparse_columns, log_transform,
)

pipe = (
    Pipeline()
    .step("missing",     normalize_missing)
    .step("drop_sparse", drop_sparse_columns, threshold=0.5)
    .step("log_fare",    log_transform, columns=["fare"])
)

result = pipe.run(df)     # PipelineResult: .df и .steps
clean  = result.df
print(pipe.audit())       # таблица: шаг, форма до/после, время, что произошло
```

### Методы `Pipeline`

| Метод | Что делает |
|---|---|
| `.step(name, func, **params)` | Добавляет шаг. Возвращает `self` (можно в цепочку). `func` принимает `df` первым аргументом и возвращает `DataFrame` или Result-объект. |
| `.run(df)` | Выполняет все шаги по очереди. Исходный `df` не меняется. Возвращает `PipelineResult` (`.df`, `.steps`). |
| `.audit()` | `DataFrame`-сводка: имя шага, форма до/после, время, краткое summary. |
| `.steps()` | Список имён шагов. |

> ⚠️ Класс `Pipeline` сам по себе **не** разделяет train/test. Для защиты от утечки
> либо прогоняйте через `Pipeline` только дослит-овую часть, либо после `split`
> используйте пары `fit`/`apply` вручную (см. [сквозной пример](#сквозной-пример)).

### Готовый шаг для выбросов

`detect_and_handle_outliers` объединяет обнаружение и обработку в один шаг —
специально, чтобы вставлять его в `Pipeline`:

```python
pipe.step("outliers", pp.detect_and_handle_outliers, method="iqr", strategy="clip")
```

---

<a id="справочник-функций"></a>

## Справочник функций

Везде `df` — `pandas.DataFrame`. Параметры после `*` передаются только по имени
(keyword-only).

<a id="загрузка"></a>

### Загрузка

#### `load_csv(path, *, encoding=None, separator=None, header=0) -> LoadResult`
Загружает CSV с автоопределением кодировки, разделителя и сжатия.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `path` | `str \| Path` | — | Путь к файлу (`.csv`, `.gz`, `.zip`, `.bz2`, `.xz`). |
| `encoding` | `str \| None` | `None` | Кодировка. `None` → автоопределение (UTF-8/16/32, BOM, cp1251, latin-1). |
| `separator` | `str \| None` | `None` | Разделитель. `None` → автоопределение (`,` `;` `\t` `\|`). |
| `header` | `int \| None` | `0` | Номер строки-заголовка. `None` → файл без заголовка. |

Возврат: `LoadResult` (`.df`, `.report`). Сводку даёт `format_load_summary(result.report)`.

#### `format_load_summary(report) -> str`
Текстовый отчёт о загрузке (кодировка, разделитель, форма, предупреждения).

<a id="пропуски"></a>

### Пропуски

#### `normalize_missing(df, na_tokens=None, *, coerce_numeric=True, coerce_threshold=0.95) -> NormalizeMissingResult`
Превращает строки-заглушки (`"N/A"`, `"unknown"`, `"-999"`, …) в `NaN` и собирает
отчёт о пропусках.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `na_tokens` | `set[str] \| None` | `None` | Набор заглушек. `None` → встроенный `DEFAULT_NA_TOKENS`. |
| `coerce_numeric` | `bool` | `True` | После очистки пытаться привести object-колонку к числу. |
| `coerce_threshold` | `float` | `0.95` | Доля значений, которые должны парситься как число, чтобы колонка конвертировалась. |

#### `drop_sparse_columns(df, threshold=0.5, columns=None) -> DropSparseResult`
Удаляет колонки, где доля пропусков ≥ `threshold`.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Порог доли пропусков (0–1). `0.5` = удалить, если пусто ≥ половины. |
| `columns` | `list[str] \| None` | `None` | Какие колонки проверять. `None` → все. |

#### `drop_rows(df, column, *, drop_values=None, keep_values=None, keep_range=None, drop_numeric=False, drop_non_numeric=False, drop_null=False) -> DropRowsResult`
Удаляет строки по условию на одну колонку.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `column` | `str` | — | Колонка, по которой фильтруем. |
| `drop_values` | `list \| None` | `None` | Удалить строки с этими значениями. |
| `keep_values` | `list \| None` | `None` | Оставить только эти значения (остальные удалить). |
| `keep_range` | `tuple[float, float] \| None` | `None` | Оставить значения в диапазоне `(min, max)`. |
| `drop_numeric` | `bool` | `False` | Удалить строки, где значение числовое. |
| `drop_non_numeric` | `bool` | `False` | Удалить строки, где значение НЕ числовое. |
| `drop_null` | `bool` | `False` | Удалить строки, где значение `NaN`. |

#### `impute(df, strategy='auto', columns=None, constant_value=None, knn_neighbors=5, knn_weights='uniform', scale_for_knn=True, add_indicator=False) -> ImputationResult`
Заполняет пропуски выбранной стратегией.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `strategy` | `str` | `'auto'` | `auto` (median для чисел, mode для категорий), `mean`, `median`, `mode`, `constant`, `knn`, `iterative` (MICE), `drop`. |
| `columns` | `list[str] \| None` | `None` | Какие колонки заполнять. `None` → все с пропусками. |
| `constant_value` | `object` | `None` | Значение для `strategy='constant'` (обязательно). |
| `knn_neighbors` | `int` | `5` | Число соседей `k` для `knn`. |
| `knn_weights` | `'uniform' \| 'distance'` | `'uniform'` | Веса соседей: равные или обратно пропорц. расстоянию. |
| `scale_for_knn` | `bool` | `True` | Нормировать колонки перед KNN (иначе крупная амплитуда подавит остальные). |
| `add_indicator` | `bool` | `False` | Добавить колонку `<col>_was_missing` (1 — было пропущено). |

#### `apply_impute(df, imputation) -> pd.DataFrame`
Заполняет `df` (обычно test) значениями, посчитанными на train. Переносятся только
стратегии с единым значением (`mean`/`median`/`mode`/`constant`/`auto`);
`knn`/`iterative`/`drop` пропускаются.

<a id="преобразования"></a>

### Преобразования

#### `log_transform(df, columns=None, method='log1p') -> TransformResult`
Логарифмирует тяжёлые правые хвосты (доходы, цены).

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Числовые колонки. `None` → подходящие автоматически. |
| `method` | `'log1p' \| 'signed_log'` | `'log1p'` | `log1p` для неотрицательных; `signed_log` если есть отрицательные. |

#### `inverse_log_transform(df, transform_result) -> pd.DataFrame`
Обращает `log_transform` по данным из `transform_result.report`.

#### `onehot_encode(df, columns=None, *, max_unique=50, drop_first=False) -> OneHotResult`
Кодирует категориальные колонки в бинарные dummy-колонки.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Категориальные колонки. `None` → все object/category. |
| `max_unique` | `int` | `50` | Пропустить колонку, если уникальных значений больше (защита от кодирования ID). |
| `drop_first` | `bool` | `False` | Убрать первую dummy (против мультиколлинеарности для линейных моделей). |

#### `parse_dates(df, columns=None, *, features=None, drop_original=True) -> DateParseResult`
Извлекает из дат числовые признаки.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Колонки с датами. |
| `features` | `list[str] \| None` | `None` | Что извлечь: `year`, `month`, `day`, `weekday`, `is_weekend`, `quarter`. `None` → набор по умолчанию. |
| `drop_original` | `bool` | `True` | Удалить исходную колонку-дату. |

#### `to_numeric(df, columns=None, *, as_int=False) -> ToNumericResult`
Приводит колонки к числу; что не парсится → `NaN`.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Колонки для приведения. |
| `as_int` | `bool` | `False` | Целочисленный результат (`Int64`) вместо `float64`. |

#### `scale(df, method='standard', columns=None) -> ScaleResult`
Масштабирует числовые колонки. **Параметры считаются по этому `df` (train).**

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `method` | `str` | `'standard'` | `standard` (z-score), `minmax` ([0,1]), `robust` (медиана/IQR), `maxabs` ([-1,1]). |
| `columns` | `list[str] \| None` | `None` | Числовые колонки. `None` → все числовые. |

#### `apply_scale(df, scaling) -> pd.DataFrame`
Применяет к `df` (обычно test) параметры масштабирования, посчитанные на train.

<a id="выбросы"></a>

### Выбросы

#### `detect_outliers(df, method='mad', columns=None, threshold=None, percentile_bounds=(1.0, 99.0), contamination=0.1, min_unique_values=1, n_neighbors=20, gmm_components=2) -> OutlierDetectionResult`
Ищет выбросы в числовых колонках. 11 методов.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `method` | `str` | `'mad'` | Одномерные: `mad`, `iqr`, `zscore`, `percentile`. Многомерные: `isolation_forest`, `lof`, `one_class_svm`, `mahalanobis`, `gmm`, `ecod`, `copod`. |
| `columns` | `list[str] \| None` | `None` | Числовые колонки. `None` → все числовые. |
| `threshold` | `float \| None` | `None` | Порог для `mad`/`zscore`. `None` → дефолт метода (mad: 3.5, zscore: 3.0). |
| `percentile_bounds` | `tuple[float, float]` | `(1.0, 99.0)` | Нижний/верхний перцентиль для `percentile`. |
| `contamination` | `float` | `0.1` | Ожидаемая доля выбросов (для многомерных методов). |
| `min_unique_values` | `int` | `1` | Пропускать колонки с меньшим числом уникальных значений. |
| `n_neighbors` | `int` | `20` | Число соседей для `lof`. |
| `gmm_components` | `int` | `2` | Число компонент для `gmm`. |

Только одномерные методы дают границы (`.bounds`), пригодные для `clip` и `apply_outliers`.

#### `handle_outliers(df, detection, strategy='clip') -> OutlierHandlingResult`
Обрабатывает найденные выбросы.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `detection` | `OutlierDetectionResult` | — | Результат `detect_outliers`. |
| `strategy` | `str` | `'clip'` | `clip` (винзоризация к границам), `drop` (удалить строки), `mark_missing` (→ `NaN`), `keep` (ничего). |

#### `apply_outliers(df, detection) -> pd.DataFrame`
Клипирует `df` (test) по границам, посчитанным на train. Работает только для
одномерных методов (есть `.bounds`).

#### `detect_and_handle_outliers(df, *, method='mad', strategy='clip', ...)`
Обёртка `detect_outliers` + `handle_outliers` одним вызовом (для `Pipeline`).
Принимает все параметры детекции + `strategy`.

<a id="разбиение"></a>

### Разбиение

#### `split_dataset(df, by, *, test_size=0.2, seed=42, n_bins=10) -> SplitResult`
Делит `df` на две части со стратификацией по колонке `by`.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `by` | `str` | — | Колонка, пропорции которой сохраняются в обеих частях. |
| `test_size` | `float` | `0.2` | Доля строк во второй части (`part_b`). |
| `seed` | `int` | `42` | Зерно генератора для воспроизводимости. |
| `n_bins` | `int` | `10` | Число квантильных бинов, если `by` — непрерывная числовая. |

Возврат: `SplitResult` (`.part_a`, `.part_b`, `.report` с таблицей пропорций).

<a id="анализ-и-профилирование"></a>

### Анализ и профилирование

Возвращают `DataFrame` (read-only, данные не меняют).

#### `profile_columns(df, *, columns=None, normality_alpha=0.05, n_unique_discrete_threshold=20, kde_bandwidth=None) -> pd.DataFrame`
Профиль каждой колонки: тип, пропуски, скошенность, модальность, нормальность,
рекомендованный метод выбросов.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `columns` | `list[str] \| None` | `None` | Какие колонки профилировать. |
| `normality_alpha` | `float` | `0.05` | Уровень значимости для теста нормальности. |
| `n_unique_discrete_threshold` | `int` | `20` | Граница «дискретная/непрерывная» по числу уникальных. |
| `kde_bandwidth` | `float \| None` | `None` | Ширина окна KDE для подсчёта пиков (модальности). |

#### `count_variations(df, *, n_samples=5) -> pd.DataFrame`
Кардинальность: уникальные значения, типы, пропуски, примеры.

| `n_samples` | `int` | `5` | Сколько примеров значений показать. |

#### `rank_correlations(df, *, method='pearson', min_abs=0.0) -> pd.DataFrame`
Пары колонок, отсортированные по `|корреляции|`.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `method` | `str` | `'pearson'` | `pearson`, `spearman`, `kendall`. |
| `min_abs` | `float` | `0.0` | Отсечь пары с `|corr|` ниже порога. |

#### `compare_datasets(df_before, df_after) -> pd.DataFrame`
Что изменилось по каждой колонке между двумя версиями df (до/после).

<a id="фильтры"></a>

### Фильтры

#### `find_rows(df, column, *, is_numeric=False, is_non_numeric=False, in_range=None, equals=None, contains=None, pattern=None, case_sensitive=True) -> pd.DataFrame`
Строки, где `column` удовлетворяет хотя бы одному критерию.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `is_numeric` / `is_non_numeric` | `bool` | `False` | Значение является / не является числом. |
| `in_range` | `tuple[float, float] \| None` | `None` | Числовое значение в диапазоне. |
| `equals` | `list \| None` | `None` | Значение из списка. |
| `contains` | `str \| None` | `None` | Подстрока. |
| `pattern` | `str \| None` | `None` | Регулярное выражение. |
| `case_sensitive` | `bool` | `True` | Учитывать регистр для `contains`/`pattern`. |

#### `find_coords(df, *, value=None, is_numeric=False, is_non_numeric=False, in_range=None, contains=None, pattern=None, case_sensitive=True, columns=None) -> pd.DataFrame`
Координаты `(row, col)` всех ячеек, удовлетворяющих условию.

#### `set_cells_to_nan(df, coords, *, drop_empty_rows=True) -> SetCellsResult`
Точечно ставит выбранные ячейки в `NaN`.

| Параметр | Тип | По умолч. | Назначение |
|---|---|---|---|
| `coords` | `list[tuple[int, int\|str]]` | — | Список координат `(строка, колонка)`. |
| `drop_empty_rows` | `bool` | `True` | Удалить строки, ставшие полностью пустыми. |

<a id="визуализация"></a>

### Визуализация

Все рисуют через matplotlib (`plt.show()` или сохранение фигуры — на вашей стороне).
Общие параметры: `columns` (`'all'` или список), `figsize`, `title`.

| Функция | Назначение | Ключевые параметры |
|---|---|---|
| `plot_histogram(df, columns='all', *, bins=30, kde=True, figsize=(8,5))` | Гистограмма + KDE по каждой числовой колонке. | `bins`, `kde` |
| `plot_violin(df, columns='all', *, figsize=(4,6))` | Форма распределения через ширину. | `columns` |
| `plot_before_after(df_before, df_after, columns='all', *, figsize=(8,5))` | Наложение KDE «до» (синий) и «после» (красный). | два df |
| `plot_value_counts(df, columns='all', *, top=20, figsize=(8,5))` | Bar-plot частот категорий. | `top` |
| `plot_target_correlation(df, target, *, method='pearson', top=None, figsize=(8,6))` | Корреляция фич с таргетом (зелёный +, красный −). | `target`, `method`, `top` |
| `plot_outlier_consensus(df, pair, methods=None, *, contamination=0.05, figsize=(9,7))` | Scatter пары: цвет = сколько методов считают точку выбросом. | `pair`, `contamination` |
| `plot_outlier_ranking(detection, *, top=30, figsize=(8,8))` | Bar-plot числа выбросов по колонкам. Принимает результат `detect_outliers`. | `top` |
| `plot_missing_heatmap(df, *, figsize=None, title=...)` | Карта процента пропусков по колонкам. | — |
| `plot_minmax_heatmap(df, *, n=20, ...)` | Топ-N мин. и макс. значений. | `n` |
| `plot_correlation_heatmap(df, *, method='pearson', annot_threshold=20, ...)` | Матрица корреляций. | `method`, `annot_threshold` |
| `plot_outlier_heatmap(df, *, method='mad', threshold=3.5, iqr_k=1.5, top_n=10, ...)` | Процент выбросов по колонкам. | `method`, `threshold`, `iqr_k`, `top_n` |
| `plot_kde_scatter(df, columns='all', *, contamination=0.05, ...)` | KDE-контуры плотности по парам. | `contamination` |
| `plot_dbscan_scatter(df, columns='all', *, eps=0.5, min_samples=5, ...)` | DBSCAN-кластеры, шум = выбросы. | `eps`, `min_samples` |
| `plot_elliptic_scatter(df, columns='all', *, contamination=0.1, ...)` | Эллипс Elliptic Envelope. | `contamination` |
| `plot_strip(df, columns='all', *, alpha=0.35, point_size=4, color=...)` | Strip-plot всех точек. | `alpha`, `point_size` |
| `plot_outlier_scatter(df, columns='all', *, ...)` | Три метода (KDE+DBSCAN+Elliptic) для каждой пары. | параметры всех трёх |

<a id="отчёты"></a>

### Отчёты

#### `format_report(report) -> str`
Превращает любой Report-объект (из `.report`) в читаемый многострочный текст.

```python
res = pp.impute(df, strategy="median")
print(pp.format_report(res.report))
```

---

<a id="сквозной-пример"></a>

## Сквозной пример (без утечки)

```python
import preprocessing as pp

# ── Загрузка ────────────────────────────────────────────────
df = pp.load_csv("data/titanic.csv").df

# ── Дослит-овая очистка (структурная, утечки нет) ───────────
df = df.drop(columns=["name", "ticket"])           # идентификаторы
df = pp.normalize_missing(df).df                    # заглушки → NaN
df = pp.drop_sparse_columns(df, threshold=0.5).df   # уберёт cabin (77% пусто)
df = pp.log_transform(df, columns=["fare"]).df      # тяжёлый хвост

# ── Разбиение ───────────────────────────────────────────────
sp = pp.split_dataset(df, by="survived", test_size=0.2)
train, test = sp.part_a, sp.part_b

# ── Выбросы: fit на train → apply на test ──────────────────
det   = pp.detect_outliers(train, method="iqr", columns=["age", "fare"])
train = pp.handle_outliers(train, det, strategy="clip").df
test  = pp.apply_outliers(test, det)

# ── Импутация: fit на train → apply на test ────────────────
imp   = pp.impute(train, strategy="median", columns=["age"])
train = imp.df
test  = pp.apply_impute(test, imp)

# ── Кодирование категорий ──────────────────────────────────
train = pp.onehot_encode(train, columns=["sex", "embarked"]).df
test  = pp.onehot_encode(test,  columns=["sex", "embarked"]).df
test  = test.reindex(columns=train.columns, fill_value=0)   # выровнять колонки

# ── Масштабирование: fit на train → apply на test ──────────
sc    = pp.scale(train, method="standard", columns=["age", "fare"])
train = sc.df
test  = pp.apply_scale(test, sc)

print(train.shape, test.shape)   # готово к обучению модели
```

---

<a id="частые-ошибки"></a>

## Частые ошибки

| Ошибка | Почему плохо | Как правильно |
|---|---|---|
| `impute`/`scale` до `split` | Параметры «видят» тест → утечка | Сначала `split`, потом `fit`/`apply` |
| Импутация до обработки выбросов | KNN/MICE искажаются выбросами | Сначала выбросы, потом `impute` |
| `onehot_encode` отдельно на train и test | Разный набор dummy-колонок | После кодирования `test.reindex(columns=train.columns, fill_value=0)` |
| `clip` для `isolation_forest` | У многомерных методов нет границ | Используйте `drop`, либо одномерный метод для `clip` |
| `strategy='constant'` без `constant_value` | `ValueError` | Передайте `constant_value=...` |
```
