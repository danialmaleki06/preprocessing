# preprocessing

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Универсальный пайплайн предобработки и оценки качества табличных данных для машинного обучения.

Библиотека закрывает путь от сырого CSV до готовой к обучению выборки: автоопределение кодировки и разделителя, нормализация пропусков, обработка выбросов (11 методов), преобразования признаков, масштабирование и разбиение на train/test — **со строгой защитой от утечки данных** и аудит-отчётом по каждому шагу.

## Чем отличается

- **Без утечки данных.** Все статистические шаги разделены на пару `fit` / `apply`: параметры считаются по train и применяются к test теми же числами. Это `impute` / `apply_impute`, `scale` / `apply_scale`, `detect_outliers` / `apply_outliers`.
- **Аудит каждого шага.** Каждая функция возвращает не только данные, но и отчёт-`dataclass`: что именно изменилось, сколько ячеек заполнено, какие колонки удалены и почему. `format_report` превращает любой отчёт в читаемый текст.
- **Надёжная загрузка.** `load_csv` сам определяет кодировку (UTF-8/16/32, BOM, charset-normalizer), разделитель и сжатие (`.gz`, `.zip`, `.bz2`, `.xz`).
- **Богатая разведка.** Профилирование колонок, кардинальность, ранжирование корреляций и 17 функций визуализации (гистограммы, violin, тепловые карты, scatter выбросов).

## Установка

```bash
git clone https://github.com/danialmaleki06/preprocessing.git
cd preprocessing
pip install -e .
```

Или только зависимости:

```bash
pip install -r requirements.txt
```

## Быстрый старт

```python
import preprocessing as pp

load = pp.load_csv("data/titanic.csv")
df = load.df
print(pp.format_load_summary(load.report))

df = pp.normalize_missing(df).df
df = pp.drop_sparse_columns(df, threshold=0.5).df

sp = pp.split_dataset(df, by="survived", test_size=0.2)
train, test = sp.part_a, sp.part_b

imp = pp.impute(train, strategy="median")
train = imp.df
test = pp.apply_impute(test, imp)

sc = pp.scale(train, method="standard", columns=["age", "fare"])
train = sc.df
test = pp.apply_scale(test, sc)
```

## Защита от утечки данных

Главный принцип библиотеки: тест не должен влиять на собственную предобработку.
Сначала параметр считается по train (`fit`), затем теми же числами применяется к test (`apply`).

```python
sp = pp.split_dataset(df, by="target")
train, test = sp.part_a, sp.part_b

sc = pp.scale(train, method="standard")
train_scaled = sc.df
test_scaled = pp.apply_scale(test, sc)

det = pp.detect_outliers(train, method="iqr")
train = pp.handle_outliers(train, det, strategy="clip").df
test = pp.apply_outliers(test, det)
```

## Pipeline

Шаги можно собрать в последовательный пайплайн с автоматическим аудитом:

```python
from preprocessing import Pipeline, normalize_missing, drop_sparse_columns, impute

pipe = (
    Pipeline()
    .step("missing", normalize_missing)
    .step("drop_sparse", drop_sparse_columns, threshold=0.5)
    .step("impute", impute, strategy="median")
)

result = pipe.run(df)
print(pipe.audit())
```

## Обзор модулей

| Модуль | Назначение |
|---|---|
| `loader` | Загрузка CSV: автоопределение кодировки, разделителя, сжатия |
| `missing` | Нормализация пропусков, удаление разреженных колонок, импутация |
| `transforms` | Логарифм, one-hot, разбор дат, приведение к числу, масштабирование |
| `outliers` | Обнаружение (11 методов) и обработка выбросов |
| `splitting` | Стратифицированное разбиение на train/test |
| `distribution` | Профилирование колонок, кардинальность, сравнение датасетов |
| `pipeline` | Последовательный пайплайн с аудит-отчётом |
| `filters` | Поиск строк/ячеек по условию, точечная замена на NaN |
| `formatters` | Превращение любого отчёта в читаемый текст |
| `heatmap`, `scatter`, `plots` | Визуализация распределений, корреляций и выбросов |

Полный список функций — в docstrings (`help(pp.impute)`).

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

