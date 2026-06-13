<h1 align="center">🧹 preprocessing</h1>

<p align="center">
  <b>Универсальный пайплайн предобработки и оценки качества данных для машинного обучения.</b><br>
  От сырого CSV до готовой к обучению выборки — <b>без утечки данных by&nbsp;design.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/tests-34%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version">
</p>

---

> **Ваша модель выдаёт 0.95 на тесте, а в проде — 0.78?**
> Чаще виновата не модель, а **утечка данных** в предобработке: тест незаметно «подсматривает» в обучение через общие медианы, масштабы и границы выбросов.
> Эта библиотека построена так, что утечка **невозможна** — каждый статистический шаг честно делится на `fit` (по train) и `apply` (на test).

## ✨ Что внутри

| | |
|---|---|
| 🔒 **Ноль утечки** | `fit`/`apply`-пары: статистика считается по train, теми же числами применяется к test |
| 🧾 **Аудит каждого шага** | каждая функция возвращает отчёт: что изменилось, сколько ячеек, какие колонки и почему |
| 📥 **Умная загрузка CSV** | авто-определение кодировки (UTF-8/16/32, BOM), разделителя и сжатия (`.gz`/`.zip`/`.bz2`/`.xz`) |
| 🎯 **11 методов выбросов** | от робастных MAD и IQR до Isolation Forest, LOF, ECOD, Mahalanobis |
| 🧬 **Импутация без наивности** | median / mode / KNN / MICE — выбросы обрабатываются до KNN, чтобы не искажать расстояния |
| 📊 **16 видов графиков** | гистограммы, violin, тепловые карты, scatter выбросов, корреляция с таргетом |
| 🔍 **Профилирование** | тип, хвосты, модальность и рекомендованный метод выбросов по каждой колонке |
| ✅ **Покрыт тестами** | 34 теста на загрузку, импутацию, выбросы, масштабирование и разбиение |

## ⚡ За 30 секунд

```python
import preprocessing as pp

df = pp.load_csv("data/titanic.csv").df           # кодировка и разделитель — сами
df = pp.normalize_missing(df).df                  # "N/A", "unknown", "-999" → NaN
df = pp.drop_sparse_columns(df, threshold=0.5).df

sp = pp.split_dataset(df, by="survived")          # стратифицированно
train, test = sp.part_a, sp.part_b

imp = pp.impute(train, strategy="median")         # медианы — по train
train, test = imp.df, pp.apply_impute(test, imp)  # те же медианы — на test
```

## 🔒 Главная идея: честная предобработка

Большинство туториалов незаметно сливают информацию из теста в обучение. Сравните:

```python
# ❌ Утечка: масштаб считается по всему датасету — тест «виден» модели
df[cols] = StandardScaler().fit_transform(df[cols])

# ✅ Честно: fit только на train, те же параметры применяются к test
sc = pp.scale(train, method="standard")
train = sc.df
test  = pp.apply_scale(test, sc)
```

То же — для импутации (`impute` / `apply_impute`) и выбросов (`detect_outliers` / `apply_outliers`).
Параметры сохраняются в отчёте, поэтому пайплайн воспроизводимо применяется хоть к одному объекту на проде.

## 📦 Установка

```bash
git clone https://github.com/danialmaleki06/preprocessing.git
cd preprocessing
pip install -e .
```

## 🧩 Собрать пайплайн целиком

```python
from preprocessing import Pipeline, normalize_missing, drop_sparse_columns, impute

pipe = (
    Pipeline()
    .step("missing", normalize_missing)
    .step("drop_sparse", drop_sparse_columns, threshold=0.5)
    .step("impute", impute, strategy="median")
)

result = pipe.run(df)
print(pipe.audit())   # таблица: шаг, форма до/после, время, что произошло
```

## 🗺️ Карта модулей

| Модуль | Зачем |
|---|---|
| `loader` | Загрузка CSV: кодировка, разделитель, сжатие — автоматически |
| `missing` | Нормализация пропусков, удаление разреженных колонок, импутация |
| `transforms` | Логарифм, one-hot, разбор дат, приведение к числу, масштабирование |
| `outliers` | Обнаружение (11 методов) и обработка выбросов (clip / drop / mark) |
| `splitting` | Стратифицированное разбиение на train/test |
| `distribution` | Профиль колонок, кардинальность, сравнение датасетов |
| `pipeline` | Цепочка шагов с авто-аудитом |
| `filters` | Поиск строк/ячеек по условию, точечная замена на NaN |
| `heatmap` · `scatter` · `plots` | 16 видов графиков для разведки |
| `formatters` | Любой отчёт → читаемый текст |

Полный справочник — в docstrings: `help(pp.impute)`.

## 🧪 Тесты

```bash
pip install -e ".[dev]"
pytest
```

## 🛣️ Roadmap

- [ ] **Data Quality Score** — интегральная оценка датасета 0–100
- [ ] Поиск дубликатов (точные + нечёткие)
- [ ] Детектор утечки таргета (фичи, подозрительно скоррелированные с ним)
- [ ] Экспорт пайплайна в **Airflow DAG**
- [ ] **Polars / PySpark** backend для больших данных
- [ ] CI (GitHub Actions) с авто-прогоном тестов
