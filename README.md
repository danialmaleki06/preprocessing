<h1 align="center">🧹 preprocessing</h1>

<p align="center">
  <b>Универсальный пайплайн предобработки и оценки качества данных для ML.</b><br>
  От сырого CSV до готовой к обучению выборки — <b>без утечки данных by&nbsp;design.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/tests-34%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version">
  <img src="https://img.shields.io/badge/code%20style-docstrings-informational" alt="Style">
</p>

<p align="center">
  <a href="#features">Возможности</a> ·
  <a href="#quickstart">Быстрый старт</a> ·
  <a href="#no-leak">Без утечки</a> ·
  <a href="#install">Установка</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

---

<a id="quickstart"></a>

## ▶️ Быстрый старт

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

<a id="features"></a>

## 📊 Ключевые возможности

| | |
|---|---|
| 🔒 **Ноль утечки** | `fit`/`apply`-пары: статистика по train, те же числа на test |
| 🧾 **Аудит каждого шага** | отчёт: что изменилось, сколько ячеек, какие колонки и почему |
| 📥 **Умная загрузка CSV** | авто-определение кодировки (UTF-8/16/32, BOM), разделителя, сжатия |
| 🎯 **11 методов выбросов** | MAD, IQR, z-score, Isolation Forest, LOF, ECOD, Mahalanobis… |
| 🧬 **Импутация без наивности** | median / mode / KNN / MICE; выбросы — до KNN, чтобы не искажать метрику |
| 📈 **16 видов графиков** | гистограммы, violin, тепловые карты, scatter выбросов, корреляции |
| 🔍 **Профилирование** | тип, хвосты, модальность, рекомендованный метод выбросов по колонке |
| ✅ **Покрыт тестами** | 34 теста: загрузка, импутация, выбросы, масштабирование, разбиение |

<a id="no-leak"></a>

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

Так же устроены импутация (`impute` / `apply_impute`) и выбросы (`detect_outliers` / `apply_outliers`).
Параметры сохраняются в отчёте — пайплайн воспроизводимо применяется хоть к одному объекту на проде.

## 🧰 Что умеет — по категориям

<details open>
<summary><b>📥 Загрузка</b></summary>

- `load_csv` — чтение CSV с авто-определением кодировки, разделителя и сжатия
</details>

<details open>
<summary><b>🕳️ Пропуски</b></summary>

- `normalize_missing` — строки-заглушки → `NaN` + авто-приведение типов
- `drop_sparse_columns`, `drop_rows` — удаление по доле пропусков / условию
- `impute` / `apply_impute` — заполнение (median/mode/KNN/MICE) без утечки
</details>

<details open>
<summary><b>🎯 Выбросы</b></summary>

- `detect_outliers` — 11 методов (одномерные + многомерные)
- `handle_outliers` — стратегии `clip` / `drop` / `mark_missing`
- `apply_outliers` — перенос границ train на test
</details>

<details open>
<summary><b>🔧 Преобразования</b></summary>

- `log_transform`, `onehot_encode`, `parse_dates`, `to_numeric`
- `scale` / `apply_scale` — standard / minmax / robust / maxabs без утечки
</details>

<details open>
<summary><b>✂️ Разбиение и 🔍 анализ</b></summary>

- `split_dataset` — стратифицированный train/test
- `profile_columns`, `count_variations`, `rank_correlations`, `compare_datasets`
</details>

<a id="install"></a>

## 🛠️ Установка

```bash
git clone https://github.com/danialmaleki06/preprocessing.git
cd preprocessing
pip install -e .
```

## 🧩 Пайплайн целиком

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

## 🗺️ Модули

| Модуль | Зачем |
|---|---|
| `loader` | Загрузка CSV: кодировка, разделитель, сжатие — автоматически |
| `missing` | Пропуски: нормализация, удаление, импутация |
| `transforms` | Логарифм, one-hot, даты, приведение к числу, масштабирование |
| `outliers` | Обнаружение (11 методов) и обработка выбросов |
| `splitting` | Стратифицированное разбиение train/test |
| `distribution` | Профиль колонок, кардинальность, сравнение датасетов |
| `pipeline` | Цепочка шагов с авто-аудитом |
| `heatmap` · `scatter` · `plots` | 16 видов графиков для разведки |

## 🧪 Тесты

```bash
pip install -e ".[dev]"
pytest
```

<a id="roadmap"></a>
