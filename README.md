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
  <img src="https://img.shields.io/badge/code%20style-docstrings-informational" alt="Style">
</p>

<p align="center">
  <a href="#why">Зачем</a> ·
  <a href="#features">Возможности</a> ·
  <a href="#quickstart">Быстрый старт</a> ·
  <a href="#no-leak">Без утечки</a> ·
  <a href="#install">Установка</a>
</p>

---
<a id="why"></a>

## 💡

Большинство инструментов решают задачи по отдельности: `pandas` чистит, `scikit-learn` масштабирует, профилировщики строят отчёты. Между этими шагами легко потерять главное — **контроль над тем, какая информация и в какой момент попадает в модель**.

Проект `preprocessing` объединяет весь путь — от чтения сырого CSV до готовой к обучению выборки — в один последовательный, прозрачный и воспроизводимый конвейер.
<a id="quickstart"></a>

## ▶️ Быстрый старт

Полный честный цикл — загрузка, очистка, стратифицированное разбиение и заполнение пропусков без утечки — умещается в несколько строк:

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

Библиотека закрывает все типовые этапы подготовки табличных данных и добавляет то, чего обычно не хватает — прозрачность и защиту от утечки:

| | |
|---|---|
| 🔒 **Ноль утечки** | `fit`/`apply`-пары: статистика по train, те же числа на test и на инференсе |
| 🧾 **Аудит каждого шага** | отчёт: что изменилось, сколько ячеек, какие колонки и почему |
| 📥 **Умная загрузка CSV** | авто-определение кодировки (UTF-8/16/32, BOM), разделителя и сжатия |
| 🎯 **11 методов выбросов** | MAD, IQR, z-score, Isolation Forest, LOF, ECOD, Mahalanobis и другие |
| 🧬 **Импутация без наивности** | median / mode / KNN / MICE; выбросы обрабатываются до KNN, чтобы не искажать метрику |
| 📈 **16 видов графиков** | гистограммы, violin, тепловые карты, scatter выбросов, корреляции с таргетом |
| 🔍 **Профилирование** | тип, хвосты, модальность и рекомендованный метод выбросов по каждой колонке |
| ✅ **Покрыт тестами** | 34 теста: загрузка, импутация, выбросы, масштабирование, разбиение |

<a id="no-leak"></a>

## 🔒 Главная идея: честная предобработка

### Что такое утечка данных

Утечка (*data leakage*) — это ситуация, когда при обучении модель получает доступ к информации, которой не будет в момент реального применения. Самый частый и коварный случай возникает именно в предобработке: вы считаете медиану, среднее, минимум-максимум или границы выбросов по **всему** датасету — включая тестовую часть — и только потом делите данные. В результате тест «подсматривает» собственные статистики, метрика на нём завышается, а в бою модель работает хуже, чем обещала валидация. Самое опасное в этой ошибке то, что она не вызывает исключений и не видна в коде — просто цифры выглядят слишком хорошо.

### Как библиотека это предотвращает

Каждый статистический шаг разделён на две операции:

- **`fit`** — параметры (медианы, масштабы, границы выбросов) считаются **только по обучающей выборке**;
- **`apply`** — те же самые параметры применяются к тесту и к новым данным на инференсе.

```python
# ❌ Утечка: масштаб считается по всему датасету — тест «виден» модели
df[cols] = StandardScaler().fit_transform(df[cols])

# ✅ Честно: fit только на train, те же параметры применяются к test
sc = pp.scale(train, method="standard")
train = sc.df
test  = pp.apply_scale(test, sc)
```

Так же устроены импутация (`impute` / `apply_impute`) и выбросы (`detect_outliers` / `apply_outliers`). Параметры не теряются — они сохраняются в отчёте шага, поэтому пайплайн можно дословно воспроизвести хоть на одном объекте в продакшене. Утечка предотвращена не дисциплиной пользователя, а самим устройством API.

## 🧠 Принципы дизайна

- **Прозрачность вместо магии.** Каждая функция возвращает не только данные, но и отчёт-объект: сколько ячеек заполнено, какие колонки удалены и почему, какие пропущены. Ничего не происходит «втихую».
- **Честность по умолчанию.** Безопасные дефолты — медиана вместо среднего, робастные методы поиска выбросов — и архитектурная защита от утечки.
- **Читаемость.** Подробные docstrings, типизация, отсутствие скрытых зависимостей. `help(pp.impute)` расскажет о функции всё.
- **Композируемость.** Любой шаг можно вызвать отдельно или собрать в `Pipeline` с автоматическим аудитом всего конвейера.

## 👥 Для кого

- **Студентам и тем, кто изучает ML** — увидеть, как выглядит корректная предобработка, и не наступать на классические грабли с утечкой.
- **ML-инженерам** — быстро собрать воспроизводимый и документированный препроцессинг, готовый к продакшену.
- **Аналитикам данных** — провести разведку (профилирование, корреляции, выбросы, графики) буквально в несколько строк.

## 🧰 Что умеет — по категориям

<details open>
<summary><b>📥 Загрузка</b></summary>

- `load_csv` — чтение CSV с автоматическим определением кодировки, разделителя и сжатия (`.gz`/`.zip`/`.bz2`/`.xz`). Никаких `UnicodeDecodeError` и угадывания разделителя вручную.
</details>

<details open>
<summary><b>🕳️ Пропуски</b></summary>

- `normalize_missing` — строки-заглушки (`"N/A"`, `"unknown"`, `"-999"`) → `NaN` с авто-приведением типов;
- `drop_sparse_columns`, `drop_rows` — удаление по доле пропусков или условию;
- `impute` / `apply_impute` — заполнение (median / mode / KNN / MICE) без утечки.
</details>

<details open>
<summary><b>🎯 Выбросы</b></summary>

- `detect_outliers` — 11 методов: одномерные (MAD, IQR, z-score, перцентили) и многомерные (Isolation Forest, LOF, One-Class SVM, Mahalanobis, GMM, ECOD, COPOD);
- `handle_outliers` — стратегии `clip` / `drop` / `mark_missing`;
- `apply_outliers` — перенос границ, посчитанных по train, на test.
</details>

<details open>
<summary><b>🔧 Преобразования</b></summary>

- `log_transform` — сжатие тяжёлых хвостов; `onehot_encode`, `parse_dates`, `to_numeric`;
- `scale` / `apply_scale` — standard / minmax / robust / maxabs, всегда без утечки.
</details>

<details open>
<summary><b>✂️ Разбиение и 🔍 анализ</b></summary>

- `split_dataset` — стратифицированный train/test с сохранением пропорций таргета;
- `profile_columns`, `count_variations`, `rank_correlations`, `compare_datasets` — разведка и сравнение «до/после».
</details>

<a id="install"></a>

## 🛠️ Установка

```bash
git clone https://github.com/danialmaleki06/preprocessing.git
cd preprocessing
pip install -e .
```

## 📖 Полный сценарий

Пример с разбиением и тремя статистическими шагами без единой утечки:

```python
import preprocessing as pp

df = pp.load_csv("data/titanic.csv").df
df = pp.normalize_missing(df).df
df = pp.drop_sparse_columns(df, threshold=0.5).df

sp = pp.split_dataset(df, by="survived", test_size=0.2)
train, test = sp.part_a, sp.part_b

imp = pp.impute(train, strategy="median")
train, test = imp.df, pp.apply_impute(test, imp)

det = pp.detect_outliers(train, method="iqr")
train = pp.handle_outliers(train, det, strategy="clip").df
test  = pp.apply_outliers(test, det)

sc = pp.scale(train, method="standard")
train, test = sc.df, pp.apply_scale(test, sc)
```

## 🧩 Пайплайн целиком

Те же шаги можно собрать в декларативную цепочку с автоматическим аудитом:

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
| `formatters` | Любой отчёт → читаемый текст |

Полный справочник по любой функции — в docstrings: `help(pp.impute)`.

## 🧪 Тесты

Качество кода подтверждено набором из 34 тестов, покрывающих загрузку, импутацию, выбросы, масштабирование и разбиение:

```bash
pip install -e ".[dev]"
pytest
`
