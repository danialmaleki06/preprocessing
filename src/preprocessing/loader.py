"""CSV loading with automatic encoding and separator detection."""
from __future__ import annotations

import bz2
import csv
import gzip
import io
import lzma
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

_SAMPLE_BYTES = 65536


@dataclass
class LoadReport:
    """Технический отчёт о процессе загрузки CSV-файла.

    Содержит только информацию о том, *как* был прочитан файл —
    какая кодировка, какой разделитель, какие технические замечания.
    Не содержит статистики данных, инференции типов и анализа пропусков:
    это задача следующих модулей пайплайна.
    """

    source_path: Path
    file_size_bytes: int
    compression: str | None

    encoding: str
    encoding_was_detected: bool

    separator: str
    separator_was_detected: bool

    bom_detected: bool

    n_rows: int
    n_columns: int
    column_names: list[str]

    duplicate_columns: list[str] = field(default_factory=list)
    load_time_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class LoadResult:
    """Результат загрузки: данные плюс технический отчёт.

    Атрибуты:
        df: прочитанный DataFrame.
        report: LoadReport с информацией о процессе чтения.
    """

    df: pd.DataFrame
    report: LoadReport


def _detect_compression(path: Path | str) -> str | None:
    """Определяет тип сжатия по расширению файла.

    Возвращает одну из строк: "gzip", "zip", "bz2", "xz" — или None,
    если расширение не распознано как сжатый формат.

    Сжатие определяется ТОЛЬКО по расширению, содержимое файла не читается.
    Сравнение регистронезависимое: ".GZ" и ".gz" дают одинаковый результат.
    """
    suffix = Path(path).suffix.lower()
    mapping = {
        ".gz": "gzip",
        ".zip": "zip",
        ".bz2": "bz2",
        ".xz": "xz",
    }
    return mapping.get(suffix)


def _detect_encoding(sample: bytes) -> tuple[str, bool]:
    """Определяет кодировку текстового сэмпла.

    Возвращает (encoding, bom_detected), где:
        encoding: имя кодировки в нижнем регистре
                  ("utf-8", "utf-8-sig", "utf-16", "utf-32",
                   "windows-1251" и т.п.).
        bom_detected: True, если кодировка определена через BOM в начале сэмпла.

    Бросает ValueError, если кодировку определить не удалось — в этом случае
    пользователь должен передать encoding= явно в load_csv.

    Алгоритм:
        1. Пустой вход → ValueError.
        2. BOM-проверки в порядке от длинного BOM к короткому
           (UTF-32 LE/BE → UTF-16 LE/BE → UTF-8 с BOM).
        3. Попытка декодировать как обычный UTF-8 (без BOM).
        4. charset-normalizer; если она вернула None — это бинарный файл,
           поднимаем ValueError. Иначе сверяемся, что предложенная кодировка
           действительно декодирует сэмпл, и возвращаем её.
        5. Если декодирование на шаге 4 всё-таки провалилось — ValueError.
    """
    if not sample:
        raise ValueError("Cannot detect encoding from empty sample")

    if sample.startswith(b"\xff\xfe\x00\x00"):
        return ("utf-32", True)
    if sample.startswith(b"\x00\x00\xfe\xff"):
        return ("utf-32", True)
    if sample.startswith(b"\xff\xfe"):
        return ("utf-16", True)
    if sample.startswith(b"\xfe\xff"):
        return ("utf-16", True)
    if sample.startswith(b"\xef\xbb\xbf"):
        return ("utf-8-sig", True)

    try:
        sample.decode("utf-8")
        return ("utf-8", False)
    except UnicodeDecodeError:
        pass

    best = from_bytes(sample).best()
    if best is None:
        raise ValueError(
            "Файл не похож на текстовый (charset-normalizer не нашёл "
            "подходящей кодировки). Передайте encoding= явно, если уверены, "
            "что это текст."
        )

    encoding = best.encoding.lower().replace("_", "-")

    try:
        sample.decode(encoding)
        return (encoding, False)
    except (UnicodeDecodeError, LookupError):
        raise ValueError(
            f"charset-normalizer предложил {encoding!r}, "
            f"но декодирование не удалось. Передайте encoding= явно."
        )


def _detect_separator(decoded_sample: str) -> str:
    """Определяет разделитель CSV по декодированному сэмплу.

    Возвращает один из четырёх кандидатов: ',' ';' '\\t' '|'.

    Алгоритм:
        1. Разбиваем сэмпл на непустые строки.
        2. Если непустых строк нет → ValueError.
        3. Берём первые ≤50 непустых строк (этого достаточно для статистики).
        4. Для каждого кандидата подсчитываем число вхождений в каждой строке.
        5. Считаем "score" кандидата:
              score = (доля строк с самой частой шириной)
                      × (сама эта ширина в столбцах).
           Высокий score означает И стабильную ширину (=правильный разделитель),
           И ненулевое число столбцов (=файл реально делится на поля).
        6. Возвращаем кандидата с максимальным score.
        7. Если все кандидаты дали score=0 (ни одной разделительной метки
           не найдено), считаем файл одноколоночным и поднимаем ValueError —
           пользователь должен передать sep= явно.
    """
    candidates = [",", ";", "\t", "|"]
    max_lines = 50

    lines = [line for line in decoded_sample.splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            "Сэмпл не содержит непустых строк — невозможно определить разделитель."
        )

    lines = lines[:max_lines]
    n_lines = len(lines)

    best_sep: str | None = None
    best_score: float = 0.0

    for sep in candidates:
        counts = [line.count(sep) for line in lines]
        most_common_count, most_common_freq = Counter(counts).most_common(1)[0]
        if most_common_count == 0:
            continue

        consistency = most_common_freq / n_lines
        score = consistency * most_common_count

        if score > best_score:
            best_score = score
            best_sep = sep

    if best_sep is None:
        raise ValueError(
            "Не удалось определить разделитель: ни один из кандидатов "
            "(',', ';', '\\t', '|') не встречается в сэмпле. Возможно, "
            "файл одноколоночный — передайте sep= явно."
        )

    return best_sep


def _read_sample_bytes(
    path: Path,
    compression: str | None,
    max_bytes: int = _SAMPLE_BYTES,
) -> bytes:
    """Читает первые `max_bytes` байт из файла, корректно обрабатывая сжатие.

    Для несжатых файлов открывает файл в бинарном режиме.
    Для сжатых (.gz, .bz2, .xz) возвращает ДЕКОМПРЕССИРОВАННЫЕ байты —
    то есть содержимое CSV, а не сырой архивный поток.
    Для .zip с несколькими файлами поднимает ValueError.
    """
    if compression is None:
        with open(path, "rb") as f:
            return f.read(max_bytes)

    if compression == "gzip":
        with gzip.open(path, "rb") as f:
            return f.read(max_bytes)

    if compression == "bz2":
        with bz2.open(path, "rb") as f:
            return f.read(max_bytes)

    if compression == "xz":
        with lzma.open(path, "rb") as f:
            return f.read(max_bytes)

    if compression == "zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if len(names) == 0:
                raise ValueError(f"ZIP-архив пустой: {path}")
            if len(names) > 1:
                raise ValueError(
                    f"ZIP-архив содержит несколько файлов ({names}); "
                    f"распакуйте нужный вручную."
                )
            with zf.open(names[0]) as f:
                return f.read(max_bytes)

    raise ValueError(f"Неизвестный тип сжатия: {compression!r}")


def _detect_duplicate_columns(
    decoded_sample: str,
    separator: str,
    header: int | None,
) -> list[str]:
    """Находит дубликаты в строке заголовка ОРИГИНАЛЬНОГО файла.

    pandas автоматически переименовывает дубликаты в `name`, `name.1`, ...,
    поэтому проверка по `df.columns.duplicated()` их уже не найдёт.
    Здесь парсим первую строку файла руками через csv.reader (он корректно
    обрабатывает кавычки) и считаем имена, встретившиеся больше одного раза.

    Возвращает отсортированный список повторяющихся имён.
    Пустой список — если header=None или строки заголовка нет.
    """
    if header is None or header < 0:
        return []

    lines = [line for line in decoded_sample.splitlines() if line.strip()]
    if header >= len(lines):
        return []

    header_line = lines[header]
    try:
        reader = csv.reader(io.StringIO(header_line), delimiter=separator)
        original_columns = next(reader, [])
    except csv.Error:
        return []

    counts = Counter(original_columns)
    return sorted([name for name, count in counts.items() if count > 1])


def _collect_warnings(df: pd.DataFrame, decoded_sample: str) -> list[str]:
    """Собирает технические замечания о прочитанном файле.

    Сейчас проверяются:
    - имена колонок с пробельными символами по краям;
    - пустые строки в сэмпле (pandas их пропускает, но пользователю
      полезно знать об этом).
    """
    warnings_list: list[str] = []

    for col in df.columns:
        if isinstance(col, str) and col != col.strip():
            warnings_list.append(
                f"Имя колонки {col!r} содержит пробельные символы по краям"
            )

    sample_lines = decoded_sample.splitlines()
    non_empty = [line for line in sample_lines if line.strip()]
    if 0 < len(non_empty) < len(sample_lines):
        empty_count = len(sample_lines) - len(non_empty)
        warnings_list.append(
            f"В сэмпле найдено {empty_count} пустых строк — pandas их пропускает"
        )

    return warnings_list


def load_csv(
    path: str | Path,
    *,
    encoding: str | None = None,
    separator: str | None = None,
    header: int | None = 0,
) -> LoadResult:
    """Загружает CSV в DataFrame с автоопределением кодировки и разделителя.

    Параметры (все после звёздочки — keyword-only):
        path: путь к файлу. Поддерживаются `.csv`, `.csv.gz`, `.csv.zip`,
              `.csv.bz2`, `.csv.xz`. URL не поддерживаются.
        encoding: имя кодировки. None = автоопределение.
        separator: разделитель. None = автоопределение.
        header: номер строки заголовка (по умолчанию 0). None = заголовка нет,
                pandas сгенерирует имена колонок 0, 1, 2, ...

    Возвращает:
        LoadResult с полями:
            .df — pandas DataFrame.
            .report — LoadReport с информацией о процессе чтения.

    Бросает:
        FileNotFoundError: файла по указанному пути нет.
        ValueError: файл пустой / бинарный / без разделителя / 0 колонок,
                    либо .zip-архив содержит несколько файлов.
        pd.errors.ParserError: pandas не смог распарсить файл — пробрасывается
                               без изменений.

    Алгоритм:
        1. Проверка существования файла и ненулевого размера.
        2. Определение типа сжатия по расширению (`_detect_compression`).
        3. Чтение первых 64 КБ ДЕКОМПРЕССИРОВАННЫХ байт (`_read_sample_bytes`).
        4. Определение кодировки из сэмпла (`_detect_encoding`),
           либо использование заданной пользователем.
        5. Декодирование сэмпла в текст (errors='replace' для обработки
           возможной обрезки многобайтового символа на границе сэмпла).
        6. Определение разделителя (`_detect_separator`),
           либо использование заданного.
        7. Чтение полного DataFrame через `pd.read_csv` с найденными
           параметрами.
        8. Валидация (≥1 колонка), сборка LoadReport, возврат LoadResult.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Путь не указывает на файл: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Файл пустой: {path}")

    start_time = time.monotonic()

    compression = _detect_compression(path)

    sample_bytes = _read_sample_bytes(path, compression)
    if not sample_bytes:
        raise ValueError(f"После распаковки файл оказался пустым: {path}")

    encoding_was_detected = encoding is None
    if encoding is None:
        encoding, bom_detected = _detect_encoding(sample_bytes)
    else:
        bom_detected = (
            sample_bytes.startswith(b"\xef\xbb\xbf")
            or sample_bytes.startswith(b"\xff\xfe")
            or sample_bytes.startswith(b"\xfe\xff")
        )

    decoded_sample = sample_bytes.decode(encoding, errors="replace")

    separator_was_detected = separator is None
    if separator is None:
        separator = _detect_separator(decoded_sample)

    df = pd.read_csv(
        path,
        encoding=encoding,
        sep=separator,
        compression=compression,
        header=header,
    )

    if len(df.columns) == 0:
        raise ValueError(f"После парсинга получено 0 колонок: {path}")

    duplicate_columns = _detect_duplicate_columns(decoded_sample, separator, header)
    warnings_list = _collect_warnings(df, decoded_sample)

    load_time = time.monotonic() - start_time

    report = LoadReport(
        source_path=path,
        file_size_bytes=file_size,
        compression=compression,
        encoding=encoding,
        encoding_was_detected=encoding_was_detected,
        separator=separator,
        separator_was_detected=separator_was_detected,
        bom_detected=bom_detected,
        n_rows=len(df),
        n_columns=len(df.columns),
        column_names=df.columns.tolist(),
        duplicate_columns=duplicate_columns,
        load_time_seconds=load_time,
        warnings=warnings_list,
    )

    return LoadResult(df=df, report=report)


def format_load_summary(report: LoadReport) -> str:
    """Возвращает человекочитаемый текстовый отчёт о загрузке.

    Утилита для удобного просмотра LoadReport в консоли:
    выводит путь, размер, кодировку, разделитель, форму DataFrame,
    дубли колонок, время загрузки, замечания.
    """
    sep_display = {
        ",": "запятая (,)",
        ";": "точка с запятой (;)",
        "\t": "табуляция (\\t)",
        "|": "вертикальная черта (|)",
    }.get(report.separator, repr(report.separator))

    encoding_origin = (
        "автоопределена" if report.encoding_was_detected else "задана пользователем"
    )
    separator_origin = (
        "автоопределён" if report.separator_was_detected else "задан пользователем"
    )

    lines = [
        "=" * 60,
        "ОТЧЁТ О ЗАГРУЗКЕ CSV",
        "=" * 60,
        "",
        "Файл:",
        f"  путь:           {report.source_path}",
        f"  размер:         {report.file_size_bytes:,} байт",
        f"  сжатие:         {report.compression or 'нет'}",
        "",
        "Параметры чтения:",
        f"  кодировка:      {report.encoding} ({encoding_origin})",
        f"  BOM:            {'обнаружен' if report.bom_detected else 'нет'}",
        f"  разделитель:    {sep_display} ({separator_origin})",
        "",
        "Структура:",
        f"  строк:          {report.n_rows:,}",
        f"  колонок:        {report.n_columns}",
    ]

    if report.column_names:
        preview = ", ".join(str(c) for c in report.column_names[:5])
        if len(report.column_names) > 5:
            preview += f", ... (+{len(report.column_names) - 5})"
        lines.append(f"  имена колонок:  {preview}")

    if report.duplicate_columns:
        lines.append(
            f"  дубли колонок:  {', '.join(report.duplicate_columns)}"
        )

    lines.extend(
        [
            "",
            f"Время загрузки:    {report.load_time_seconds:.3f} сек",
        ]
    )

    if report.warnings:
        lines.append("")
        lines.append("Замечания:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    lines.append("=" * 60)
    return "\n".join(lines)
