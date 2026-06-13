"""Scatter-графики для визуального поиска выбросов."""

from __future__ import annotations

from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler


def _resolve_columns(df: pd.DataFrame, columns: list[str] | str) -> list[str]:
    from preprocessing._typing_utils import warn_numeric_objects

    numeric = df.select_dtypes(include="number").columns.tolist()
    if columns == "all":
        skipped_object = [c for c in df.columns if c not in numeric]
        if skipped_object:
            warn_numeric_objects(df, skipped_object, "scatter plot")
        return numeric
    valid = []
    skipped_non_numeric = []
    for col in columns:
        if col not in df.columns:
            print(f"  [!] Столбец '{col}' не найден — пропущен.")
        elif col not in numeric:
            print(f"  [!] Столбец '{col}' не числовой — пропущен.")
            skipped_non_numeric.append(col)
        else:
            valid.append(col)
    if skipped_non_numeric:
        warn_numeric_objects(df, skipped_non_numeric, "scatter plot")
    return valid


def _pairs(cols: list[str]) -> list[tuple[str, str]]:
    return list(combinations(cols, 2))


def _kde_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
    contamination: float,
    figsize: tuple[int, int],
    title: str | None,
) -> None:
    data = df[[x, y]].dropna().astype(float).reset_index(drop=True)
    if len(data) < 10:
        print(f"  [!] {x} vs {y}: недостаточно данных, пропущено.")
        return

    kde = gaussian_kde(data.values.T)
    density = kde(data.values.T)
    threshold = np.percentile(density, contamination * 100)
    is_outlier = density < threshold

    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(data=data, x=x, y=y, ax=ax, levels=6,
                color="steelblue", linewidths=1.2, alpha=0.7)
    sns.kdeplot(data=data, x=x, y=y, ax=ax, levels=6,
                fill=True, cmap="Blues", alpha=0.25)
    ax.scatter(data[x][~is_outlier], data[y][~is_outlier],
               c="#4878cf", alpha=0.45, s=25,
               label=f"Норма: {(~is_outlier).sum()}")
    ax.scatter(data[x][is_outlier], data[y][is_outlier],
               c="#d62728", alpha=0.85, s=60, marker="X", zorder=5,
               label=f"Выброс: {is_outlier.sum()}  (нижние {contamination * 100:.0f}% плотности)")
    ax.set_title(title or f"KDE  [{x}  vs  {y}]", fontsize=13, pad=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_kde_scatter(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    contamination: float = 0.05,
    figsize: tuple[int, int] = (9, 7),
    title: str | None = None,
) -> None:
    """KDE-контуры плотности. Для каждой пары из указанных столбцов.

    columns       -- список столбцов или ``"all"`` для всех числовых.
    contamination -- доля точек считаемых выбросами (0.05 = нижние 5% плотности).

    Пример::

        plot_kde_scatter(df, "all")
        plot_kde_scatter(df, ["age", "fare"])
        plot_kde_scatter(df, ["age", "fare", "sibsp"], contamination=0.03)
    """
    cols = _resolve_columns(df, columns)
    pairs = _pairs(cols)
    if not pairs:
        print("Нужно минимум 2 числовых столбца.")
        return
    for i, (x, y) in enumerate(pairs, 1):
        print(f"  KDE  {i}/{len(pairs)}:  {x}  vs  {y}")
        _kde_pair(df, x, y, contamination, figsize, title)


def _dbscan_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
    eps: float,
    min_samples: int,
    figsize: tuple[int, int],
    title: str | None,
) -> None:
    data = df[[x, y]].dropna().astype(float).reset_index(drop=True)
    if len(data) < 10:
        print(f"  [!] {x} vs {y}: недостаточно данных, пропущено.")
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[[x, y]])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", max(n_clusters, 1))

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        pts = data[mask]
        if cluster_id == -1:
            ax.scatter(pts[x], pts[y], c="#d62728", marker="X", s=70,
                       alpha=0.85, zorder=5, label=f"Выброс (шум): {n_noise}")
            continue
        color = palette[cluster_id % len(palette)]
        ax.scatter(pts[x], pts[y], color=color, alpha=0.5, s=30,
                   label=f"Кластер {cluster_id + 1}: {mask.sum()}")
        if mask.sum() >= 3:
            try:
                hull = ConvexHull(pts[[x, y]].values)
                hull_pts = pts[[x, y]].values[hull.vertices]
                ax.add_patch(Polygon(hull_pts, closed=True,
                                     facecolor=(*color, 0.10),
                                     edgecolor=color, linewidth=1.8, linestyle="--"))
            except Exception:
                pass

    ax.set_title(
        title or (f"DBSCAN  (eps={eps}, min_samples={min_samples})"
                  f"  —  {n_clusters} кластер(ов), {n_noise} выбросов"
                  f"  [{x}  vs  {y}]"),
        fontsize=12, pad=10,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_dbscan_scatter(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    figsize: tuple[int, int] = (9, 7),
    title: str | None = None,
) -> None:
    """DBSCAN: кластеры обведены Convex Hull, шум = выбросы. Для каждой пары столбцов.

    columns     -- список столбцов или ``"all"`` для всех числовых.
    eps         -- расстояние между точками одного кластера (в σ после нормализации).
    min_samples -- минимум точек для образования кластера.

    Пример::

        plot_dbscan_scatter(df, "all")
        plot_dbscan_scatter(df, ["age", "fare"], eps=0.4, min_samples=8)
    """
    cols = _resolve_columns(df, columns)
    pairs = _pairs(cols)
    if not pairs:
        print("Нужно минимум 2 числовых столбца.")
        return
    for i, (x, y) in enumerate(pairs, 1):
        print(f"  DBSCAN  {i}/{len(pairs)}:  {x}  vs  {y}")
        _dbscan_pair(df, x, y, eps, min_samples, figsize, title)


def _elliptic_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
    contamination: float,
    figsize: tuple[int, int],
    title: str | None,
) -> None:
    data = df[[x, y]].dropna().astype(float).reset_index(drop=True)
    if len(data) < 10:
        print(f"  [!] {x} vs {y}: недостаточно данных, пропущено.")
        return

    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        preds = ee.fit_predict(data[[x, y]].values)
    except ValueError as exc:
        print(f"  [!] {x} vs {y}: Elliptic Envelope не применим — {exc}")
        return

    is_outlier = preds == -1

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[x][~is_outlier], data[y][~is_outlier],
               c="#4878cf", alpha=0.5, s=25,
               label=f"Норма: {(~is_outlier).sum()}")
    ax.scatter(data[x][is_outlier], data[y][is_outlier],
               c="#d62728", alpha=0.85, s=60, marker="X", zorder=5,
               label=f"Выброс: {is_outlier.sum()}  (contamination={contamination})")

    x_grid = np.linspace(data[x].min(), data[x].max(), 300)
    y_grid = np.linspace(data[y].min(), data[y].max(), 300)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = ee.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-999, 0], colors=["#ffd700"], alpha=0.10)
    ax.contour(xx, yy, zz, levels=[0], colors=["#e07b00"], linewidths=2.0, linestyles="--")

    boundary_patch = mpatches.Patch(edgecolor="#e07b00", facecolor="#ffd700",
                                    alpha=0.4, label="Граница эллипса")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [boundary_patch], fontsize=9)
    ax.set_title(
        title or f"Elliptic Envelope  (contamination={contamination})  [{x}  vs  {y}]",
        fontsize=12, pad=10,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_elliptic_scatter(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    contamination: float = 0.1,
    figsize: tuple[int, int] = (9, 7),
    title: str | None = None,
) -> None:
    """Elliptic Envelope: эллипс вокруг основного облака, снаружи — выбросы. Для каждой пары.

    columns       -- список столбцов или ``"all"`` для всех числовых.
    contamination -- ожидаемая доля выбросов (0.1 = 10%).

    Пример::

        plot_elliptic_scatter(df, "all")
        plot_elliptic_scatter(df, ["age", "fare"], contamination=0.05)
    """
    cols = _resolve_columns(df, columns)
    pairs = _pairs(cols)
    if not pairs:
        print("Нужно минимум 2 числовых столбца.")
        return
    for i, (x, y) in enumerate(pairs, 1):
        print(f"  Elliptic  {i}/{len(pairs)}:  {x}  vs  {y}")
        _elliptic_pair(df, x, y, contamination, figsize, title)


def plot_strip(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    figsize: tuple[int, int] = (5, 8),
    alpha: float = 0.35,
    point_size: int = 4,
    color: str = "#4878cf",
) -> None:
    """Strip plot для каждой указанной колонки отдельным графиком.

    Каждая точка = одна строка датасета. Горизонтальный джиттер добавляется
    автоматически, чтобы перекрывающиеся точки были видны.
    Под заголовком — базовая статистика: n точек, NaN, min / median / max.

    columns     -- список числовых столбцов или ``"all"`` для всех числовых. Для каждого рисуется свой график.
    figsize     -- размер каждого графика.
    alpha       -- прозрачность точек (0..1). Меньше — лучше видны кластеры.
    point_size  -- размер точки в пунктах.
    color       -- цвет точек (hex или имя matplotlib-цвета).

    Пример::

        plot_strip(df, ["Amount", "V1", "V3"])
        plot_strip(df, ["fare", "age"], figsize=(4, 7), alpha=0.2)
    """
    valid = _resolve_columns(df, columns)
    if not valid:
        print("Нет подходящих числовых столбцов.")
        return

    for col in valid:
        data = df[col].dropna()
        n_total = len(df[col])
        n_nan   = int(df[col].isna().sum())
        n_pts   = len(data)

        fig, ax = plt.subplots(figsize=figsize)

        sns.stripplot(
            y=data,
            ax=ax,
            size=point_size,
            alpha=alpha,
            color=color,
            jitter=True,
            native_scale=False,
        )

        stats_line = (
            f"n={n_pts}"
            + (f"  NaN={n_nan}" if n_nan else "")
            + f"  min={data.min():.4g}"
            + f"  median={data.median():.4g}"
            + f"  max={data.max():.4g}"
        )
        ax.set_title(f"{col}", fontsize=13, pad=6)
        ax.set_xlabel(stats_line, fontsize=8, color="#555")
        ax.set_xticks([])
        ax.set_ylabel(col, fontsize=10)

        plt.tight_layout()
        plt.show()


def plot_outlier_scatter(
    df: pd.DataFrame,
    columns: list[str] | str = "all",
    *,
    kde_contamination: float = 0.05,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    elliptic_contamination: float = 0.1,
    figsize: tuple[int, int] = (9, 7),
) -> None:
    """Все три метода для каждой пары из указанных столбцов.

    columns -- список столбцов или ``"all"`` для всех числовых.

    Пример::

        plot_outlier_scatter(df, "all")
        plot_outlier_scatter(df, ["age", "fare", "sibsp"])
        plot_outlier_scatter(df, ["age", "fare"], dbscan_eps=0.3)
    """
    cols = _resolve_columns(df, columns)
    pairs = _pairs(cols)
    if not pairs:
        print("Нужно минимум 2 числовых столбца.")
        return

    for i, (x, y) in enumerate(pairs, 1):
        print(f"\n-- Пара {i}/{len(pairs)}: {x} vs {y} --")
        print("  -- 1/3  KDE")
        _kde_pair(df, x, y, kde_contamination, figsize, None)
        print("  -- 2/3  DBSCAN")
        _dbscan_pair(df, x, y, dbscan_eps, dbscan_min_samples, figsize, None)
        print("  -- 3/3  Elliptic Envelope")
        _elliptic_pair(df, x, y, elliptic_contamination, figsize, None)
