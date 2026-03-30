"""
Helper functions for geospatial ML training notebook.
Contains plotting utilities and common operations to keep the notebook clean.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Color palettes and constants
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get("UCB_PROJECT_ROOT", Path(__file__).resolve().parent)
).resolve()
DATA_ROOT = Path(os.environ.get("UCB_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
VECTOR_CRS = "EPSG:4326"
ANALYSIS_CRS = "EPSG:32638"

CONTINUOUS_CMAP = "viridis"
DIVERGING_CMAP = "RdYlBu_r"
CATEGORICAL_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]
ANOMALY_COLORS = {"normal": "#34495E", "anomaly": "#D35400"}
ALTERATION_COLORS = {
    "Background": "#f0f0f0",
    "Advanced Argillic": "#d73027",
    "Phyllic": "#fc8d59",
    "Gossan": "#fee090",
    "Argillic": "#e0f3f8",
    "Propylitic": "#91bfdb",
    "Laterite": "#4575b4",
}

DEFAULT_DATA_CONFIG = {
    # Rasters
    "continuous_raster_path": DATA_ROOT / "raster/spectral/idx_clay_hydroxyls.tif",
    "categorical_raster_path": None,  # GeoTIFF with class labels
    # Vector data
    "vector_path": DATA_ROOT / "vector/lithology.geojson",
    "geochem_points_path": DATA_ROOT / "vector/geochem.geojson",
    # Raster data
    "spectral_indices_dir": DATA_ROOT / "raster/spectral",
    "geophysics_dir": DATA_ROOT / "raster/geophys",
    # Prospectivity mapping
    "prospectivity_feature_rasters": [],  # List of raster paths (GeoTIFF)
    "prospectivity_training_points_path": None,  # GeoJSON with known deposits
}


# =============================================================================
# Data summary functions
# =============================================================================


def summarize_geochem(gdf, feature_cols=None, max_cols=10):
    """
    Print a summary of geochem data structure and contents.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geochemical data
    feature_cols : list
        Feature columns to summarize (auto-detected if None)
    max_cols : int
        Maximum columns to show in detailed stats
    """
    print("=" * 60)
    print("GEOCHEMICAL DATA SUMMARY")
    print("=" * 60)

    # Basic info
    print(f"\nRecords: {len(gdf)}")
    print(f"Total columns: {len(gdf.columns)}")

    # Detect feature columns if not provided
    if feature_cols is None:
        exclude = [
            "geometry",
            "X",
            "Y",
            "x",
            "y",
            "coord_x",
            "coord_y",
            "OBJECTID",
            "FID",
            "id",
        ]
        feature_cols = [
            c
            for c in gdf.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    print(f"Numeric feature columns: {len(feature_cols)}")

    # Coordinate info
    if len(gdf) > 0 and hasattr(gdf, "geometry"):
        xs = gdf.geometry.x
        ys = gdf.geometry.y
        print(f"\nSpatial extent:")
        print(f"  X: {xs.min():.2f} to {xs.max():.2f}")
        print(f"  Y: {ys.min():.2f} to {ys.max():.2f}")

    # Element groups
    majors = [c for c in feature_cols if "percent" in c.lower() or c.endswith("_pct")]
    traces_ppm = [c for c in feature_cols if "ppm" in c.lower()]
    traces_ppb = [c for c in feature_cols if "ppb" in c.lower()]

    print(f"\nElement breakdown:")
    print(f"  Major oxides (percent): {len(majors)}")
    print(f"  Trace elements (ppm): {len(traces_ppm)}")
    print(f"  Trace elements (ppb): {len(traces_ppb)}")

    # Missing data
    missing_counts = gdf[feature_cols].isnull().sum()
    cols_with_missing = (missing_counts > 0).sum()
    total_missing = missing_counts.sum()
    total_cells = len(gdf) * len(feature_cols)

    print(f"\nMissing data:")
    print(f"  Columns with missing: {cols_with_missing} / {len(feature_cols)}")
    print(
        f"  Total missing cells: {total_missing} / {total_cells} ({100*total_missing/total_cells:.1f}%)"
    )

    # Quick stats for subset of columns
    print(f"\nSample statistics (first {min(max_cols, len(feature_cols))} columns):")
    print("-" * 60)
    stats_cols = feature_cols[:max_cols]
    stats_df = gdf[stats_cols].describe().T[["mean", "std", "min", "max"]]
    stats_df.columns = ["Mean", "Std", "Min", "Max"]
    print(stats_df.to_string())
    print("=" * 60)


def summarize_imputation(original_df, imputed_array, feature_cols):
    """
    Print summary of imputation results.

    Parameters
    ----------
    original_df : pd.DataFrame
        DataFrame with missing values
    imputed_array : np.ndarray
        Array after imputation
    feature_cols : list
        Column names corresponding to imputed_array
    """
    print("=" * 60)
    print("IMPUTATION SUMMARY")
    print("=" * 60)

    total_missing = original_df[feature_cols].isnull().sum().sum()
    total_cells = len(original_df) * len(feature_cols)

    print(f"\nTotal values imputed: {total_missing}")
    print(f"Percentage of data imputed: {100*total_missing/total_cells:.2f}%")

    # Per-column summary
    print(f"\nPer-column breakdown:")
    print("-" * 60)
    print(f"{'Column':<25} {'N Missing':>10} {'% Missing':>10} {'Imputed Value':>15}")
    print("-" * 60)

    imputed_df = pd.DataFrame(imputed_array, columns=feature_cols)
    for i, col in enumerate(feature_cols):
        n_missing = original_df[col].isnull().sum()
        if n_missing > 0:
            pct_missing = 100 * n_missing / len(original_df)
            # The imputed value (mean for mean imputation)
            imputed_val = imputed_df[col].iloc[original_df[col].isnull().values].mean()
            print(
                f"{col:<25} {n_missing:>10} {pct_missing:>9.1f}% {imputed_val:>15.2f}"
            )

    print("=" * 60)


# =============================================================================
# Data visualization functions
# =============================================================================


def plot_raster(
    data,
    title="Raster Data",
    cmap=CONTINUOUS_CMAP,
    vmin=None,
    vmax=None,
    ax=None,
    colorbar=True,
    extent=None,
    robust_stretch=True,
    origin="upper",
):
    """
    Plot a 2D raster array with optional colorbar.

    Parameters
    ----------
    data : np.ndarray
        2D array to plot
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Value range for colormap
    ax : matplotlib.axes.Axes
        Axes to plot on (creates new figure if None)
    colorbar : bool
        Whether to add colorbar
    extent : tuple
        (xmin, xmax, ymin, ymax) for georeferenced display
    robust_stretch : bool
        Use 2nd-98th percentile stretch

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Apply robust stretch if requested
    if robust_stretch and vmin is None and vmax is None:
        valid_data = data[~np.isnan(data)] if np.any(np.isnan(data)) else data.flatten()
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin=origin)
    ax.set_title(title)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)

    return ax


def plot_vector(
    gdf,
    column=None,
    title="Vector Data",
    cmap=CONTINUOUS_CMAP,
    categorical=False,
    ax=None,
    legend=True,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.7,
    markersize=30,
    categorical_cmap=None,
):
    """
    Plot a GeoDataFrame with optional column coloring.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector data to plot
    column : str
        Column to use for coloring
    title : str
        Plot title
    cmap : str
        Colormap for continuous data
    categorical : bool
        Whether column is categorical
    ax : matplotlib.axes.Axes
        Axes to plot on
    legend : bool
        Whether to show legend
    edgecolor : str
        Edge color for polygons
    linewidth : float
        Line width
    alpha : float
        Transparency
    markersize : int
        Size for point geometries

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    geom_type = gdf.geometry.iloc[0].geom_type

    if column is not None:
        if categorical:
            cmap_obj = categorical_cmap
            if cmap_obj is None:
                cmap_obj = mcolors.ListedColormap(
                    CATEGORICAL_COLORS[: gdf[column].nunique()]
                )
            gdf.plot(
                column=column,
                ax=ax,
                legend=legend,
                cmap=cmap_obj,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                markersize=markersize if "Point" in geom_type else None,
            )
        else:
            gdf.plot(
                column=column,
                ax=ax,
                legend=legend,
                cmap=cmap,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                markersize=markersize if "Point" in geom_type else None,
            )
    else:
        gdf.plot(
            ax=ax,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            color="steelblue",
            markersize=markersize if "Point" in geom_type else None,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def plot_geometry_types(gdf_point, gdf_line, gdf_poly, figsize=(15, 5)):
    """
    Plot point, line, and polygon geometries side by side.

    Parameters
    ----------
    gdf_point : geopandas.GeoDataFrame
        Point geometries
    gdf_line : geopandas.GeoDataFrame
        Line geometries
    gdf_poly : geopandas.GeoDataFrame
        Polygon geometries
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_vector(
        gdf_point, ax=axes[0], title="Point Geometries", markersize=50, legend=False
    )
    plot_vector(
        gdf_line, ax=axes[1], title="Line Geometries", linewidth=2, legend=False
    )
    plot_vector(
        gdf_poly, ax=axes[2], title="Polygon Geometries", alpha=0.6, legend=False
    )

    plt.tight_layout()
    return fig, axes


def plot_raster_vs_vector(raster_data, gdf, extent=None, figsize=(14, 6)):
    """
    Plot raster and vector data side by side for comparison.

    Parameters
    ----------
    raster_data : np.ndarray
        2D raster array
    gdf : geopandas.GeoDataFrame
        Vector data
    extent : tuple
        Raster extent (xmin, xmax, ymin, ymax)
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_raster(raster_data, ax=axes[0], title="Raster (Gridded)", extent=extent)
    plot_vector(gdf, ax=axes[1], title="Vector (Discrete Features)")

    plt.tight_layout()
    return fig, axes


def plot_continuous_vs_categorical(
    continuous_data, categorical_data, extent=None, figsize=(14, 6)
):
    """
    Plot continuous and categorical rasters side by side.

    Parameters
    ----------
    continuous_data : np.ndarray
        Continuous value raster
    categorical_data : np.ndarray
        Categorical/classified raster
    extent : tuple
        Raster extent
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_raster(
        continuous_data,
        ax=axes[0],
        title="Continuous Data",
        cmap=CONTINUOUS_CMAP,
        extent=extent,
    )

    # For categorical, use discrete colormap
    n_classes = int(np.nanmax(categorical_data)) + 1
    cmap_cat = mcolors.ListedColormap(CATEGORICAL_COLORS[:n_classes])

    im = axes[1].imshow(categorical_data, cmap=cmap_cat, extent=extent, origin="upper")
    axes[1].set_title("Categorical Data")

    # Add discrete colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=range(n_classes))
    cbar.set_label("Class")

    plt.tight_layout()
    return fig, axes


# =============================================================================
# EDA and transformation plotting
# =============================================================================


def plot_distribution(
    data,
    title="Distribution",
    ax=None,
    bins=50,
    show_stats=True,
    color="steelblue",
    ncols=5,
    max_plots=None,
    figsize=None,
    titles=None,
):
    """
    Plot histogram with optional statistics overlay.

    If data is a DataFrame, dict, or 2D array-like and ax is None, this will
    render multiple distributions in a subplot grid.

    Parameters
    ----------
    data : array-like, pandas.DataFrame, dict, or 2D array-like
        Data to plot
    title : str
        Plot title (used as suptitle for subplot grids)
    ax : matplotlib.axes.Axes
        Axes to plot on
    bins : int
        Number of histogram bins
    show_stats : bool
        Whether to show mean/median/std
    color : str
        Histogram color
    ncols : int
        Number of columns for subplot grids
    max_plots : int or None
        Max number of distributions to plot in grids
    figsize : tuple or None
        Figure size for grids
    titles : list or None
        Titles for 2D array-like data

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        is_df = isinstance(data, pd.DataFrame)
        is_dict = isinstance(data, dict)
        is_2d = isinstance(data, (list, tuple, np.ndarray)) and np.ndim(data) == 2
        if is_df or is_dict or is_2d:
            if is_df:
                series_dict = {col: data[col].values for col in data.columns}
            elif is_dict:
                series_dict = data
            else:
                if titles is None:
                    titles = [f"Var {i + 1}" for i in range(np.shape(data)[1])]
                series_dict = {t: np.array(data)[:, i] for i, t in enumerate(titles)}

            items = list(series_dict.items())
            if max_plots is not None:
                items = items[:max_plots]

            n = len(items)
            ncols = min(max(ncols, 1), n)
            nrows = int(np.ceil(n / ncols))
            if figsize is None:
                figsize = (3 * ncols, 2.5 * nrows)

            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, constrained_layout=True
            )
            axes = np.array(axes).flatten()
            for ax_i, (label, values) in zip(axes, items):
                plot_distribution(
                    values,
                    title=str(label),
                    ax=ax_i,
                    bins=bins,
                    show_stats=show_stats,
                    color=color,
                )

            for ax_i in axes[len(items) :]:
                ax_i.set_visible(False)

            if title:
                fig.suptitle(title)
            return axes[0]

        fig, ax = plt.subplots(figsize=(8, 5))

    data_clean = np.array(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    ax.hist(data_clean, bins=bins, color=color, alpha=0.7, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    if show_stats:
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)
        std_val = np.std(data_clean)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median: {median_val:.2f}",
        )

        stats_text = f"Std: {std_val:.2f}\nMin: {np.min(data_clean):.2f}\nMax: {np.max(data_clean):.2f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.legend(loc="upper left")

    return ax


def plot_transformation_comparison(
    original, transformed, transform_name="Transformed", figsize=(14, 5)
):
    """
    Plot original vs transformed data distributions.

    Parameters
    ----------
    original : array-like
        Original data
    transformed : array-like
        Transformed data
    transform_name : str
        Name of transformation applied
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_distribution(original, title="Original Distribution", ax=axes[0])
    plot_distribution(transformed, title=f"{transform_name} Distribution", ax=axes[1])

    plt.tight_layout()
    return fig, axes


def plot_correlation_matrix(
    df, title="Correlation Matrix", figsize=(10, 8), annot=True, cmap="coolwarm"
):
    """
    Plot correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells
    cmap : str
        Colormap

    Returns
    -------
    fig, ax : tuple
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)

    return fig, ax


# =============================================================================
# Missing data visualization
# =============================================================================


def plot_missing_data_pattern(df, figsize=(12, 6)):
    """
    Visualize missing data patterns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Missing percentage by column (preserve original column order)
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        axes[0].bar(
            range(len(missing_pct)),
            missing_pct.values,
            color="coral",
            edgecolor="white",
        )
        axes[0].set_ylabel("Missing %")
        axes[0].set_title("Missing % by Column")
        axes[0].axhline(y=50, color="red", linestyle="--", alpha=0.5)

        # Handle x-axis labels based on number of columns
        n_cols = len(missing_pct)
        if n_cols <= 15:
            # Show all labels
            axes[0].set_xticks(range(n_cols))
            axes[0].set_xticklabels(
                missing_pct.index, rotation=45, ha="right", fontsize=8
            )
        elif n_cols <= 30:
            # Show every other label
            step = 2
            axes[0].set_xticks(range(0, n_cols, step))
            axes[0].set_xticklabels(
                missing_pct.index[::step], rotation=45, ha="right", fontsize=7
            )
        else:
            # Show every nth label to keep ~10-15 labels visible
            step = max(n_cols // 12, 3)
            axes[0].set_xticks(range(0, n_cols, step))
            axes[0].set_xticklabels(
                missing_pct.index[::step], rotation=45, ha="right", fontsize=7
            )
        axes[0].set_xlabel(f"Columns ({n_cols} with missing data)")
    else:
        axes[0].text(
            0.5,
            0.5,
            "No missing data",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=14,
        )
        axes[0].set_title("Missing % by Column")

    # Missing data heatmap (sample if large)
    sample_df = df.iloc[: min(100, len(df)), : min(20, len(df.columns))]

    import seaborn as sns

    sns.heatmap(
        sample_df.isnull(), cbar=False, yticklabels=False, cmap="YlOrRd", ax=axes[1]
    )
    axes[1].set_title("Missing Data Pattern (sample)")
    axes[1].set_xlabel("Columns")

    plt.tight_layout()
    return fig, axes


def plot_imputation_comparison(original, imputed, column_name="Value", figsize=(14, 5)):
    """
    Compare distributions before and after imputation.

    Parameters
    ----------
    original : array-like
        Original data with missing values
    imputed : array-like
        Data after imputation
    column_name : str
        Name of the variable
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original with missing
    orig_clean = np.array(original)[~np.isnan(original)]
    axes[0].hist(
        orig_clean, bins=30, alpha=0.7, color="steelblue", label="Original (observed)"
    )
    axes[0].set_title(f"{column_name}: Before Imputation")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    n_missing = np.sum(np.isnan(original))
    axes[0].text(
        0.95,
        0.95,
        f"N missing: {n_missing}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # After imputation
    axes[1].hist(imputed, bins=30, alpha=0.7, color="green", label="After imputation")
    axes[1].set_title(f"{column_name}: After Imputation")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    return fig, axes


# =============================================================================
# Spatial analysis visualization
# =============================================================================


def plot_semivariogram(
    lags, semivariance, model_fit=None, title="Semivariogram", ax=None
):
    """
    Plot empirical semivariogram with optional model fit.

    Parameters
    ----------
    lags : array-like
        Lag distances
    semivariance : array-like
        Semivariance values
    model_fit : array-like
        Fitted model values (optional)
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes to plot on

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(lags, semivariance, color="steelblue", s=50, label="Empirical")

    if model_fit is not None:
        ax.plot(lags, model_fit, color="red", linewidth=2, label="Model fit")
        ax.legend()

    ax.set_xlabel("Lag Distance")
    ax.set_ylabel("Semivariance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_spatial_autocorrelation(
    gdf, values, title="Spatial Autocorrelation", figsize=(14, 5)
):
    """
    Visualize spatial autocorrelation with map and Moran scatterplot.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Spatial data
    values : array-like
        Values to analyze
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Spatial map of values
    gdf_plot = gdf.copy()
    gdf_plot["value"] = values
    gdf_plot.plot(
        column="value", ax=axes[0], legend=True, cmap=CONTINUOUS_CMAP, markersize=30
    )
    axes[0].set_title(f"{title}: Spatial Distribution")

    # Moran scatterplot (standardized value vs spatial lag)
    from scipy.spatial import KDTree

    coords = np.array([[g.x, g.y] for g in gdf.geometry])
    tree = KDTree(coords)

    # Compute spatial lag (mean of k nearest neighbors)
    k = min(8, len(coords) - 1)
    distances, indices = tree.query(coords, k=k + 1)

    values_std = (values - np.mean(values)) / np.std(values)
    spatial_lag = np.array([np.mean(values_std[idx[1:]]) for idx in indices])

    axes[1].scatter(values_std, spatial_lag, alpha=0.5, color="steelblue")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Add regression line
    z = np.polyfit(values_std, spatial_lag, 1)
    p = np.poly1d(z)
    x_line = np.linspace(values_std.min(), values_std.max(), 100)
    axes[1].plot(
        x_line,
        p(x_line),
        "r-",
        linewidth=2,
        label=f"Slope (Moran's I proxy): {z[0]:.3f}",
    )

    axes[1].set_xlabel("Standardized Value")
    axes[1].set_ylabel("Spatial Lag")
    axes[1].set_title("Moran Scatterplot")
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


# =============================================================================
# ML results visualization
# =============================================================================


def plot_interpolation_results(
    original_points,
    original_values,
    interpolated_grid,
    extent,
    method_name="Interpolation",
    figsize=(14, 5),
):
    """
    Plot interpolation input and output side by side.

    Parameters
    ----------
    original_points : array-like
        (N, 2) array of point coordinates
    original_values : array-like
        Values at original points
    interpolated_grid : np.ndarray
        2D interpolated grid
    extent : tuple
        (xmin, xmax, ymin, ymax) for grid
    method_name : str
        Name of interpolation method
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original points
    sc = axes[0].scatter(
        original_points[:, 0],
        original_points[:, 1],
        c=original_values,
        cmap=CONTINUOUS_CMAP,
        s=50,
        edgecolor="white",
    )
    axes[0].set_title("Sample Points")
    plt.colorbar(sc, ax=axes[0], shrink=0.8)

    # Interpolated surface
    plot_raster(
        interpolated_grid,
        ax=axes[1],
        title=f"{method_name} Result",
        extent=extent,
        robust_stretch=False,
        origin="lower",
    )

    # Overlay points on interpolated surface
    axes[1].scatter(
        original_points[:, 0],
        original_points[:, 1],
        c="red",
        s=10,
        alpha=0.5,
        marker=".",
    )

    plt.tight_layout()
    return fig, axes


def plot_pca_results(pca, feature_names, figsize=(14, 5), annot=False):
    """
    Plot PCA explained variance and loadings.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    feature_names : list
        Names of original features
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Explained variance
    n_components = len(pca.explained_variance_ratio_)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    axes[0].bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        label="Individual",
    )
    axes[0].plot(range(1, n_components + 1), cum_var, "ro-", label="Cumulative")
    axes[0].axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("PCA Explained Variance")
    axes[0].legend()
    axes[0].set_xticks(range(1, n_components + 1))

    # Loadings heatmap
    n_show = min(3, n_components)
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        columns=[f"PC{i+1}" for i in range(n_show)],
        index=feature_names,
    )

    import seaborn as sns

    sns.heatmap(loadings, cmap="RdBu_r", center=0, annot=annot, fmt=".2f", ax=axes[1])
    axes[1].set_title("PCA Loadings")

    plt.tight_layout()
    return fig, axes


def plot_clustering_results(
    data,
    labels,
    centers=None,
    title="Clustering Results",
    ax=None,
    feature_x=0,
    feature_y=1,
):
    """
    Plot 2D clustering results.

    Parameters
    ----------
    data : np.ndarray
        (N, D) data array
    labels : array-like
        Cluster labels
    centers : np.ndarray
        Cluster centers (optional)
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes to plot on
    feature_x, feature_y : int
        Feature indices for x and y axes

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n_clusters = len(np.unique(labels))
    colors = CATEGORICAL_COLORS[:n_clusters]

    for i, color in enumerate(colors):
        mask = labels == i
        ax.scatter(
            data[mask, feature_x],
            data[mask, feature_y],
            c=color,
            label=f"Cluster {i}",
            alpha=0.6,
            s=30,
        )

    if centers is not None:
        ax.scatter(
            centers[:, feature_x],
            centers[:, feature_y],
            c="black",
            marker="X",
            s=200,
            edgecolor="white",
            linewidth=2,
            label="Centers",
        )

    ax.set_xlabel(f"Feature {feature_x}")
    ax.set_ylabel(f"Feature {feature_y}")
    ax.set_title(title)
    ax.legend()

    return ax


def compute_cluster_centroids(values, labels):
    """Compute centroid coordinates for each cluster label."""
    centroids = []
    for label in sorted(np.unique(labels)):
        mask = labels == label
        centroids.append(values[mask].mean(axis=0))
    return np.vstack(centroids)


def plot_kmeans_pca_scatter(
    X_pca, labels, title="K-means in PCA Space", ax=None, markersize=40
):
    """Plot PC1 vs PC2 with cluster centroids.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes containing the scatter plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    n_clusters = len(np.unique(labels))
    colors = CATEGORICAL_COLORS[:n_clusters]

    for i, color in enumerate(colors):
        mask = labels == i
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f"Cluster {i}",
            alpha=0.7,
            s=markersize,
        )

    centroids = compute_cluster_centroids(X_pca[:, :2], labels)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=200,
        edgecolor="white",
        linewidth=1.5,
        label="Centroids",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_elbow_silhouette(
    k_range, inertias, silhouettes, figsize=(14, 5), preferred_k=None
):
    """
    Plot elbow curve and silhouette scores for K-means.

    Parameters
    ----------
    k_range : array-like
        Range of k values tested
    inertias : array-like
        Inertia values for each k
    silhouettes : array-like
        Silhouette scores for each k
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Elbow plot
    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(k_range, silhouettes, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].grid(True, alpha=0.3)

    # Mark preferred or best k
    if preferred_k is None:
        best_k_idx = np.argmax(silhouettes)
        mark_k = k_range[best_k_idx]
        label = f"Best k={mark_k}"
    else:
        mark_k = preferred_k
        label = f"Preferred k={mark_k}"

    axes[1].axvline(mark_k, color="red", linestyle="--", label=label)
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


def plot_anomaly_scores(
    gdf, scores, binary_labels=None, title="Anomaly Detection", figsize=(14, 5)
):
    """
    Plot anomaly scores as continuous and binary classification.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Spatial data with geometries
    scores : array-like
        Continuous anomaly scores
    binary_labels : array-like
        Binary labels (1=anomaly, -1 or 0=normal), optional
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    gdf_plot = gdf.copy()
    gdf_plot["score"] = scores

    # Continuous scores
    gdf_plot.plot(column="score", ax=axes[0], legend=True, cmap="YlOrRd", markersize=30)
    axes[0].set_title(f"{title}: Anomaly Scores")

    # Binary classification
    if binary_labels is not None:
        gdf_plot["is_anomaly"] = binary_labels

        normal_mask = gdf_plot["is_anomaly"] <= 0
        anomaly_mask = gdf_plot["is_anomaly"] > 0

        gdf_plot[normal_mask].plot(
            ax=axes[1], color=ANOMALY_COLORS["normal"], markersize=30, label="Normal"
        )
        gdf_plot[anomaly_mask].plot(
            ax=axes[1], color=ANOMALY_COLORS["anomaly"], markersize=50, label="Anomaly"
        )
        axes[1].legend()
    else:
        # Use threshold on scores
        threshold = np.percentile(scores, 95)
        normal_mask = scores < threshold

        gdf_plot[normal_mask].plot(
            ax=axes[1], color=ANOMALY_COLORS["normal"], markersize=30, label="Normal"
        )
        gdf_plot[~normal_mask].plot(
            ax=axes[1],
            color=ANOMALY_COLORS["anomaly"],
            markersize=50,
            label="Anomaly (top 5%)",
        )
        axes[1].legend()

    axes[1].set_title(f"{title}: Classification")

    plt.tight_layout()
    return fig, axes


def plot_alteration_map(class_map, class_names=None, figsize=(10, 8)):
    """
    Plot alteration type classification map.

    Parameters
    ----------
    class_map : np.ndarray
        2D array with class labels (0=background, 1-6=alteration types)
    class_names : list
        Names for each class
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : tuple
    """
    if class_names is None:
        class_names = list(ALTERATION_COLORS.keys())

    fig, ax = plt.subplots(figsize=figsize)

    colors = list(ALTERATION_COLORS.values())
    cmap = mcolors.ListedColormap(colors[: len(class_names)])

    im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=len(class_names) - 1)
    ax.set_title("Alteration Type Classification")

    # Create legend
    patches = [
        Patch(facecolor=colors[i], label=class_names[i])
        for i in range(len(class_names))
    ]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    return fig, ax


def plot_prospectivity_map(prob_map, extent=None, threshold=0.5, figsize=(14, 5)):
    """
    Plot prospectivity probability map with thresholded classification.

    Parameters
    ----------
    prob_map : np.ndarray
        2D probability array
    extent : tuple
        (xmin, xmax, ymin, ymax) for display
    threshold : float
        Classification threshold
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Probability map
    im0 = axes[0].imshow(
        prob_map, cmap="RdYlGn", vmin=0, vmax=1, extent=extent, origin="upper"
    )
    axes[0].set_title("Prospectivity Probability")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Thresholded
    classified = (prob_map >= threshold).astype(int)
    cmap_binary = mcolors.ListedColormap(["#d73027", "#1a9850"])

    im1 = axes[1].imshow(classified, cmap=cmap_binary, extent=extent, origin="upper")
    axes[1].set_title(f"Prospective Areas (threshold={threshold})")

    patches = [
        Patch(facecolor="#d73027", label="Low"),
        Patch(facecolor="#1a9850", label="High"),
    ]
    axes[1].legend(handles=patches, loc="upper right")

    plt.tight_layout()
    return fig, axes


def plot_roc_pr_curves(y_true, y_prob, figsize=(14, 5)):
    """
    Plot ROC and Precision-Recall curves.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(
        fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})"
    )
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    axes[1].plot(
        recall, precision, color="steelblue", linewidth=2, label=f"PR (AP = {ap:.3f})"
    )
    axes[1].axhline(
        baseline, color="gray", linestyle="--", label=f"Baseline = {baseline:.3f}"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_feature_importance(importance_df, top_n=15, figsize=(10, 6)):
    """
    Plot feature importance bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    df_sorted = importance_df.nlargest(top_n, "importance")

    ax.barh(range(len(df_sorted)), df_sorted["importance"], color="steelblue")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Synthetic data generation
# =============================================================================


def generate_synthetic_geochemistry(
    n_samples=500,
    n_elements=10,
    n_clusters=4,
    spatial_extent=(0, 1000, 0, 1000),
    cluster_std=100,
    random_state=42,
):
    """
    Generate synthetic geochemistry point data with spatial clustering.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_elements : int
        Number of geochemical elements
    n_clusters : int
        Number of distinct geochemical populations
    spatial_extent : tuple
        (xmin, xmax, ymin, ymax)
    cluster_std : float
        Spatial spread of clusters
    random_state : int
        Random seed

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Synthetic geochemistry data
    """
    import geopandas as gpd
    from shapely.geometry import Point

    np.random.seed(random_state)

    xmin, xmax, ymin, ymax = spatial_extent

    # Generate cluster centers
    cluster_centers_xy = np.column_stack(
        [
            np.random.uniform(xmin + cluster_std, xmax - cluster_std, n_clusters),
            np.random.uniform(ymin + cluster_std, ymax - cluster_std, n_clusters),
        ]
    )

    # Generate distinct geochemical signatures for each cluster
    element_names = [f"Element_{i+1}" for i in range(n_elements)]
    cluster_signatures = np.random.lognormal(
        mean=3, sigma=1, size=(n_clusters, n_elements)
    )

    # Assign samples to clusters
    samples_per_cluster = n_samples // n_clusters

    all_coords = []
    all_values = []
    all_labels = []

    for c in range(n_clusters):
        n_c = samples_per_cluster if c < n_clusters - 1 else n_samples - len(all_coords)

        # Spatial positions (clustered)
        coords = np.column_stack(
            [
                np.random.normal(cluster_centers_xy[c, 0], cluster_std, n_c),
                np.random.normal(cluster_centers_xy[c, 1], cluster_std, n_c),
            ]
        )

        # Geochemical values (based on signature + noise)
        values = cluster_signatures[c] * np.random.lognormal(0, 0.3, (n_c, n_elements))

        all_coords.append(coords)
        all_values.append(values)
        all_labels.extend([c] * n_c)

    coords = np.vstack(all_coords)
    values = np.vstack(all_values)

    # Clip to extent
    coords[:, 0] = np.clip(coords[:, 0], xmin, xmax)
    coords[:, 1] = np.clip(coords[:, 1], ymin, ymax)

    # Create GeoDataFrame
    df = pd.DataFrame(values, columns=element_names)
    df["X"] = coords[:, 0]
    df["Y"] = coords[:, 1]
    df["true_cluster"] = all_labels

    geometry = [Point(x, y) for x, y in coords]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32610")

    return gdf


def generate_synthetic_raster(
    shape=(200, 200),
    extent=(0, 1000, 0, 1000),
    n_anomalies=5,
    anomaly_radius=30,
    background_mean=10,
    anomaly_strength=50,
    noise_level=2,
    random_state=42,
):
    """
    Generate synthetic continuous raster with anomalies.

    Parameters
    ----------
    shape : tuple
        (rows, cols) of output raster
    extent : tuple
        (xmin, xmax, ymin, ymax) spatial extent
    n_anomalies : int
        Number of anomalous regions
    anomaly_radius : float
        Radius of anomalies in grid cells
    background_mean : float
        Mean background value
    anomaly_strength : float
        Added value at anomaly centers
    noise_level : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns
    -------
    data : np.ndarray
        2D raster array
    extent : tuple
        Spatial extent
    """
    np.random.seed(random_state)

    rows, cols = shape

    # Background with spatial trend
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xx, yy = np.meshgrid(x, y)

    # Gentle spatial trend
    data = background_mean + 2 * np.sin(2 * np.pi * xx) + 2 * np.cos(2 * np.pi * yy)

    # Add anomalies
    for _ in range(n_anomalies):
        cx = np.random.randint(anomaly_radius, cols - anomaly_radius)
        cy = np.random.randint(anomaly_radius, rows - anomaly_radius)

        # Gaussian anomaly
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
                if dist < anomaly_radius * 2:
                    data[i, j] += anomaly_strength * np.exp(
                        -(dist**2) / (2 * anomaly_radius**2)
                    )

    # Add noise
    data += np.random.normal(0, noise_level, shape)

    return data, extent


def generate_synthetic_vector_geometries(
    n_points=50, n_lines=10, n_polygons=8, extent=(0, 1000, 0, 1000), random_state=42
):
    """
    Generate synthetic point, line, and polygon geometries.

    Parameters
    ----------
    n_points : int
        Number of points
    n_lines : int
        Number of line features
    n_polygons : int
        Number of polygon features
    extent : tuple
        (xmin, xmax, ymin, ymax)
    random_state : int
        Random seed

    Returns
    -------
    gdf_points, gdf_lines, gdf_polygons : tuple of GeoDataFrames
    """
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon

    np.random.seed(random_state)
    xmin, xmax, ymin, ymax = extent

    # Points
    points = [
        Point(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        for _ in range(n_points)
    ]
    gdf_points = gpd.GeoDataFrame(
        {"id": range(n_points), "value": np.random.uniform(0, 100, n_points)},
        geometry=points,
        crs="EPSG:32610",
    )

    # Lines (random walks)
    lines = []
    for _ in range(n_lines):
        start = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]
        coords = [start]
        for _ in range(np.random.randint(3, 8)):
            new_pt = [
                coords[-1][0] + np.random.normal(0, 50),
                coords[-1][1] + np.random.normal(0, 50),
            ]
            new_pt[0] = np.clip(new_pt[0], xmin, xmax)
            new_pt[1] = np.clip(new_pt[1], ymin, ymax)
            coords.append(new_pt)
        lines.append(LineString(coords))

    gdf_lines = gpd.GeoDataFrame(
        {"id": range(n_lines), "length": [l.length for l in lines]},
        geometry=lines,
        crs="EPSG:32610",
    )

    # Polygons (random convex shapes)
    polygons = []
    for _ in range(n_polygons):
        cx = np.random.uniform(xmin + 50, xmax - 50)
        cy = np.random.uniform(ymin + 50, ymax - 50)
        n_vertices = np.random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
        radii = np.random.uniform(20, 80, n_vertices)
        coords = [
            (cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)
        ]
        coords.append(coords[0])  # close polygon
        polygons.append(Polygon(coords))

    gdf_polygons = gpd.GeoDataFrame(
        {"id": range(n_polygons), "area": [p.area for p in polygons]},
        geometry=polygons,
        crs="EPSG:32610",
    )

    return gdf_points, gdf_lines, gdf_polygons


def add_missing_data(
    df, missing_pct=0.1, columns=None, pattern="random", random_state=42
):
    """
    Add missing values to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    missing_pct : float or tuple
        Percentage of values to make missing (0-1) or (min, max) range
    columns : list
        Columns to add missing values to (None = all numeric)
    pattern : str
        'random' or 'spatial' (clustered missing)
    random_state : int
        Random seed

    Returns
    -------
    df_missing : pd.DataFrame
        DataFrame with missing values
    """
    rng = np.random.default_rng(random_state)
    df_missing = df.copy()

    if columns is None:
        columns = df_missing.select_dtypes(include=[np.number]).columns.tolist()

    missing_summary = {}
    for col in columns:
        if col in ["X", "Y", "geometry"]:
            continue

        if isinstance(missing_pct, (tuple, list)) and len(missing_pct) == 2:
            low, high = missing_pct
            pct = rng.uniform(low, high)
        else:
            pct = missing_pct
        n_missing = int(len(df_missing) * pct)
        missing_summary[col] = {"pct": pct * 100, "n": n_missing}

        if pattern == "random":
            missing_idx = rng.choice(df_missing.index, n_missing, replace=False)
        else:  # spatial clustering of missing values
            # Cluster missing values in one corner.
            if "geometry" in df_missing.columns:
                xs = df_missing.geometry.x
                ys = df_missing.geometry.y
                corner_mask = (xs < xs.median()) & (ys < ys.median())
                corner_idx = df_missing[corner_mask].index
                if len(corner_idx) >= n_missing:
                    missing_idx = rng.choice(corner_idx, n_missing, replace=False)
                else:
                    missing_idx = rng.choice(df_missing.index, n_missing, replace=False)
            elif "X" in df_missing.columns and "Y" in df_missing.columns:
                corner_mask = (df_missing["X"] < df_missing["X"].median()) & (
                    df_missing["Y"] < df_missing["Y"].median()
                )
                corner_idx = df_missing[corner_mask].index
                if len(corner_idx) >= n_missing:
                    missing_idx = rng.choice(corner_idx, n_missing, replace=False)
                else:
                    missing_idx = rng.choice(df_missing.index, n_missing, replace=False)
            else:
                missing_idx = rng.choice(df_missing.index, n_missing, replace=False)

        df_missing.loc[missing_idx, col] = np.nan

    # Store summary as attribute for later access
    df_missing.attrs["missing_summary"] = missing_summary
    return df_missing


# =============================================================================
# I/O helpers
# =============================================================================


def load_raster(path):
    """Load a single-band raster and return data, extent, and CRS."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return data, extent, src.crs


def load_vector(path):
    """Load a vector dataset into a GeoDataFrame."""
    import geopandas as gpd

    return gpd.read_file(path)


def to_analysis_crs(gdf):
    """Project vector data into the fixed analysis CRS."""
    return gdf.to_crs(ANALYSIS_CRS)


def get_point_xy(gdf):
    """Return point coordinates as an ``(n, 2)`` numpy array."""
    if len(gdf) == 0:
        return np.empty((0, 2), dtype=float)
    geom_types = set(gdf.geometry.geom_type.unique())
    if not geom_types.issubset({"Point"}):
        raise ValueError(f"Expected point geometries, got: {sorted(geom_types)}")
    return np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])


def get_geochem_columns(geochem_gdf):
    """Return all non-geometry geochemistry columns."""
    return [c for c in geochem_gdf.columns if c != "geometry"]


def rasterize_lithology(gdf, shape, extent, value_col=None):
    """Rasterize a lithology GeoDataFrame into a categorical raster."""
    import pandas as pd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    xmin, xmax, ymin, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, shape[1], shape[0])

    if value_col is None:
        candidates = [
            c for c in gdf.columns if c.lower() in ["lithology", "unit", "rocktype"]
        ]
        value_col = candidates[0] if candidates else None

    if value_col is None:
        values = pd.factorize(gdf.index)[0] + 1
        shapes = list(zip(gdf.geometry, values))
    else:
        values = pd.factorize(gdf[value_col])[0] + 1
        shapes = list(zip(gdf.geometry, values))

    raster = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return raster


# =============================================================================
# Notebook workflow helpers
# =============================================================================


def resolve_path(path):
    """Return a project-rooted absolute path for relative inputs."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def require_path(path, name, allow_dir=False):
    """Validate that a required path exists."""
    if not path:
        raise ValueError(f"Missing required path for {name}.")
    p = resolve_path(path)
    if allow_dir:
        if not p.exists() or not p.is_dir():
            raise ValueError(f"{name} must be an existing directory: {path}")
    else:
        if not p.exists():
            raise ValueError(f"{name} must be an existing file: {path}")
    return p


def require_one(path_candidates, label):
    """Return the first existing path from a list of candidates."""
    candidates = []
    for p in path_candidates:
        path = resolve_path(p)
        candidates.append(path)
        if path.exists():
            return path
    raise ValueError(
        f"Missing required output for {label}: {', '.join(str(p) for p in candidates)}"
    )


def load_training_data(data_config=None):
    """
    Load the core raster and vector datasets used in the teaching notebooks.

    This helper applies the project's fixed assumptions about file locations and
    coordinate reference systems. It reads the main continuous raster, the
    lithology polygons, and the geochemistry points, then returns them in a
    single dictionary so notebook cells do not need to manage I/O themselves.

    If no categorical raster is supplied, the lithology polygons are rasterized
    onto the same grid as the continuous raster.
    """
    if data_config is None:
        data_config = DEFAULT_DATA_CONFIG
    continuous_path = require_path(
        data_config.get("continuous_raster_path"), "continuous_raster_path"
    )
    continuous_raster, raster_extent, raster_crs = load_raster(continuous_path)
    working_crs = ANALYSIS_CRS

    vector_path = require_path(data_config.get("vector_path"), "vector_path")
    vector_gdf = to_analysis_crs(load_vector(vector_path))

    if data_config.get("categorical_raster_path"):
        categorical_path = require_path(
            data_config.get("categorical_raster_path"), "categorical_raster_path"
        )
        categorical_raster, _, _ = load_raster(categorical_path)
    else:
        categorical_raster = rasterize_lithology(
            vector_gdf, continuous_raster.shape, raster_extent
        )

    geochem_path = require_path(
        data_config.get("geochem_points_path"), "geochem_points_path"
    )
    geochem_gdf = to_analysis_crs(load_vector(geochem_path))

    print("\nTotal Files: 16 \n - Vector Files: 3 \n - Raster Files: 13\n")
    print("Raster shape:", continuous_raster.shape)
    print("Working CRS:", working_crs)
    print("Vector records:", len(vector_gdf))
    print("Geochem records:", len(geochem_gdf))

    return {
        "continuous_raster": continuous_raster,
        "raster_extent": raster_extent,
        "raster_crs": raster_crs,
        "working_crs": working_crs,
        "vector_gdf": vector_gdf,
        "categorical_raster": categorical_raster,
        "geochem_gdf": geochem_gdf,
    }


def plot_data_format_examples(
    data_config=None, vector_gdf=None, geochem_gdf=None, figsize=(12, 10)
):
    """Plot example vector and raster formats from configured paths."""
    if data_config is None:
        data_config = DEFAULT_DATA_CONFIG
    from pathlib import Path

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_vector(
        vector_gdf, ax=axes[0, 0], title="Lithology (Polygons)", categorical=True
    )
    plot_vector(geochem_gdf, ax=axes[0, 1], title="Geochem (Points)", markersize=30)

    geo_dir = require_path(
        data_config.get("geophysics_dir"), "geophysics_dir", allow_dir=True
    )
    geo_tifs = sorted(Path(geo_dir).glob("*.tif"))
    if not geo_tifs:
        raise ValueError(f"No GeoTIFFs found in {geo_dir}")

    geophys_data, geophys_extent, _ = load_raster(geo_tifs[0])
    plot_raster(
        geophys_data, ax=axes[1, 0], title="Geophysics (Raster)", extent=geophys_extent
    )

    spec_dir = require_path(
        data_config.get("spectral_indices_dir"), "spectral_indices_dir", allow_dir=True
    )
    spec_tifs = sorted(Path(spec_dir).glob("*.tif"))
    if not spec_tifs:
        raise ValueError(f"No GeoTIFFs found in {spec_dir}")

    spectral_data, spectral_extent, _ = load_raster(spec_tifs[0])
    plot_raster(
        spectral_data,
        ax=axes[1, 1],
        title="Spectral Index (Raster)",
        extent=spectral_extent,
    )

    plt.tight_layout()
    return fig, axes


def prepare_geochem_features(geochem_gdf, exclude_cols=None, value_hint="cu"):
    """
    Pick numeric geochemistry columns and a default example variable.

    This is a convenience helper for workflows that still want an explicit list
    of assay columns plus one representative variable for interpolation or
    plotting. It filters out common coordinate and identifier fields, then
    chooses a default ``value_col`` by looking for a name that contains
    ``value_hint``.
    """
    numeric_cols = geochem_gdf.select_dtypes(include=[np.number]).columns.tolist()
    base_exclude = {"X", "Y", "id", "coord_x", "coord_y", "elevation_m"}
    if exclude_cols:
        base_exclude.update(exclude_cols)
    feature_cols = [c for c in numeric_cols if c not in base_exclude]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in geochem data.")

    value_candidates = [c for c in numeric_cols if value_hint in c.lower()]
    value_col = value_candidates[0] if value_candidates else feature_cols[0]
    return feature_cols, value_col


def log_transform(values):
    """Apply a stable log1p transform with a non-negative shift."""
    values = np.array(values, dtype=float)
    if values.ndim == 2:
        shift = np.nanmin(values, axis=0)
        return np.log1p(values - shift + 1)
    shift = np.nanmin(values)
    return np.log1p(values - shift + 1)


def apply_transform(values, transform="log1p"):
    """Apply a named transform to values."""
    if transform in (None, "none", "identity"):
        return np.array(values, dtype=float)
    if transform == "log1p":
        return log_transform(values)
    raise ValueError(f"Unsupported transform: {transform}")


def clip_quantiles(values, quantiles=None):
    """Clip values to the provided quantile range."""
    if quantiles is None:
        return np.array(values, dtype=float)
    if not isinstance(quantiles, (tuple, list)) or len(quantiles) != 2:
        raise ValueError("quantiles must be a (low, high) tuple")
    q_low, q_high = quantiles
    values = np.array(values, dtype=float)
    if values.ndim == 2:
        lows = np.nanquantile(values, q_low, axis=0)
        highs = np.nanquantile(values, q_high, axis=0)
        return np.clip(values, lows, highs)
    low = np.nanquantile(values, q_low)
    high = np.nanquantile(values, q_high)
    return np.clip(values, low, high)


def scale_features(df, feature_cols):
    """Standardize numeric features and return a DataFrame."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    return scaled_df, scaler


def mean_impute(values):
    """Apply mean imputation and return imputed values and the imputer."""
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")
    imputed_values = imputer.fit_transform(values)
    return imputed_values, imputer


def impute_values(values, strategy="mean"):
    """Apply imputation and return imputed values and the imputer."""
    from sklearn.impute import SimpleImputer

    if strategy not in {"mean", "median", "most_frequent"}:
        raise ValueError(f"Unsupported impute strategy: {strategy}")
    imputer = SimpleImputer(strategy=strategy)
    imputed_values = imputer.fit_transform(values)
    return imputed_values, imputer


def prepare_pca_inputs(
    geochem_gdf,
    feature_cols=None,
    exclude_cols=None,
    transform="log1p",
    scale_features=True,
    clip_quantiles_range=None,
):
    """
    Build the numeric matrix used as input to PCA.

    The function starts from the geochemistry GeoDataFrame, selects the element
    columns to include, optionally clips extreme values, applies the chosen
    transform, and optionally standardizes each column so all variables are on a
    comparable scale. The returned dictionary keeps both the intermediate arrays
    and the final matrix so the notebook can explain what PCA is operating on.

    Returns
    -------
    dict
        ``X_geochem`` is the raw numeric matrix after column selection,
        ``X_log`` is the transformed version, ``X_scaled`` is the matrix passed
        to PCA, ``pca_cols`` records the element names in column order, and
        ``scaler`` stores the fitted ``StandardScaler`` when scaling is used.
    """
    from sklearn.preprocessing import StandardScaler

    if feature_cols is None:
        feature_cols = get_geochem_columns(geochem_gdf)

    pca_exclude = {"id", "elevation_m"}
    if exclude_cols:
        pca_exclude.update(exclude_cols)
    pca_cols = [c for c in feature_cols if c not in pca_exclude]

    X_geochem = geochem_gdf[pca_cols].values
    X_geochem = clip_quantiles(X_geochem, clip_quantiles_range)
    X_transformed = apply_transform(X_geochem, transform=transform)

    scaler = None
    X_scaled = X_transformed
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)

    print(f"Original dimensions: {X_geochem.shape[1]}")

    return {
        "X_geochem": X_geochem,
        "X_log": X_transformed,
        "X_scaled": X_scaled,
        "pca_cols": pca_cols,
        "scaler": scaler,
    }


def plot_pca_variance(pca, figsize=(7, 5)):
    """Plot explained variance for PCA."""
    fig, ax = plt.subplots(figsize=figsize)
    n_components = len(pca.explained_variance_ratio_)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    ax.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        label="Individual",
    )
    ax.plot(range(1, n_components + 1), cum_var, "ro-", label="Cumulative")
    thresholds = [0.5, 0.75, 0.9]
    colors = ["#8e6c8a", "#5b8c5a", "#6c757d"]
    for thresh, color in zip(thresholds, colors):
        idx = np.argmax(cum_var >= thresh) + 1
        ax.axhline(y=thresh, color=color, linestyle="--", alpha=0.6)
        ax.axvline(x=idx, color=color, linestyle=":", alpha=0.6)
        ax.text(
            idx + 0.2,
            thresh + 0.02,
            f"{int(thresh*100)}% -> {idx} PCs",
            color=color,
            fontsize=9,
        )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    tick_step = max(1, n_components // 12)
    ax.set_xticks(range(1, n_components + 1, tick_step))
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_pca_loadings(
    pca, feature_names, n_components=5, top_n_pos=5, top_n_neg=5, figsize=(8, 10)
):
    """Plot top positive and negative PCA loadings per component."""
    n_show = min(n_components, len(pca.components_))
    fig, axes = plt.subplots(n_show, 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes[:n_show]):
        loadings = pd.Series(pca.components_[i], index=feature_names)
        top_pos = loadings[loadings > 0].sort_values(ascending=False).head(top_n_pos)
        top_neg = loadings[loadings < 0].sort_values().head(top_n_neg)
        top = pd.concat([top_neg, top_pos]).sort_values()
        colors = ["#d95f02" if v < 0 else "#1b9e77" for v in top.values]
        ax.barh(top.index, top.values, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        var_pct = pca.explained_variance_ratio_[i] * 100
        ax.set_title(
            f"PC{i+1} Top +{top_n_pos} / -{top_n_neg} Loadings "
            f"({var_pct:.1f}% variance)"
        )
        ax.set_xlabel("Loading")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return fig, axes


def prepare_interpolation_inputs(
    geochem_gdf, value_col, grid_resolution=60, padding=0.05
):
    """Build interpolation grid and inputs from point data."""
    sample_coords = get_point_xy(geochem_gdf)
    sample_values = geochem_gdf[value_col].values

    xmin, xmax = sample_coords[:, 0].min(), sample_coords[:, 0].max()
    ymin, ymax = sample_coords[:, 1].min(), sample_coords[:, 1].max()

    pad_x = (xmax - xmin) * padding
    pad_y = (ymax - ymin) * padding
    interp_extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    xmin, xmax, ymin, ymax = interp_extent
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("Invalid interpolation extent.")

    if width >= height:
        nx = grid_resolution
        ny = max(10, int(round(grid_resolution * height / width)))
    else:
        ny = grid_resolution
        nx = max(10, int(round(grid_resolution * width / height)))

    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    return sample_coords, sample_values, grid_points, (ny, nx), interp_extent


def show_plot():
    """Render the current matplotlib figure."""
    plt.show()


def run_idw_interpolation(
    sample_coords,
    sample_values,
    grid_points,
    grid_shape,
    interp_extent,
    power=2,
    n_neighbors=12,
    show=True,
):
    """Run IDW interpolation and plot the result."""
    idw_pred = idw_interpolation(
        sample_coords, sample_values, grid_points, power=power, n_neighbors=n_neighbors
    )
    idw_grid = idw_pred.reshape(grid_shape)
    plot_interpolation_results(
        sample_coords, sample_values, idw_grid, interp_extent, method_name="IDW"
    )
    if show:
        plt.show()
    return idw_grid


def run_kriging_interpolation(
    sample_coords,
    sample_values,
    grid_points,
    grid_shape,
    interp_extent,
    n_neighbors=12,
    show=True,
):
    """Fit a variogram and run ordinary kriging with a plot."""
    lags, semivar = compute_semivariogram(sample_coords, sample_values)
    nugget, sill, range_param = fit_variogram(lags, semivar)

    kriging_pred, kriging_var = ordinary_kriging(
        sample_coords,
        sample_values,
        grid_points,
        nugget,
        sill,
        range_param,
        n_neighbors=n_neighbors,
    )
    kriging_grid = kriging_pred.reshape(grid_shape)
    plot_interpolation_results(
        sample_coords,
        sample_values,
        kriging_grid,
        interp_extent,
        method_name="Ordinary Kriging",
    )
    if show:
        plt.show()
    return kriging_grid, kriging_var.reshape(grid_shape)


def run_interpolation_workflow(
    geochem_gdf,
    value_col,
    grid_resolution=60,
    padding=0.05,
    power=2,
    n_neighbors=12,
    show=True,
):
    """Run IDW and Kriging interpolation demos with plots."""
    sample_coords, sample_values, grid_points, grid_shape, interp_extent = (
        prepare_interpolation_inputs(
            geochem_gdf, value_col, grid_resolution=grid_resolution, padding=padding
        )
    )

    idw_pred = idw_interpolation(
        sample_coords, sample_values, grid_points, power=power, n_neighbors=n_neighbors
    )
    idw_grid = idw_pred.reshape(grid_shape)
    plot_interpolation_results(
        sample_coords, sample_values, idw_grid, interp_extent, method_name="IDW"
    )
    if show:
        plt.show()

    lags, semivar = compute_semivariogram(sample_coords, sample_values)
    nugget, sill, range_param = fit_variogram(lags, semivar)

    kriging_pred, kriging_var = ordinary_kriging(
        sample_coords,
        sample_values,
        grid_points,
        nugget,
        sill,
        range_param,
        n_neighbors=n_neighbors,
    )
    kriging_grid = kriging_pred.reshape(grid_shape)
    plot_interpolation_results(
        sample_coords,
        sample_values,
        kriging_grid,
        interp_extent,
        method_name="Ordinary Kriging",
    )
    if show:
        plt.show()

    return {
        "idw_grid": idw_grid,
        "kriging_grid": kriging_grid,
        "kriging_variance": kriging_var.reshape(grid_shape),
        "interp_extent": interp_extent,
    }


def plot_spatial_pca_components(
    geochem_gdf, X_pca, pca, n_components=3, cmap=DIVERGING_CMAP, figsize=(16, 5)
):
    """Plot the first few PCA components in map view."""
    fig, axes = plt.subplots(1, n_components, figsize=figsize)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        if i >= X_pca.shape[1]:
            ax.axis("off")
            continue
        gdf_temp = geochem_gdf.copy()
        gdf_temp[f"PC{i+1}"] = X_pca[:, i]
        gdf_temp.plot(column=f"PC{i+1}", ax=ax, legend=True, cmap=cmap, markersize=30)
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")

    plt.tight_layout()
    return fig, axes


def interactive_spatial_pca_components(
    geochem_gdf,
    X_pca,
    pca,
    n_components=5,
    cmap=DIVERGING_CMAP,
    markersize=30,
    figsize=(6, 5),
):
    """Interactive PCA component map with dropdown selection."""
    import ipywidgets as widgets
    from IPython.display import display

    n_show = min(n_components, X_pca.shape[1])
    options = [
        (f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% variance)", i)
        for i in range(n_show)
    ]
    comp_dropdown = widgets.Dropdown(options=options, description="Component")
    output = widgets.Output()

    def render(*_):
        output.clear_output(wait=True)
        with output:
            fig, ax = plt.subplots(figsize=figsize)
            i = comp_dropdown.value
            gdf_temp = geochem_gdf.copy()
            gdf_temp["component"] = X_pca[:, i]
            gdf_temp.plot(
                column="component", ax=ax, legend=True, cmap=cmap, markersize=markersize
            )
            ax.set_title(
                f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% variance)"
            )
            display(fig)
            plt.close(fig)

    comp_dropdown.observe(render, names="value")
    render()
    display(widgets.VBox([comp_dropdown, output]))
    return comp_dropdown


def run_pca_workflow(geochem_gdf, feature_cols, exclude_cols=None, show=True):
    """Run PCA with standard preprocessing and plots."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    pca_exclude = {"id", "elevation_m"}
    if exclude_cols:
        pca_exclude.update(exclude_cols)
    pca_cols = [c for c in feature_cols if c not in pca_exclude]

    X_geochem = geochem_gdf[pca_cols].values
    X_log = np.log(X_geochem + 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print(f"Original dimensions: {X_geochem.shape[1]}")
    plot_pca_results(pca, pca_cols, figsize=(14, 5))
    if show:
        plt.show()

    plot_spatial_pca_components(geochem_gdf, X_pca, pca)
    if show:
        plt.show()

    return {
        "pca": pca,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "pca_cols": pca_cols,
    }


def plot_silhouette_scores(k_range, silhouettes, preferred_k=None, figsize=(7, 5)):
    """Plot silhouette scores across k values."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_range, silhouettes, "o-", color="teal")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Scores by k")
    ax.grid(True, alpha=0.3)

    if preferred_k is not None:
        ax.axvline(
            preferred_k,
            color="tomato",
            linestyle="--",
            alpha=0.7,
            label=f"Chosen k = {preferred_k}",
        )
        ax.legend()

    plt.tight_layout()
    return fig, ax


def choose_lithology_column(vector_gdf, candidates=None):
    """Pick a representative lithology column if available."""
    if candidates is None:
        candidates = [
            "lithology_family",
            "main_lithology",
            "geological_era",
            "tectonic_setting",
        ]
    for col in candidates:
        if col in vector_gdf.columns:
            return col
    return None


def plot_clusters_on_lithology(
    vector_gdf, geochem_gdf, cluster_labels, lith_column=None, figsize=(10, 8)
):
    """Overlay clustered points on lithology polygons."""
    fig, ax = plt.subplots(figsize=figsize)
    cluster_colors = [
        "#D55E00",
        "#0072B2",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#999999",
    ]
    lith_colors = [
        "#d9e7f5",
        "#eadccf",
        "#dbe8d2",
        "#e6d9eb",
        "#f0e3bf",
        "#d9e7ea",
        "#ecd6d6",
        "#dfdcf0",
    ]

    if lith_column is None:
        lith_column = choose_lithology_column(vector_gdf)

    lith_handles = []
    if lith_column:
        lith_series = vector_gdf[lith_column].dropna().astype(str)
        lith_units = pd.Index(sorted(lith_series.unique()))
        n_lith = len(lith_units)
        if n_lith > len(lith_colors):
            base = plt.cm.tab20(np.linspace(0, 1, n_lith))[:, :3]
            softened = 0.55 * base + 0.45
            lith_cmap = mcolors.ListedColormap(softened)
            lith_color_values = [tuple(color) for color in softened]
        else:
            lith_color_values = lith_colors[:n_lith]
            lith_cmap = mcolors.ListedColormap(lith_color_values)

        plot_vector(
            vector_gdf,
            column=lith_column,
            categorical=True,
            categorical_cmap=lith_cmap,
            ax=ax,
            title="Clusters on Lithology",
            alpha=0.60,
            edgecolor="#8a8a8a",
            linewidth=0.35,
            legend=False,
        )
        lith_handles = [
            Patch(facecolor=color, edgecolor="#8a8a8a", linewidth=0.5, label=unit)
            for unit, color in zip(lith_units, lith_color_values)
        ]
    else:
        vector_gdf.plot(
            ax=ax, facecolor="#ececec", edgecolor="#9aa0a6", linewidth=0.35, alpha=0.45
        )
        ax.set_title("Clusters on Lithology")

    gdf_clustered = geochem_gdf.copy()
    gdf_clustered["cluster"] = cluster_labels
    unique_clusters = sorted(pd.unique(cluster_labels))
    n_clusters = len(unique_clusters)
    cluster_color_values = cluster_colors[:n_clusters]
    cluster_cmap = mcolors.ListedColormap(cluster_color_values)
    gdf_clustered.plot(
        column="cluster",
        ax=ax,
        legend=False,
        categorical=True,
        cmap=cluster_cmap,
        markersize=22,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.95,
    )

    cluster_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=f"Cluster {cluster}",
        )
        for cluster, color in zip(unique_clusters, cluster_color_values)
    ]
    cluster_legend = ax.legend(
        handles=cluster_handles,
        title="Cluster Legend",
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
    )
    ax.add_artist(cluster_legend)

    if lith_handles:
        if len(lith_handles) > 8:
            extra_height = min(0.22 * len(lith_handles), 6.0)
            fig.set_size_inches(figsize[0], figsize[1] + extra_height)
            fig.legend(
                handles=lith_handles,
                title=lith_column.replace("_", " ").title(),
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02),
                frameon=True,
                framealpha=0.95,
                facecolor="white",
                fontsize=7.5,
                title_fontsize=9,
                ncol=1,
            )
            fig.subplots_adjust(bottom=0.36)
        else:
            ax.legend(
                handles=lith_handles,
                title=lith_column.replace("_", " ").title(),
                loc="upper right",
                frameon=True,
                framealpha=0.95,
                facecolor="white",
                fontsize=8,
                title_fontsize=9,
                ncol=1,
            )

    ax.set_facecolor("#fcfcfc")
    if not lith_handles or len(lith_handles) <= 8:
        plt.tight_layout()
    return fig, ax


def run_kmeans_overlay(
    vector_gdf, geochem_gdf, X_scaled, n_clusters=4, lith_column=None, show=True
):
    """Cluster geochemistry and plot on lithology."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    plot_clusters_on_lithology(
        vector_gdf, geochem_gdf, cluster_labels, lith_column=lith_column
    )
    if show:
        plt.show()
    return cluster_labels


def run_isolation_forest(
    geochem_gdf,
    feature_cols,
    exclude_cols=None,
    contamination=0.05,
    n_estimators=200,
    random_state=42,
    show=True,
):
    """Run Isolation Forest anomaly detection with a plot."""
    from sklearn.ensemble import IsolationForest

    anom_exclude = {"id", "coord_x", "coord_y", "elevation_m"}
    if exclude_cols:
        anom_exclude.update(exclude_cols)
    anom_cols = [c for c in feature_cols if c not in anom_exclude]

    X_anom = geochem_gdf[anom_cols].values
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    labels = iso.fit_predict(X_anom)
    scores = -iso.decision_function(X_anom)

    plot_anomaly_scores(
        geochem_gdf, scores, binary_labels=labels, title="Isolation Forest"
    )
    if show:
        plt.show()

    return labels, scores


def prepare_anomaly_inputs(geochem_gdf, feature_cols, exclude_cols=None):
    """Select numeric inputs for anomaly detection."""
    anom_exclude = {"id", "coord_x", "coord_y", "elevation_m"}
    if exclude_cols:
        anom_exclude.update(exclude_cols)
    anom_cols = [c for c in feature_cols if c not in anom_exclude]
    X_anom = geochem_gdf[anom_cols].values
    return X_anom, anom_cols


def fit_isolation_forest_model(
    X_anom, contamination=0.05, n_estimators=200, random_state=42
):
    """Fit Isolation Forest and return labels and anomaly scores."""
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    labels = iso.fit_predict(X_anom)
    scores = -iso.decision_function(X_anom)
    return labels, scores


def plot_anomaly_score_distribution(scores, contamination=0.05, figsize=(7, 5)):
    """Plot anomaly score distribution with a percentile cutoff."""
    scores = np.array(scores, dtype=float)
    cutoff = np.nanpercentile(scores, 100 * (1 - contamination))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(scores, bins=30, color="slategray", edgecolor="white", alpha=0.8)
    ax.axvline(
        cutoff,
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"Anomaly cutoff ({contamination:.0%})",
    )
    ax.set_title("Isolation Forest Score Distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    return fig, ax


def default_alteration_weights():
    """Default alteration weights for spectral halo classification."""
    return {
        "Advanced Argillic": {
            "Clay_AlOH": 0.4,
            "Silica": 0.25,
            "Iron_Oxide": 0.15,
            "Alt_Composite": 0.2,
        },
        "Phyllic": {
            "Clay_AlOH": 0.4,
            "Iron_Oxide": 0.25,
            "Silica": 0.15,
            "Alt_Composite": 0.2,
        },
        "Argillic": {
            "Clay_AlOH": 0.4,
            "Silica": 0.2,
            "Laterite": 0.2,
            "Alt_Composite": 0.2,
        },
        "Propylitic": {
            "Ferrous_Iron": 0.4,
            "Clay_AlOH": 0.2,
            "Laterite": 0.2,
            "Silica": 0.2,
        },
        "Gossan": {
            "Iron_Oxide": 0.35,
            "Gossan": 0.35,
            "Silica": 0.15,
            "Alt_Composite": 0.15,
        },
        "Laterite": {"Laterite": 0.6, "Iron_Oxide": 0.2, "Clay_AlOH": 0.2},
    }


def summarize_alteration_classes(class_map, class_names):
    """Print class counts for alteration classification."""
    print("Alteration type classes:")
    print("0: Background")
    for i, name in enumerate(class_names[1:], 1):
        count = (class_map == i).sum()
        print(f"{i}: {name} ({count} pixels, {count / class_map.size * 100:.1f}%)")


def load_spectral_indices_dir(spectral_dir=None):
    """Load spectral indices GeoTIFFs from a directory."""
    from pathlib import Path

    if spectral_dir is None:
        spectral_dir = DEFAULT_DATA_CONFIG.get("spectral_indices_dir")
    spectral_dir = require_path(spectral_dir, "spectral_indices_dir", allow_dir=True)
    spectral_indices = {}
    spectral_extent = None
    for tif_path in sorted(Path(spectral_dir).glob("*.tif")):
        data, extent, _ = load_raster(tif_path)
        data = np.array(data, dtype=float)
        data[~np.isfinite(data)] = np.nan
        spectral_indices[tif_path.stem] = data
        if spectral_extent is None:
            spectral_extent = extent

    if not spectral_indices:
        raise ValueError(f"No GeoTIFFs found in {spectral_dir}")

    return spectral_indices, spectral_extent


def spectral_valid_mask(spectral_indices):
    """Derive a finite-value mask from the first spectral index."""
    first_index = next(iter(spectral_indices.values()))
    return np.isfinite(first_index)


def plot_spectral_indices_grid(spectral_indices, cols=3, figsize_per=(5, 4)):
    """Plot a grid of spectral indices."""
    n_indices = len(spectral_indices)
    rows = int(np.ceil(n_indices / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, data) in zip(axes, spectral_indices.items()):
        plot_raster(data, ax=ax, title=name, cmap="viridis", robust_stretch=True)

    for ax in axes[len(spectral_indices) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig, axes


def run_halo_detection_workflow(
    spectral_indices,
    presence_quantile=0.9,
    sigma_px=50,
    clip_q=(0.01, 0.99),
    valid_mask=None,
):
    """Compute KDE surfaces and halo masks for each spectral index."""
    kde_surfaces = {}
    halo_masks = {}
    for name, data in spectral_indices.items():
        kde, mask = compute_halo_detection(
            data,
            presence_quantile=presence_quantile,
            sigma_px=sigma_px,
            clip_q=clip_q,
            valid_mask=valid_mask,
        )
        kde_surfaces[name] = kde
        halo_masks[name] = mask

    return kde_surfaces, halo_masks


def plot_halo_detection_results(kde_surfaces, halo_masks, cols=3, figsize_per=(5, 5)):
    """Plot KDE surfaces with halo contours."""
    n_items = len(kde_surfaces)
    rows = int(np.ceil(n_items / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, kde) in zip(axes, kde_surfaces.items()):
        ax.imshow(kde, cmap="viridis", origin="upper")
        ax.contour(halo_masks[name], levels=[0.5], colors="red", linewidths=2)
        ax.set_title(f"{name} - High Density Halo")
        ax.axis("off")

    for ax in axes[len(kde_surfaces) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig, axes


def classify_alteration_types(
    kde_surfaces, alteration_weights, valid_mask=None, confidence_percentile=20
):
    """Classify alteration types using weighted KDE surfaces."""
    raster_shape = next(iter(kde_surfaces.values())).shape
    kde_normalized = {name: normalize_kde(kde) for name, kde in kde_surfaces.items()}

    alteration_scores = {}
    valid_classes = []
    for alt_type, weights in alteration_weights.items():
        score = np.zeros(raster_shape)
        used = 0
        for index_name, weight in weights.items():
            if index_name in kde_normalized:
                score += weight * kde_normalized[index_name]
                used += 1
        if used > 0:
            alteration_scores[alt_type] = normalize_kde(score)
            valid_classes.append(alt_type)

    if not valid_classes:
        raise ValueError("No matching spectral indices found for alteration weights.")

    all_scores = np.stack([alteration_scores[k] for k in valid_classes], axis=-1)
    class_map = np.argmax(all_scores, axis=-1) + 1

    max_scores = np.max(all_scores, axis=-1)
    finite_mask = np.isfinite(max_scores)
    if np.any(finite_mask):
        confidence_threshold = np.nanpercentile(
            max_scores[finite_mask], confidence_percentile
        )
    else:
        confidence_threshold = 0

    class_map[max_scores < confidence_threshold] = 0
    if valid_mask is not None:
        class_map[~valid_mask] = 0

    class_names = ["Background"] + valid_classes
    return class_map, class_names, alteration_scores


def run_spectral_halo_workflow(
    spectral_dir,
    alteration_weights=None,
    presence_quantile=0.9,
    sigma_px=50,
    clip_q=(0.01, 0.99),
    confidence_percentile=20,
    show=True,
):
    """Load spectral indices, detect halos, and classify alteration types."""
    spectral_indices, _ = load_spectral_indices_dir(spectral_dir)
    spectral_indices = map_spectral_indices(spectral_indices)
    if not spectral_indices:
        raise ValueError("No spectral indices matched canonical names.")

    first_index = next(iter(spectral_indices.values()))
    valid_mask = np.isfinite(first_index)
    print(f"Loaded {len(spectral_indices)} spectral indices")

    plot_spectral_indices_grid(spectral_indices)
    if show:
        plt.show()

    kde_surfaces, halo_masks = run_halo_detection_workflow(
        spectral_indices,
        presence_quantile=presence_quantile,
        sigma_px=sigma_px,
        clip_q=clip_q,
        valid_mask=valid_mask,
    )
    plot_halo_detection_results(kde_surfaces, halo_masks)
    if show:
        plt.show()

    if alteration_weights is None:
        alteration_weights = default_alteration_weights()

    class_map, class_names, alteration_scores = classify_alteration_types(
        kde_surfaces,
        alteration_weights,
        valid_mask=valid_mask,
        confidence_percentile=confidence_percentile,
    )

    summarize_alteration_classes(class_map, class_names)

    plot_alteration_map(class_map, class_names=class_names, figsize=(12, 10))
    if show:
        plt.show()

    print(
        """
SPECTRAL HALO CLASSIFICATION:
-----------------------------
This unsupervised approach identifies alteration types based on:
1. Spectral index values (proxy for mineralogy)
2. Spatial density patterns (KDE)
3. Weighted combinations based on expected assemblages

Validation should include:
- Field verification of predicted alteration types
- Comparison with known mineralization
- Cross-validation with other datasets (geochemistry, geophysics)
"""
    )

    return {
        "spectral_indices": spectral_indices,
        "kde_surfaces": kde_surfaces,
        "halo_masks": halo_masks,
        "alteration_scores": alteration_scores,
        "class_map": class_map,
        "class_names": class_names,
    }


def display_data_cube_viewer(ml_dir="data/ML", cube_name="DCG.nc"):
    """Interactive data cube viewer for ML workflow outputs."""
    from pathlib import Path
    import xarray as xr
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    ml_dir = require_path(ml_dir, "ml_dir", allow_dir=True)
    cube_path = require_one([Path(ml_dir) / cube_name], cube_name)

    print("Loaded data cube from:", cube_path)

    # Copy file locally if on S3 mount (required for HDF5/NetCDF random access)
    import shutil
    import tempfile

    local_cube_path = cube_path
    if "/mnt/" in str(cube_path) or "/s3/" in str(cube_path):
        local_cube_path = Path(tempfile.gettempdir()) / Path(cube_path).name
        if not local_cube_path.exists():
            print(f"Copying to local storage: {local_cube_path}")
            shutil.copy(cube_path, local_cube_path)

    try:
        cube = xr.open_dataset(local_cube_path, engine="h5netcdf")
    except Exception:
        cube = xr.open_dataset(local_cube_path, engine="netcdf4")
    if not cube.data_vars:
        raise ValueError(f"No data variables found in {cube_path}")

    options = []
    for name, array in cube.data_vars.items():
        if name == "spatial_ref":
            continue
        if array.ndim <= 2:
            options.append((name, (name, None)))
        else:
            stacked = array.stack(layer=array.dims[:-2])
            for i in range(stacked.shape[0]):
                options.append((f"{name} [layer {i + 1}]", (name, i)))

    var_dropdown = widgets.Dropdown(options=options, description="Variable")
    output = widgets.Output()

    def render(*_):
        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            name, layer = var_dropdown.value
            array = cube[name]
            if array.ndim == 1:
                ax.plot(array.values)
                ax.set_title(name)
                ax.grid(True, alpha=0.3)
            elif array.ndim == 2:
                ax.imshow(array.values, cmap="viridis")
                ax.set_title(name)
                ax.axis("off")
            else:
                stacked = array.stack(layer=array.dims[:-2])
                data = stacked.isel(layer=layer).values
                ax.imshow(data, cmap="viridis")
                ax.set_title(f"{name} [layer {layer + 1}]")
                ax.axis("off")
            plt.show()

    var_dropdown.observe(render, names="value")
    render()
    display(widgets.VBox([var_dropdown, output]))


def get_ml_artifacts(ml_dir):
    """Collect ML output artifact paths."""
    from pathlib import Path

    ml_dir = require_path(ml_dir, "ml_dir", allow_dir=True)

    model_outputs = [
        require_one([Path(ml_dir) / f"output{i}.tif"], f"output{i}.tif")
        for i in range(1, 7)
    ]
    stacked_raw_path = require_one(
        [Path(ml_dir) / "stacked_raw.tif"], "stacked_raw.tif"
    )
    stacked_result_path = require_one(
        [Path(ml_dir) / "stacked_result.tif"], "stacked_result.tif"
    )
    shap_path = require_one(
        [Path(ml_dir) / "SHAP.png", Path(ml_dir) / "SHAP.tif"], "SHAP"
    )
    roc_paths = [
        require_one(
            [Path(ml_dir) / f"ROC{i}.png", Path(ml_dir) / f"ROC{i}.tif"], f"ROC{i}"
        )
        for i in range(1, 7)
    ]
    stacked_roc_path = require_one(
        [Path(ml_dir) / "stacked_ROC.png", Path(ml_dir) / "stacked_ROC.tif"],
        "stacked_ROC",
    )

    return {
        "ml_dir": ml_dir,
        "model_outputs": model_outputs,
        "stacked_raw_path": stacked_raw_path,
        "stacked_result_path": stacked_result_path,
        "shap_path": shap_path,
        "roc_paths": roc_paths,
        "stacked_roc_path": stacked_roc_path,
    }


def load_display(path, title):
    """Load and display either a PNG or a raster."""
    if path.suffix.lower() == ".png":
        img = plt.imread(path)
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()
        return None, None
    data, extent, _ = load_raster(path)
    return data, extent


def display_model_outputs(ml_dir="data/ML"):
    """Visualize ML model outputs and stacked results."""
    artifacts = get_ml_artifacts(ml_dir)
    print("Loaded workflow artifacts from:", artifacts["ml_dir"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, path in zip(axes, artifacts["model_outputs"]):
        data, extent, _ = load_raster(path)
        plot_raster(data, ax=ax, title=path.stem, extent=extent, robust_stretch=True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, path in zip(
        axes, [artifacts["stacked_raw_path"], artifacts["stacked_result_path"]]
    ):
        data, extent, _ = load_raster(path)
        plot_raster(data, ax=ax, title=path.stem, extent=extent, robust_stretch=True)
    plt.tight_layout()
    plt.show()


def display_interpretability_outputs(ml_dir="data/ML"):
    """Visualize interpretability outputs (SHAP and ROC curves)."""
    artifacts = get_ml_artifacts(ml_dir)
    print("Loaded interpretability artifacts from:", artifacts["ml_dir"])

    shap_path = artifacts["shap_path"]
    if shap_path.suffix.lower() == ".png":
        load_display(shap_path, "SHAP (Beeswarm)")
    else:
        shap_data, shap_extent = load_display(shap_path, "SHAP (Beeswarm)")
        plot_raster(
            shap_data,
            ax=None,
            title="SHAP (Beeswarm)",
            extent=shap_extent,
            robust_stretch=True,
        )
        plt.show()

    roc_paths = artifacts["roc_paths"]
    roc_pngs = [p for p in roc_paths if p.suffix.lower() == ".png"]
    roc_tifs = [p for p in roc_paths if p.suffix.lower() != ".png"]

    if roc_tifs:
        print("Skipping non-PNG ROC files:", [p.name for p in roc_tifs])

    if not roc_pngs:
        raise ValueError("No ROC PNGs found; expected ROC1.png ... ROC6.png")

    cols = 3
    rows = int(np.ceil(len(roc_pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, roc_pngs):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem)
        ax.axis("off")
    for ax in axes[len(roc_pngs) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    stacked_roc_path = artifacts["stacked_roc_path"]
    if stacked_roc_path.suffix.lower() == ".png":
        load_display(stacked_roc_path, "stacked_ROC")
    else:
        stacked_roc_data, stacked_roc_extent = load_display(
            stacked_roc_path, "stacked_ROC"
        )
        plot_raster(
            stacked_roc_data,
            ax=None,
            title="stacked_ROC",
            extent=stacked_roc_extent,
            robust_stretch=True,
        )
        plt.show()


def display_workflow_outputs(ml_dir="data/ML"):
    """Visualize ML workflow outputs (model + interpretability)."""
    display_model_outputs(ml_dir)
    display_interpretability_outputs(ml_dir)


# =============================================================================
# Interpolation helpers
# =============================================================================


def idw_interpolation(
    sample_coords, sample_values, grid_points, power=2, n_neighbors=12
):
    """Inverse Distance Weighting interpolation."""
    from scipy.spatial import KDTree

    tree = KDTree(sample_coords)
    distances, indices = tree.query(grid_points, k=n_neighbors)
    distances = np.maximum(distances, 1e-10)
    weights = 1 / (distances**power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    interpolated = np.sum(weights_normalized * sample_values[indices], axis=1)
    return interpolated


def compute_semivariogram(coords, values, n_lags=12, max_lag=None):
    """Compute empirical semivariogram."""
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(coords, coords)
    if max_lag is None:
        max_lag = np.percentile(dist_matrix, 50)

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

    semivariance = []
    for i in range(n_lags):
        mask = (dist_matrix > lag_edges[i]) & (dist_matrix <= lag_edges[i + 1])
        if mask.sum() > 0:
            ii, jj = np.where(mask)
            sq_diff = (values[ii] - values[jj]) ** 2
            semivariance.append(0.5 * np.mean(sq_diff))
        else:
            semivariance.append(np.nan)

    return lag_centers, np.array(semivariance)


def spherical_variogram(h, nugget, sill, range_param):
    """Spherical variogram model."""
    h = np.asarray(h)
    result = np.zeros_like(h, dtype=float)

    mask = h > 0
    h_norm = h[mask] / range_param

    within_range = h_norm <= 1
    result_temp = np.zeros_like(h_norm)
    result_temp[within_range] = nugget + (sill - nugget) * (
        1.5 * h_norm[within_range] - 0.5 * h_norm[within_range] ** 3
    )
    result_temp[~within_range] = sill

    result[mask] = result_temp
    return result


def fit_variogram(lags, semivar):
    """Simple variogram fitting."""
    from scipy.optimize import curve_fit

    valid = ~np.isnan(semivar)
    lags_clean = lags[valid]
    semivar_clean = semivar[valid]

    if len(lags_clean) < 3:
        return 0, np.max(semivar_clean), np.max(lags_clean)

    nugget_init = semivar_clean[0] if semivar_clean[0] > 0 else 0
    sill_init = np.max(semivar_clean)
    range_init = np.max(lags_clean) / 2

    try:
        popt, _ = curve_fit(
            spherical_variogram,
            lags_clean,
            semivar_clean,
            p0=[nugget_init, sill_init, range_init],
            bounds=([0, 0, 1], [sill_init, sill_init * 2, np.max(lags_clean) * 2]),
            maxfev=5000,
        )
        return popt
    except Exception:
        return nugget_init, sill_init, range_init


def ordinary_kriging(
    sample_coords, sample_values, grid_points, nugget, sill, range_param, n_neighbors=12
):
    """Simple Ordinary Kriging implementation."""
    from scipy.spatial import KDTree

    tree = KDTree(sample_coords)
    predictions = np.zeros(len(grid_points))
    variances = np.zeros(len(grid_points))

    for i, point in enumerate(grid_points):
        distances, indices = tree.query(point, k=n_neighbors)

        if np.min(distances) < 1e-10:
            predictions[i] = sample_values[indices[0]]
            variances[i] = 0
            continue

        local_coords = sample_coords[indices]
        local_values = sample_values[indices]

        n = len(local_coords)
        K = np.zeros((n + 1, n + 1))

        for j in range(n):
            for k in range(n):
                dist = np.linalg.norm(local_coords[j] - local_coords[k])
                K[j, k] = sill - spherical_variogram(dist, nugget, sill, range_param)

        K[n, :n] = 1
        K[:n, n] = 1
        K[n, n] = 0

        k0 = np.zeros(n + 1)
        for j in range(n):
            dist = np.linalg.norm(local_coords[j] - point)
            k0[j] = sill - spherical_variogram(dist, nugget, sill, range_param)
        k0[n] = 1

        try:
            weights = np.linalg.solve(K, k0)
            predictions[i] = np.dot(weights[:n], local_values)
            variances[i] = sill - np.dot(weights[:n], k0[:n]) - weights[n]
        except Exception:
            w = 1 / (distances**2)
            predictions[i] = np.sum(w * local_values) / np.sum(w)
            variances[i] = np.var(local_values)

    return predictions, np.maximum(variances, 0)


# =============================================================================
# Spectral halo helpers
# =============================================================================


def generate_spectral_index(
    shape, n_anomalies=3, background=0.3, anomaly_strength=0.4, random_state=42
):
    """Generate synthetic spectral index with anomalous regions."""
    rng = np.random.default_rng(random_state)
    data = rng.normal(background, 0.1, shape)

    for _ in range(n_anomalies):
        cx, cy = rng.integers(20, 80, 2)
        radius = rng.integers(10, 25)
        y, x = np.ogrid[: shape[0], : shape[1]]
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < radius**2
        data[mask] += rng.uniform(anomaly_strength * 0.5, anomaly_strength)

    return np.clip(data, 0, 1)


def compute_halo_detection(
    index_data, presence_quantile=0.9, sigma_px=5, clip_q=(0.01, 0.99), valid_mask=None
):
    """Detect high-density halos using KDE and K-means."""
    from sklearn.cluster import KMeans
    from scipy.ndimage import gaussian_filter

    data = np.array(index_data, dtype=float)
    data[~np.isfinite(data)] = np.nan

    if np.all(np.isnan(data)):
        shape = data.shape
        return np.zeros(shape), np.zeros(shape, dtype=bool)

    lo, hi = np.nanquantile(data, clip_q)
    clipped = np.clip(data, lo, hi)
    median_val = np.nanmedian(clipped)
    iqr = np.nanpercentile(clipped, 75) - np.nanpercentile(clipped, 25)
    if not np.isfinite(iqr) or iqr == 0:
        std = np.nanstd(clipped)
        iqr = std if np.isfinite(std) and std > 0 else 1.0
    z_scored = (clipped - median_val) / iqr
    z_scored = np.where(np.isfinite(z_scored), z_scored, median_val)

    threshold = np.nanquantile(z_scored, presence_quantile)
    presence_mask = z_scored >= threshold
    if valid_mask is not None:
        presence_mask = presence_mask & valid_mask

    kde_surface = gaussian_filter(
        presence_mask.astype(float), sigma=sigma_px, mode="nearest"
    )

    kde_flat = kde_surface.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(kde_flat).reshape(index_data.shape)

    cluster_means = [kde_surface[cluster_labels == i].mean() for i in range(2)]
    high_density_label = np.argmax(cluster_means)
    high_density_mask = cluster_labels == high_density_label

    return kde_surface, high_density_mask


def normalize_kde(kde, percentile_low=2, percentile_high=98):
    """Robust normalization of KDE surface."""
    data = np.array(kde, dtype=float)
    data[~np.isfinite(data)] = np.nan
    if np.all(np.isnan(data)):
        return np.zeros_like(data)

    p_low, p_high = np.nanpercentile(data, [percentile_low, percentile_high])
    normalized = (data - p_low) / (p_high - p_low + 1e-10)
    return np.clip(normalized, 0, 1)


def map_spectral_indices(spectral_indices):
    """Map raw spectral index names to canonical names for weighting."""

    def canonical(name):
        key = name.lower()
        if "clay" in key and ("hydrox" in key or "aloh" in key):
            return "Clay_AlOH"
        if "iron_oxide" in key or "ferric" in key or "fe3" in key:
            return "Iron_Oxide"
        if "ferrous" in key or "fe2" in key:
            return "Ferrous_Iron"
        if "silica" in key or "quartz" in key:
            return "Silica"
        if "gossan" in key:
            return "Gossan"
        if "laterite" in key:
            return "Laterite"
        if "sabins" in key or "hydrothermal" in key or "alteration" in key:
            return "Alt_Composite"
        return None

    grouped = {}
    for name, data in spectral_indices.items():
        canon = canonical(name)
        if canon is None:
            continue
        grouped.setdefault(canon, []).append(np.array(data, dtype=float))

    mapped = {}
    for canon, arrays in grouped.items():
        stacked = np.stack(arrays, axis=0)
        finite_mask = np.isfinite(stacked)
        counts = finite_mask.sum(axis=0)
        summed = np.nansum(stacked, axis=0)
        mapped[canon] = np.divide(
            summed, counts, out=np.zeros_like(summed, dtype=float), where=counts > 0
        )

    return mapped


# =============================================================================
# ML Workflow helpers
# =============================================================================


def prepare_ml_labels(geochem_gdf, targets_gdf, radius_m=500):
    """
    Create binary labels for supervised ML from known occurrence locations.

    Parameters
    ----------
    geochem_gdf : geopandas.GeoDataFrame
        Geochemical sample points.  Must have a geometry column.
    targets_gdf : geopandas.GeoDataFrame
        Known mineral occurrence / deposit points.
    radius_m : float
        Any geochem sample within this distance (meters) of a deposit is
        labeled positive (1).  All others are background (0).

    Returns
    -------
    y_labels : np.ndarray, shape (n_samples,)
        Binary integer array aligned with geochem_gdf row order.

    Notes
    -----
    The current project assumes all notebook data have already been projected
    into the fixed analysis CRS, so distance is measured directly in meters
    using Euclidean nearest-neighbor search.
    """
    from scipy.spatial import cKDTree

    geochem_xy = np.column_stack([geochem_gdf.geometry.x, geochem_gdf.geometry.y])
    targets_xy = np.column_stack([targets_gdf.geometry.x, targets_gdf.geometry.y])

    tree = cKDTree(targets_xy)
    dist, _ = tree.query(geochem_xy, k=1)
    y_labels = (dist <= radius_m).astype(int)

    n_pos = int(y_labels.sum())
    n_neg = int(len(y_labels) - n_pos)
    print(
        f"Positive samples (within {radius_m}m of deposit): {n_pos} | Background: {n_neg}"
    )

    return y_labels


def add_lithology_features(geochem_gdf, lith_gdf, col="lithology_family"):
    """
    Assign a lithology class to each geochem sample via spatial join and
    return one-hot encoded columns aligned to geochem_gdf.

    Parameters
    ----------
    geochem_gdf : GeoDataFrame - sample point locations
    lith_gdf    : GeoDataFrame - lithology polygons
    col         : str          - column in lith_gdf to encode (default: "lithology_family")

    Returns
    -------
    dummies : pd.DataFrame, shape (n_samples, n_classes)
        One-hot encoded lithology columns; rows with no polygon match are all
        zeros.

    Notes
    -----
    Each output column is a binary indicator for one lithology class. This lets
    lithology be concatenated with raster and geochemistry predictors before
    running a machine-learning model.
    """
    import geopandas as gpd

    joined = gpd.sjoin(
        geochem_gdf[["geometry"]],
        lith_gdf[[col, "geometry"]],
        how="left",
        predicate="within",
    )
    # Keep first match per point (index_right may duplicate on overlapping polygons)
    joined = joined[~joined.index.duplicated(keep="first")]

    dummies = pd.get_dummies(joined[col], prefix="lith", dtype=float).reindex(
        geochem_gdf.index, fill_value=0.0
    )
    print(
        f"Lithology feature columns ({len(dummies.columns)}): {list(dummies.columns)}"
    )
    return dummies


def extract_raster_values(geochem_gdf, *raster_dirs):
    """
    Sample raster values at every geochemistry point.

    Parameters
    ----------
    geochem_gdf : geopandas.GeoDataFrame
        Point locations to sample at.
    *raster_dirs : Path or str
        One or more directories containing .tif files.  Each directory's stem
        is used as a prefix for the layer names (e.g. a dir named ``spectral``
        produces names like ``spectral_idx_clay_hydroxyls``).

    Returns
    -------
    X_raw : np.ndarray, shape (n_samples, n_rasters)
        Sampled values; NaN where a point falls outside a raster's extent or
        on a nodata cell.
    predictor_names : list[str]
        One name per column of ``X_raw``.

    Notes
    -----
    This helper is the bridge between map-based raster layers and
    sample-by-sample machine-learning tables. Each raster becomes one predictor
    column, and each geochemistry point becomes one row.
    """
    import rasterio
    from pathlib import Path

    layers = {}
    for raster_dir in raster_dirs:
        raster_dir = Path(raster_dir)
        prefix = raster_dir.stem
        tif_paths = sorted(raster_dir.glob("*.tif"))
        if not tif_paths:
            raise FileNotFoundError(f"No .tif files found in {raster_dir}")
        for path in tif_paths:
            with rasterio.open(path) as src:
                xs = geochem_gdf.geometry.x.values
                ys = geochem_gdf.geometry.y.values
                coords = list(zip(xs, ys))
                vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
                if src.nodata is not None:
                    vals[vals == src.nodata] = np.nan
            layers[f"{prefix}_{path.stem}"] = vals

    predictor_names = list(layers.keys())
    X_raw = np.column_stack(list(layers.values()))
    print(f"Raster layers extracted: {len(predictor_names)}")
    return X_raw, predictor_names


def spatial_checkerboard_split(gdf, y=None, cell_size_m=5000, random_state=42):
    """
    Split point data into train/test using a checkerboard in map space.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Point geometries to split.
    y : array-like, optional
        Binary labels aligned with ``gdf``. When provided, the function chooses
        the checkerboard offset/parity that preserves at least one positive in
        both train and test when possible.
    cell_size_m : float
        Checkerboard cell size in meters.
    random_state : int
        Reserved for API stability. Not currently used.

    Returns
    -------
    split : dict
        Train/test masks plus checkerboard metadata.

    Notes
    -----
    Nearby samples tend to be similar, so a random split can make model
    performance look better than it really is. This helper uses alternating map
    cells to keep neighboring samples together and produce a stricter spatial
    evaluation.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")

    metric_crs = ANALYSIS_CRS
    metric_gdf = gdf
    xs = metric_gdf.geometry.x.to_numpy()
    ys = metric_gdf.geometry.y.to_numpy()
    x_min = xs.min()
    y_min = ys.min()

    if y is not None:
        y = np.asarray(y)
        if len(y) != len(gdf):
            raise ValueError("y must have the same length as gdf.")

    candidates = []
    half = cell_size_m / 2.0
    offsets = [(0.0, 0.0), (half, 0.0), (0.0, half), (half, half)]

    for x_offset_m, y_offset_m in offsets:
        x_idx = np.floor((xs - x_min + x_offset_m) / cell_size_m).astype(int)
        y_idx = np.floor((ys - y_min + y_offset_m) / cell_size_m).astype(int)
        checker_id = x_idx + y_idx
        checkerboard = checker_id % 2

        for test_parity in (0, 1):
            test_mask = checkerboard == test_parity
            train_mask = ~test_mask
            if not train_mask.any() or not test_mask.any():
                continue

            candidate = {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "checkerboard": checkerboard,
                "x_index": x_idx,
                "y_index": y_idx,
                "metric_crs": metric_crs,
                "cell_size_m": float(cell_size_m),
                "x_offset_m": float(x_offset_m),
                "y_offset_m": float(y_offset_m),
                "test_parity": int(test_parity),
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
            }

            if y is not None:
                train_pos = int(y[train_mask].sum())
                test_pos = int(y[test_mask].sum())
                train_neg = int(train_mask.sum() - train_pos)
                test_neg = int(test_mask.sum() - test_pos)
                candidate.update(
                    {
                        "train_pos": train_pos,
                        "test_pos": test_pos,
                        "train_neg": train_neg,
                        "test_neg": test_neg,
                        "valid_class_split": (
                            train_pos > 0
                            and test_pos > 0
                            and train_neg > 0
                            and test_neg > 0
                        ),
                    }
                )

            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "Could not construct a checkerboard split from the provided points."
        )

    if y is None:
        best = min(candidates, key=lambda c: abs(c["test_size"] - c["train_size"]))
    else:
        valid = [c for c in candidates if c["valid_class_split"]]
        pool = valid if valid else candidates
        best = min(
            pool,
            key=lambda c: (
                0 if c.get("test_pos", 0) > 0 else 1,
                c.get("test_pos", 0) if c.get("test_pos", 0) > 0 else np.inf,
                abs(c["test_size"] - c["train_size"]),
            ),
        )

    return best


# =============================================================================
# Visualization helpers (wrapping boilerplate for notebook clarity)
# =============================================================================


def plot_data_overview(
    geochem_gdf,
    lith_gdf,
    tgt_gdf,
    figsize=(10, 8),
    geochem_color_col=None,
    geochem_cmap="viridis",
    spectral_dir=None,
    geophys_dir=None,
    raster_ncols=4,
):
    """
    Plot the main study-area overview map used early in the notebook.

    The map combines lithology polygons, geochemistry sample locations, and
    known mineral occurrences in one figure so students can orient themselves
    before moving into PCA, clustering, or supervised learning. When
    ``geochem_color_col`` is provided, the sample points are colored by that
    numeric column and a colorbar is added automatically. When raster
    directories are provided, a second figure previews the raster stack.
    """
    from matplotlib.lines import Line2D
    import rasterio
    from pathlib import Path

    fig, ax = plt.subplots(figsize=figsize)
    lith_units = pd.Index(lith_gdf["lithology_family"].dropna().unique())
    if len(lith_units) <= len(CATEGORICAL_COLORS):
        unit_colors = CATEGORICAL_COLORS[: len(lith_units)]
    else:
        unit_colors = [
            mcolors.to_hex(plt.get_cmap("tab20", len(lith_units))(i))
            for i in range(len(lith_units))
        ]
    lith_cmap = mcolors.ListedColormap(unit_colors)

    plot_vector(
        lith_gdf,
        column="lithology_family",
        categorical=True,
        ax=ax,
        title="Sample Locations and Known Deposits",
        alpha=0.3,
        edgecolor="gray",
        linewidth=0.4,
        legend=False,
        categorical_cmap=lith_cmap,
    )

    if geochem_color_col is None:
        geochem_gdf.plot(
            ax=ax, color="steelblue", markersize=15, alpha=0.7, label="Geochem samples"
        )
    else:
        if geochem_color_col not in geochem_gdf.columns:
            raise ValueError(f"Column not found in geochem data: {geochem_color_col}")
        values = geochem_gdf[geochem_color_col]
        sc = ax.scatter(
            geochem_gdf.geometry.x,
            geochem_gdf.geometry.y,
            c=values,
            cmap=geochem_cmap,
            s=18,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(geochem_color_col)

    tgt_gdf.plot(
        ax=ax,
        marker="*",
        color="gold",
        markersize=180,
        edgecolor="black",
        linewidth=0.8,
        label="Known deposits",
        zorder=5,
    )

    sample_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="steelblue",
            markeredgecolor="white",
            markersize=7,
            label="Geochem samples",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=14,
            label="Known deposits",
        ),
    ]
    sample_legend = ax.legend(handles=sample_handles, loc="upper right", title="Legend")
    ax.add_artist(sample_legend)

    lith_handles = [
        Patch(facecolor=color, edgecolor="gray", label=unit)
        for unit, color in zip(lith_units, unit_colors)
    ]
    ax.legend(
        handles=lith_handles,
        loc="lower left",
        title="Lithology units",
        fontsize=8,
        title_fontsize=9,
    )
    plt.tight_layout()

    if spectral_dir is None and geophys_dir is None:
        return fig, ax

    spectral_paths = []
    geophys_paths = []
    if spectral_dir is not None:
        spectral_paths = sorted(Path(spectral_dir).glob("*.tif"))
    if geophys_dir is not None:
        geophys_paths = sorted(Path(geophys_dir).glob("*.tif"))
    all_raster_paths = spectral_paths + geophys_paths

    ncols = max(1, raster_ncols)
    nrows = -(-len(all_raster_paths) // ncols)
    fig_rasters, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, path in enumerate(all_raster_paths):
        ax_r = axes_flat[i]
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            bounds = src.bounds
            extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        cmap = "RdBu_r" if path in geophys_paths else "YlOrBr"
        vmin, vmax = np.nanpercentile(data, [2, 98])
        ax_r.imshow(
            data,
            extent=extent,
            origin="upper",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        label = ("mag " if path in geophys_paths else "spec ") + path.stem.replace(
            "idx_", ""
        ).replace("_", " ").replace("AMF", "")
        ax_r.set_title(label.strip(), fontsize=8, pad=3)
        ax_r.set_xticks([])
        ax_r.set_yticks([])

    for j in range(len(all_raster_paths), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig_rasters.suptitle(
        f"Raster Data Preview - {len(all_raster_paths)} layers", fontsize=11, y=1.01
    )
    fig_rasters.tight_layout()
    return fig, ax, fig_rasters


def plot_feature_overview(
    geochem_gdf,
    spectral_dir,
    geophys_dir,
    lith_gdf,
    y,
    radius_m,
    tgt_gdf,
    geochem_X,
    X_raw,
    predictor_names,
    feature_cols=None,
    preview_elements=(
        "Cu_ppm_icp",
        "Mo_ppm_icp",
        "Au_ppb_icp",
        "As_ppm_icp",
        "Sb_ppm_icp",
        "W_ppm_icp",
    ),
):
    """
    Summarize the predictor stack used in the supervised-learning section.

    A single grid is created. The first panel shows the binary training labels
    at the sample locations, followed by sampled lithology, several sampled
    geochemistry maps, and the sampled raster predictors as colored points.

    This gives students a quick visual check of what information the model is
    learning from before the train/test split and Random Forest steps.
    """
    if feature_cols is None:
        feature_cols = get_geochem_columns(geochem_gdf)
    feature_cols = list(feature_cols)
    preview_geochem_cols = [e for e in preview_elements if e in feature_cols]
    if len(preview_geochem_cols) < 6:
        extras = [c for c in feature_cols if c not in preview_geochem_cols]
        preview_geochem_cols.extend(extras[: 6 - len(preview_geochem_cols)])
    xs = geochem_gdf.geometry.x.values
    ys = geochem_gdf.geometry.y.values

    lith_col = choose_lithology_column(lith_gdf)
    sampled_lith = None
    if lith_col is not None:
        joined = gpd.sjoin(
            geochem_gdf[["geometry"]],
            lith_gdf[[lith_col, "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined[~joined.index.duplicated(keep="first")].reindex(
            geochem_gdf.index
        )
        sampled_lith = joined[lith_col].fillna("No match").astype(str)

    panel_specs = [("labels", "training labels", y)]
    if sampled_lith is not None:
        panel_specs.append(("lith", lith_col.replace("_", " "), sampled_lith))
    for geochem_col in preview_geochem_cols:
        geochem_idx = feature_cols.index(geochem_col)
        panel_specs.append(("geochem", geochem_col, geochem_X[:, geochem_idx]))
    panel_specs.extend(
        ("raster", name, X_raw[:, i]) for i, name in enumerate(predictor_names)
    )

    n_panels = len(panel_specs)
    ncols = 5
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(3.4, nrows * 3.0)))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (kind, name, values) in enumerate(panel_specs):
        ax = axes_flat[i]
        plot_vector(lith_gdf, ax=ax, alpha=0.08, edgecolor="gray", linewidth=0.2)

        if kind == "labels":
            neg = y == 0
            pos = y == 1
            ax.scatter(
                xs[neg], ys[neg], color="steelblue", s=10, linewidths=0, alpha=0.55
            )
            ax.scatter(xs[pos], ys[pos], color="red", s=16, linewidths=0, alpha=0.85)
            ax.set_title(f"{name} ({radius_m // 1000} km)", fontsize=7, pad=2)
        elif kind == "lith":
            categories = pd.Index(pd.unique(values))
            colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
            color_lookup = {
                category: colors[j] for j, category in enumerate(categories)
            }
            point_colors = [color_lookup[val] for val in values]
            ax.scatter(xs, ys, c=point_colors, s=12, linewidths=0, alpha=0.9)
            ax.set_title(f"lithology: {name}", fontsize=7, pad=2)
        else:
            cmap = (
                "plasma"
                if kind == "geochem"
                else ("RdBu_r" if name.startswith("mag ") else "YlOrBr")
            )
            sc = ax.scatter(xs, ys, c=values, cmap=cmap, s=12, linewidths=0, alpha=0.9)
            plt.colorbar(sc, ax=ax, shrink=0.75)
            title = f"geochem: {name}" if kind == "geochem" else name
            ax.set_title(title, fontsize=7, pad=2)

        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Supervised ML Inputs at Sample Points",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    return fig, axes


def get_feature_importance(rf, feature_names):
    """Return a DataFrame of feature importances sorted descending."""
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_probability_map(
    gdf_valid, y_valid, y_prob_all, lith_gdf, tgt_gdf, radius_m, figsize=(16, 7)
):
    """Map the model's predicted probability at each valid sample location."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_vector(lith_gdf, ax=ax, alpha=0.2, edgecolor="gray", linewidth=0.4)
    sc = ax.scatter(
        gdf_valid.geometry.x,
        gdf_valid.geometry.y,
        c=y_prob_all,
        cmap="plasma",
        vmin=0,
        vmax=1,
        s=30,
        alpha=0.85,
        zorder=3,
    )
    tgt_gdf.plot(
        ax=ax,
        marker="*",
        color="white",
        markersize=150,
        edgecolor="black",
        linewidth=0.8,
        label="Known deposits",
        zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="Predicted probability", shrink=0.7)
    ax.set_title("Predicted Probability (all samples)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig, ax


def plot_spatial_split(gdf_valid, split, lith_gdf, tgt_gdf=None, figsize=(8, 5)):
    """
    Map showing which samples landed in the train vs test set after a spatial
    checkerboard split, with faint grid lines showing the cell boundaries.
    """
    from pyproj import Transformer

    train_mask = split["train_mask"]
    test_mask = split["test_mask"]
    cell_size = split["cell_size_m"]
    cell_km = cell_size / 1000
    metric_crs = split["metric_crs"]
    x_off = split["x_offset_m"]
    y_off = split["y_offset_m"]
    display_crs = gdf_valid.crs

    fig, ax = plt.subplots(figsize=figsize)
    plot_vector(lith_gdf, ax=ax, alpha=0.15, edgecolor="gray", linewidth=0.3)

    # ── Checkerboard grid lines ────────────────────────────────────────────
    # The split anchors the grid at (xs.min() - x_offset, ys.min() - y_offset)
    # in projected coords, so we must recompute the same origin here.
    metric_gdf = gdf_valid.to_crs(metric_crs)
    xs_m = metric_gdf.geometry.x.values
    ys_m = metric_gdf.geometry.y.values

    x_origin = xs_m.min() - x_off
    y_origin = ys_m.min() - y_off

    # Grid line positions covering the data extent plus one cell of padding
    def grid_lines(origin, values, size):
        k0 = int(np.floor((values.min() - origin) / size)) - 1
        k1 = int(np.ceil((values.max() - origin) / size)) + 1
        return [origin + k * size for k in range(k0, k1 + 1)]

    x_lines = grid_lines(x_origin, xs_m, cell_size)
    y_lines = grid_lines(y_origin, ys_m, cell_size)

    t = Transformer.from_crs(metric_crs, display_crs, always_xy=True)
    y_lo, y_hi = y_lines[0], y_lines[-1]
    x_lo, x_hi = x_lines[0], x_lines[-1]

    # Add intermediate points so lines stay accurate after reprojection
    n_seg = 10
    for x in x_lines:
        ys_line = np.linspace(y_lo, y_hi, n_seg)
        lons, lats = t.transform(np.full(n_seg, x), ys_line)
        ax.plot(lons, lats, color="dimgray", linewidth=0.4, alpha=0.35, zorder=2)

    for y in y_lines:
        xs_line = np.linspace(x_lo, x_hi, n_seg)
        lons, lats = t.transform(xs_line, np.full(n_seg, y))
        ax.plot(lons, lats, color="dimgray", linewidth=0.4, alpha=0.35, zorder=2)

    # ── Sample points ──────────────────────────────────────────────────────
    gdf_valid[train_mask].plot(
        ax=ax,
        color="steelblue",
        markersize=12,
        alpha=0.7,
        label=f"Train  (n={train_mask.sum()})",
        zorder=3,
    )
    gdf_valid[test_mask].plot(
        ax=ax,
        color="darkorange",
        markersize=12,
        alpha=0.7,
        label=f"Test   (n={test_mask.sum()})",
        zorder=3,
    )

    if tgt_gdf is not None:
        tgt_gdf.plot(
            ax=ax,
            marker="*",
            color="gold",
            markersize=150,
            edgecolor="black",
            linewidth=0.8,
            label="Known deposits",
            zorder=5,
        )

    # Clamp view to the actual data extent (grid lines extend beyond it)
    gx = gdf_valid.geometry.x
    gy = gdf_valid.geometry.y
    margin_x = (gx.max() - gx.min()) * 0.02
    margin_y = (gy.max() - gy.min()) * 0.02
    ax.set_xlim(gx.min() - margin_x, gx.max() + margin_x)
    ax.set_ylim(gy.min() - margin_y, gy.max() + margin_y)

    ax.set_title(
        f"Spatial Checkerboard Split - {cell_km:.0f} km cells\n"
        f"(adjacent cells alternate train/test to reduce spatial leakage)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax
