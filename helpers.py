"""Helper functions used by the geospatial ML training notebook."""

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

__all__ = [
    "ANALYSIS_CRS",
    "add_lithology_features",
    "apply_transform",
    "choose_lithology_column",
    "clip_quantiles",
    "compute_cluster_centroids",
    "extract_raster_values",
    "get_feature_importance",
    "get_geochem_columns",
    "log_transform",
    "plot_clusters_on_lithology",
    "plot_data_overview",
    "plot_elbow_silhouette",
    "plot_feature_importance",
    "plot_feature_overview",
    "plot_kmeans_pca_scatter",
    "plot_pca_loadings",
    "plot_pca_variance",
    "plot_probability_map",
    "plot_roc_pr_curves",
    "plot_spatial_pca_components",
    "plot_spatial_split",
    "plot_vector",
    "prepare_ml_labels",
    "prepare_pca_inputs",
    "spatial_checkerboard_split",
]


def _categorical_cmap(n_categories):
    """Return a categorical colormap sized for the requested count."""
    if n_categories <= len(CATEGORICAL_COLORS):
        return mcolors.ListedColormap(CATEGORICAL_COLORS[:n_categories])
    return mcolors.ListedColormap(plt.get_cmap("tab20", n_categories).colors)


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
    """Plot a GeoDataFrame with optional column coloring."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(title)
    if gdf.empty:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return ax

    plot_kwargs = {
        "ax": ax,
        "legend": legend if column is not None else False,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "alpha": alpha,
    }
    if "Point" in gdf.geometry.iloc[0].geom_type:
        plot_kwargs["markersize"] = markersize

    if column is None:
        gdf.plot(color="steelblue", **plot_kwargs)
    elif categorical:
        gdf.plot(
            column=column,
            cmap=categorical_cmap or _categorical_cmap(gdf[column].nunique()),
            **plot_kwargs,
        )
    else:
        gdf.plot(column=column, cmap=cmap, **plot_kwargs)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax


def compute_cluster_centroids(values, labels):
    """Compute centroid coordinates for each cluster label."""
    return np.vstack([values[labels == label].mean(axis=0) for label in np.unique(labels)])


def plot_kmeans_pca_scatter(
    X_pca, labels, title="K-means in PCA Space", ax=None, markersize=40
):
    """Plot PC1 vs PC2 with cluster centroids."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    unique_labels = np.unique(labels)
    colors = _categorical_cmap(len(unique_labels)).colors
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[color],
            label=f"Cluster {label}",
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


def plot_elbow_silhouette(k_range, inertias, silhouettes, figsize=(14, 5)):
    """Plot elbow and silhouette curves for K-means."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, axes


def plot_roc_pr_curves(y_true, y_prob, figsize=(14, 5)):
    """Plot ROC and precision-recall curves."""
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

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

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)
    axes[1].plot(
        recall,
        precision,
        color="steelblue",
        linewidth=2,
        label=f"PR (AP = {average_precision:.3f})",
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
    """Plot the top feature importances."""
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


def get_geochem_columns(geochem_gdf):
    """Return all non-geometry geochemistry columns."""
    return [column for column in geochem_gdf.columns if column != "geometry"]


def log_transform(values):
    """Apply a stable log1p transform after shifting values non-negative."""
    values = np.asarray(values, dtype=float)
    shift = np.nanmin(values, axis=0) if values.ndim == 2 else np.nanmin(values)
    return np.log1p(values - shift + 1)


def apply_transform(values, transform="log1p"):
    """Apply a named transform to an array."""
    if transform in (None, "none", "identity"):
        return np.asarray(values, dtype=float)
    if transform == "log1p":
        return log_transform(values)
    raise ValueError(f"Unsupported transform: {transform}")


def clip_quantiles(values, quantiles=None):
    """Clip values to a quantile range."""
    values = np.asarray(values, dtype=float)
    if quantiles is None:
        return values
    if not isinstance(quantiles, (tuple, list)) or len(quantiles) != 2:
        raise ValueError("quantiles must be a (low, high) tuple")

    q_low, q_high = quantiles
    if values.ndim == 2:
        return np.clip(
            values,
            np.nanquantile(values, q_low, axis=0),
            np.nanquantile(values, q_high, axis=0),
        )
    return np.clip(values, np.nanquantile(values, q_low), np.nanquantile(values, q_high))


def prepare_pca_inputs(
    geochem_gdf,
    feature_cols=None,
    exclude_cols=None,
    transform="log1p",
    scale_features=True,
    clip_quantiles_range=None,
):
    """Prepare the numeric matrix used as PCA input."""
    from sklearn.preprocessing import StandardScaler

    if feature_cols is None:
        feature_cols = get_geochem_columns(geochem_gdf)

    pca_cols = [c for c in feature_cols if c not in {"id", "elevation_m"}]
    if exclude_cols:
        excluded = set(exclude_cols)
        pca_cols = [c for c in pca_cols if c not in excluded]

    X_geochem = clip_quantiles(
        geochem_gdf[pca_cols].to_numpy(),
        clip_quantiles_range,
    )
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
    for threshold, color in zip([0.5, 0.75, 0.9], ["#8e6c8a", "#5b8c5a", "#6c757d"]):
        index = np.argmax(cum_var >= threshold) + 1
        ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.6)
        ax.axvline(x=index, color=color, linestyle=":", alpha=0.6)
        ax.text(
            index + 0.2,
            threshold + 0.02,
            f"{int(threshold * 100)}% -> {index} PCs",
            color=color,
            fontsize=9,
        )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.set_xticks(range(1, n_components + 1, max(1, n_components // 12)))
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
        ax.barh(top.index, top.values, color=["#d95f02" if value < 0 else "#1b9e77" for value in top.values])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(
            f"PC{i + 1} Top +{top_n_pos} / -{top_n_neg} Loadings "
            f"({pca.explained_variance_ratio_[i] * 100:.1f}% variance)"
        )
        ax.set_xlabel("Loading")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return fig, axes


def plot_spatial_pca_components(
    geochem_gdf, X_pca, pca, n_components=3, cmap=DIVERGING_CMAP, figsize=(16, 5)
):
    """Plot the first few PCA components in map view."""
    fig, axes = plt.subplots(1, n_components, figsize=figsize)
    axes = np.atleast_1d(axes)

    for index, ax in enumerate(axes):
        if index >= X_pca.shape[1]:
            ax.axis("off")
            continue
        gdf_temp = geochem_gdf.copy()
        column = f"PC{index + 1}"
        gdf_temp[column] = X_pca[:, index]
        gdf_temp.plot(column=column, ax=ax, legend=True, cmap=cmap, markersize=30)
        ax.set_title(
            f"{column} ({pca.explained_variance_ratio_[index] * 100:.1f}% variance)"
        )

    plt.tight_layout()
    return fig, axes


def choose_lithology_column(vector_gdf, candidates=None):
    """Pick a representative lithology column if available."""
    if candidates is None:
        candidates = [
            "lithology_family",
            "main_lithology",
            "geological_era",
            "tectonic_setting",
        ]
    return next((column for column in candidates if column in vector_gdf.columns), None)


def plot_clusters_on_lithology(
    vector_gdf, geochem_gdf, cluster_labels, lith_column=None, figsize=(10, 8)
):
    """Overlay clustered geochemistry samples on lithology polygons."""
    fig, ax = plt.subplots(figsize=figsize)

    if lith_column is None:
        lith_column = choose_lithology_column(vector_gdf)

    lith_handles = []
    if lith_column:
        lith_units = pd.Index(sorted(vector_gdf[lith_column].dropna().astype(str).unique()))
        lith_cmap = _categorical_cmap(len(lith_units))
        lith_colors = list(lith_cmap.colors)[: len(lith_units)]
        plot_vector(
            vector_gdf,
            column=lith_column,
            categorical=True,
            categorical_cmap=lith_cmap,
            ax=ax,
            title="Clusters on Lithology",
            alpha=0.6,
            edgecolor="#8a8a8a",
            linewidth=0.35,
            legend=False,
        )
        lith_handles = [
            Patch(facecolor=color, edgecolor="#8a8a8a", linewidth=0.5, label=unit)
            for unit, color in zip(lith_units, lith_colors)
        ]
    else:
        vector_gdf.plot(ax=ax, color="#ececec", edgecolor="#9aa0a6", linewidth=0.35, alpha=0.45)
        ax.set_title("Clusters on Lithology")

    gdf_clustered = geochem_gdf.copy()
    gdf_clustered["cluster"] = cluster_labels
    cluster_ids = pd.Index(pd.unique(cluster_labels))
    cluster_colors = list(_categorical_cmap(len(cluster_ids)).colors)[: len(cluster_ids)]
    gdf_clustered.plot(
        column="cluster",
        ax=ax,
        legend=False,
        categorical=True,
        cmap=mcolors.ListedColormap(cluster_colors),
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
            label=f"Cluster {cluster_id}",
        )
        for cluster_id, color in zip(cluster_ids, cluster_colors)
    ]
    cluster_legend = ax.legend(handles=cluster_handles, title="Cluster Legend", loc="lower left")
    ax.add_artist(cluster_legend)

    if lith_handles:
        ax.legend(
            handles=lith_handles,
            title=lith_column.replace("_", " ").title(),
            loc="upper right",
            fontsize=8,
            title_fontsize=9,
        )

    plt.tight_layout()
    return fig, ax


def prepare_ml_labels(geochem_gdf, targets_gdf, radius_m=500):
    """Label samples as positive when they fall within a target radius."""
    from scipy.spatial import cKDTree

    geochem_xy = np.column_stack([geochem_gdf.geometry.x, geochem_gdf.geometry.y])
    targets_xy = np.column_stack([targets_gdf.geometry.x, targets_gdf.geometry.y])
    distances, _ = cKDTree(targets_xy).query(geochem_xy, k=1)
    y_labels = (distances <= radius_m).astype(int)

    n_pos = int(y_labels.sum())
    print(
        f"Positive samples (within {radius_m}m of deposit): {n_pos} | "
        f"Background: {len(y_labels) - n_pos}"
    )
    return y_labels


def add_lithology_features(geochem_gdf, lith_gdf, col="lithology_family"):
    """One-hot encode the lithology intersecting each geochemistry sample."""
    joined = gpd.sjoin(
        geochem_gdf[["geometry"]],
        lith_gdf[[col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    dummies = pd.get_dummies(joined[col], prefix="lith", dtype=float).reindex(
        geochem_gdf.index,
        fill_value=0.0,
    )
    print(f"Lithology feature columns ({len(dummies.columns)}): {list(dummies.columns)}")
    return dummies


def extract_raster_values(geochem_gdf, *raster_dirs):
    """Sample raster values at every geochemistry point."""
    import rasterio
    from pathlib import Path

    coords = list(zip(geochem_gdf.geometry.x.to_numpy(), geochem_gdf.geometry.y.to_numpy()))
    layers = {}
    for raster_dir in raster_dirs:
        raster_dir = Path(raster_dir)
        tif_paths = sorted(raster_dir.glob("*.tif"))
        if not tif_paths:
            raise FileNotFoundError(f"No .tif files found in {raster_dir}")

        prefix = raster_dir.stem
        for path in tif_paths:
            with rasterio.open(path) as src:
                values = np.array([sample[0] for sample in src.sample(coords)], dtype=float)
                if src.nodata is not None:
                    values[values == src.nodata] = np.nan
            layers[f"{prefix}_{path.stem}"] = values

    predictor_names = list(layers)
    X_raw = np.column_stack([layers[name] for name in predictor_names])
    print(f"Raster layers extracted: {len(predictor_names)}")
    return X_raw, predictor_names


def spatial_checkerboard_split(gdf, y=None, cell_size_m=5000):
    """Split point data into train and test using a map-space checkerboard."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")

    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    x_min = xs.min()
    y_min = ys.min()

    if y is not None:
        y = np.asarray(y)
        if len(y) != len(gdf):
            raise ValueError("y must have the same length as gdf.")

    candidates = []
    half_cell = cell_size_m / 2.0
    for x_offset_m, y_offset_m in [(0.0, 0.0), (half_cell, 0.0), (0.0, half_cell), (half_cell, half_cell)]:
        x_index = np.floor((xs - x_min + x_offset_m) / cell_size_m).astype(int)
        y_index = np.floor((ys - y_min + y_offset_m) / cell_size_m).astype(int)
        checkerboard = (x_index + y_index) % 2

        for test_parity in (0, 1):
            test_mask = checkerboard == test_parity
            train_mask = ~test_mask
            if not train_mask.any() or not test_mask.any():
                continue

            candidate = {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "checkerboard": checkerboard,
                "x_index": x_index,
                "y_index": y_index,
                "metric_crs": ANALYSIS_CRS,
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
                candidate.update(
                    {
                        "train_pos": train_pos,
                        "test_pos": test_pos,
                        "train_neg": int(train_mask.sum() - train_pos),
                        "test_neg": int(test_mask.sum() - test_pos),
                        "valid_class_split": (
                            train_pos > 0
                            and test_pos > 0
                            and train_pos < train_mask.sum()
                            and test_pos < test_mask.sum()
                        ),
                    }
                )
            candidates.append(candidate)

    if not candidates:
        raise ValueError("Could not construct a checkerboard split from the provided points.")

    if y is None:
        return min(candidates, key=lambda candidate: abs(candidate["test_size"] - candidate["train_size"]))

    valid = [candidate for candidate in candidates if candidate["valid_class_split"]]
    pool = valid or candidates
    return min(
        pool,
        key=lambda candidate: (
            0 if candidate.get("test_pos", 0) > 0 else 1,
            candidate.get("test_pos", 0) if candidate.get("test_pos", 0) > 0 else np.inf,
            abs(candidate["test_size"] - candidate["train_size"]),
        ),
    )


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
    """Plot the study-area overview map and optional raster previews."""
    from matplotlib.lines import Line2D
    from pathlib import Path
    import rasterio

    lith_units = pd.Index(lith_gdf["lithology_family"].dropna().unique())
    unit_colors = (
        CATEGORICAL_COLORS[: len(lith_units)]
        if len(lith_units) <= len(CATEGORICAL_COLORS)
        else [mcolors.to_hex(color) for color in plt.get_cmap("tab20", len(lith_units)).colors]
    )

    fig, ax = plt.subplots(figsize=figsize)
    plot_vector(
        lith_gdf,
        column="lithology_family",
        categorical=True,
        categorical_cmap=mcolors.ListedColormap(unit_colors),
        ax=ax,
        title="Sample Locations and Known Deposits",
        alpha=0.3,
        edgecolor="gray",
        linewidth=0.4,
        legend=False,
    )

    if geochem_color_col is None:
        geochem_gdf.plot(ax=ax, color="steelblue", markersize=15, alpha=0.7)
    else:
        if geochem_color_col not in geochem_gdf.columns:
            raise ValueError(f"Column not found in geochem data: {geochem_color_col}")
        scatter = ax.scatter(
            geochem_gdf.geometry.x,
            geochem_gdf.geometry.y,
            c=geochem_gdf[geochem_color_col],
            cmap=geochem_cmap,
            s=18,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label=geochem_color_col)

    tgt_gdf.plot(
        ax=ax,
        marker="*",
        color="gold",
        markersize=180,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    sample_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="steelblue", markeredgecolor="white", markersize=7, label="Geochem samples"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=14, label="Known deposits"),
    ]
    sample_legend = ax.legend(handles=sample_handles, loc="upper right", title="Legend")
    ax.add_artist(sample_legend)
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="gray", label=unit) for unit, color in zip(lith_units, unit_colors)],
        loc="lower left",
        title="Lithology units",
        fontsize=8,
        title_fontsize=9,
    )
    plt.tight_layout()

    if spectral_dir is None and geophys_dir is None:
        return fig, ax

    spectral_paths = sorted(Path(spectral_dir).glob("*.tif")) if spectral_dir is not None else []
    geophys_paths = sorted(Path(geophys_dir).glob("*.tif")) if geophys_dir is not None else []
    raster_paths = spectral_paths + geophys_paths
    ncols = max(1, raster_ncols)
    nrows = -(-len(raster_paths) // ncols)
    fig_rasters, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_raster, path in zip(axes_flat, raster_paths):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            bounds = src.bounds
        vmin, vmax = np.nanpercentile(data, [2, 98])
        ax_raster.imshow(
            data,
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
            origin="upper",
            aspect="auto",
            cmap="RdBu_r" if path in geophys_paths else "YlOrBr",
            vmin=vmin,
            vmax=vmax,
        )
        label = ("mag " if path in geophys_paths else "spec ") + path.stem.replace("idx_", "").replace("_", " ").replace("AMF", "")
        ax_raster.set_title(label.strip(), fontsize=8, pad=3)
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])

    for ax_raster in axes_flat[len(raster_paths):]:
        ax_raster.set_visible(False)

    fig_rasters.suptitle(f"Raster Data Preview - {len(raster_paths)} layers", fontsize=11, y=1.01)
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
    """Summarize the predictor stack used in the supervised-learning section."""
    del spectral_dir, geophys_dir, tgt_gdf

    if feature_cols is None:
        feature_cols = get_geochem_columns(geochem_gdf)
    feature_cols = list(feature_cols)

    preview_geochem_cols = [column for column in preview_elements if column in feature_cols]
    if len(preview_geochem_cols) < 6:
        extras = [column for column in feature_cols if column not in preview_geochem_cols]
        preview_geochem_cols.extend(extras[: 6 - len(preview_geochem_cols)])

    xs = geochem_gdf.geometry.x.to_numpy()
    ys = geochem_gdf.geometry.y.to_numpy()

    lith_col = choose_lithology_column(lith_gdf)
    sampled_lith = None
    if lith_col is not None:
        joined = gpd.sjoin(
            geochem_gdf[["geometry"]],
            lith_gdf[[lith_col, "geometry"]],
            how="left",
            predicate="within",
        )
        sampled_lith = (
            joined[~joined.index.duplicated(keep="first")]
            .reindex(geochem_gdf.index)[lith_col]
            .fillna("No match")
            .astype(str)
        )

    panel_specs = [("labels", "training labels", y)]
    if sampled_lith is not None:
        panel_specs.append(("lith", lith_col.replace("_", " "), sampled_lith))
    for geochem_col in preview_geochem_cols:
        panel_specs.append(
            ("geochem", geochem_col, geochem_X[:, feature_cols.index(geochem_col)])
        )
    panel_specs.extend(("raster", name, X_raw[:, index]) for index, name in enumerate(predictor_names))

    ncols = 5
    nrows = -(-len(panel_specs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(3.4, nrows * 3.0)))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (kind, name, values) in zip(axes_flat, panel_specs):
        plot_vector(lith_gdf, ax=ax, alpha=0.08, edgecolor="gray", linewidth=0.2)

        if kind == "labels":
            neg = y == 0
            pos = y == 1
            ax.scatter(xs[neg], ys[neg], color="steelblue", s=10, linewidths=0, alpha=0.55)
            ax.scatter(xs[pos], ys[pos], color="red", s=16, linewidths=0, alpha=0.85)
            ax.set_title(f"{name} ({radius_m // 1000} km)", fontsize=7, pad=2)
        elif kind == "lith":
            categories = pd.Index(pd.unique(values))
            color_lookup = dict(zip(categories, plt.cm.tab20(np.linspace(0, 1, len(categories)))))
            ax.scatter(xs, ys, c=[color_lookup[value] for value in values], s=12, linewidths=0, alpha=0.9)
            ax.set_title(f"lithology: {name}", fontsize=7, pad=2)
        else:
            cmap = "plasma" if kind == "geochem" else ("RdBu_r" if name.startswith("mag ") else "YlOrBr")
            scatter = ax.scatter(xs, ys, c=values, cmap=cmap, s=12, linewidths=0, alpha=0.9)
            plt.colorbar(scatter, ax=ax, shrink=0.75)
            ax.set_title(f"geochem: {name}" if kind == "geochem" else name, fontsize=7, pad=2)

        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(panel_specs):]:
        ax.set_visible(False)

    fig.suptitle("Supervised ML Inputs at Sample Points", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig, axes


def get_feature_importance(rf, feature_names):
    """Return a DataFrame of feature importances sorted descending."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_probability_map(
    gdf_valid, y_valid, y_prob_all, lith_gdf, tgt_gdf, radius_m, figsize=(16, 7)
):
    """Map predicted probability at each valid sample location."""
    del y_valid, radius_m

    fig, ax = plt.subplots(figsize=figsize)
    plot_vector(lith_gdf, ax=ax, alpha=0.2, edgecolor="gray", linewidth=0.4)
    scatter = ax.scatter(
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
    plt.colorbar(scatter, ax=ax, label="Predicted probability", shrink=0.7)
    ax.set_title("Predicted Probability (all samples)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_spatial_split(gdf_valid, split, lith_gdf, tgt_gdf=None, figsize=(8, 5)):
    """Plot the checkerboard split and train/test sample assignments."""
    from pyproj import Transformer

    train_mask = split["train_mask"]
    test_mask = split["test_mask"]
    cell_size = split["cell_size_m"]

    fig, ax = plt.subplots(figsize=figsize)
    plot_vector(lith_gdf, ax=ax, alpha=0.15, edgecolor="gray", linewidth=0.3)

    metric_gdf = gdf_valid.to_crs(split["metric_crs"])
    xs_m = metric_gdf.geometry.x.to_numpy()
    ys_m = metric_gdf.geometry.y.to_numpy()
    x_origin = xs_m.min() - split["x_offset_m"]
    y_origin = ys_m.min() - split["y_offset_m"]

    def grid_lines(origin, values, size):
        start = int(np.floor((values.min() - origin) / size)) - 1
        stop = int(np.ceil((values.max() - origin) / size)) + 1
        return [origin + step * size for step in range(start, stop + 1)]

    x_lines = grid_lines(x_origin, xs_m, cell_size)
    y_lines = grid_lines(y_origin, ys_m, cell_size)
    transformer = Transformer.from_crs(split["metric_crs"], gdf_valid.crs, always_xy=True)

    for x_line in x_lines:
        ys_line = np.linspace(y_lines[0], y_lines[-1], 10)
        lons, lats = transformer.transform(np.full(10, x_line), ys_line)
        ax.plot(lons, lats, color="dimgray", linewidth=0.4, alpha=0.35, zorder=2)

    for y_line in y_lines:
        xs_line = np.linspace(x_lines[0], x_lines[-1], 10)
        lons, lats = transformer.transform(xs_line, np.full(10, y_line))
        ax.plot(lons, lats, color="dimgray", linewidth=0.4, alpha=0.35, zorder=2)

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

    gx = gdf_valid.geometry.x
    gy = gdf_valid.geometry.y
    margin_x = (gx.max() - gx.min()) * 0.02
    margin_y = (gy.max() - gy.min()) * 0.02
    ax.set_xlim(gx.min() - margin_x, gx.max() + margin_x)
    ax.set_ylim(gy.min() - margin_y, gy.max() + margin_y)
    ax.set_title(
        f"Spatial Checkerboard Split - {cell_size / 1000:.0f} km cells\n"
        "(adjacent cells alternate train/test to reduce spatial leakage)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax
