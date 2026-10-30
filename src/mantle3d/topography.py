"""Topography loading and resampling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def resolve_topography_path(root: str | Path, config: dict) -> Path | None:
    """Return the configured DEM path, if one is defined."""
    topo_cfg = config.get("topography", {})
    path = topo_cfg.get("path")
    if not path:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def load_topography_xyz(root: str | Path, config: dict) -> np.ndarray | None:
    """Load an XYZ topography text file from the repository configuration."""
    path = resolve_topography_path(root, config)
    if path is None:
        return None
    try:
        values = np.loadtxt(path)
    except ValueError:
        values = np.loadtxt(path, skiprows=1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected topography XYZ with shape (n, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Topography file contains non-finite values: {path}")
    return values.astype(float, copy=False)


def resample_topography(
    mesh,
    station_xyz: np.ndarray,
    config: dict,
    root: str | Path,
    grid_points: int,
) -> tuple[np.ndarray, str]:
    """Sample configured DEM on the mesh footprint and append station elevations."""
    dem = load_topography_xyz(root, config)
    if dem is None:
        return _station_interpolated_topography(mesh, station_xyz, grid_points), "stations"

    x_grid = np.linspace(mesh.nodes_x.min(), mesh.nodes_x.max(), int(grid_points))
    y_grid = np.linspace(mesh.nodes_y.min(), mesh.nodes_y.max(), int(grid_points))
    xx, yy = np.meshgrid(x_grid, y_grid)
    targets = np.c_[xx.ravel(), yy.ravel()]
    _, indices = cKDTree(dem[:, :2]).query(targets, k=1)
    dem_surface = np.c_[targets, dem[indices, 2]]
    return np.vstack([dem_surface, np.asarray(station_xyz, dtype=float)]), "dem_plus_stations"


def topography_reference_elevation(station_xyz: np.ndarray, config: dict, root: str | Path) -> float:
    """Return maximum elevation from DEM plus stations when available."""
    station_max = float(np.nanmax(np.asarray(station_xyz)[:, 2]))
    dem = load_topography_xyz(root, config)
    if dem is None:
        return station_max
    return float(max(station_max, np.nanmax(dem[:, 2])))


def _station_interpolated_topography(mesh, station_xyz: np.ndarray, grid_points: int) -> np.ndarray:
    from scipy.interpolate import griddata

    x_grid = np.linspace(mesh.nodes_x.min(), mesh.nodes_x.max(), int(grid_points))
    y_grid = np.linspace(mesh.nodes_y.min(), mesh.nodes_y.max(), int(grid_points))
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz_linear = griddata(station_xyz[:, :2], station_xyz[:, 2], (xx, yy), method="linear")
    zz_nearest = griddata(station_xyz[:, :2], station_xyz[:, 2], (xx, yy), method="nearest")
    zz = np.where(np.isnan(zz_linear), zz_nearest, zz_linear)
    return np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
