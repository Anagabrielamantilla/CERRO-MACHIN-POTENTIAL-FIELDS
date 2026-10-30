"""Model export helpers for NumPy and ParaView-compatible VTK files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def model_vector_to_grid(model: np.ndarray, shape_cells: tuple[int, int, int]) -> np.ndarray:
    """Return a 3-D model cube using the TensorMesh cell ordering."""
    model = np.asarray(model, dtype=float)
    if model.size != int(np.prod(shape_cells)):
        raise ValueError(
            f"Model size {model.size} does not match mesh shape {shape_cells}"
        )
    return model.reshape(shape_cells, order="F")


def save_model_npy(
    path: str | Path, model: np.ndarray, shape_cells: tuple[int, int, int]
) -> Path:
    """Save the full inverted model as a 3-D NumPy array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, model_vector_to_grid(model, shape_cells))
    return path


def _write_values(handle, values: np.ndarray, per_line: int = 6) -> None:
    values = np.asarray(values).ravel(order="F")
    for start in range(0, values.size, per_line):
        line = " ".join(f"{value:.12g}" for value in values[start : start + per_line])
        handle.write(f"{line}\n")


def apply_active_mask(model: np.ndarray, active: np.ndarray | None) -> np.ndarray:
    """Set inactive, above-topography cells to NaN."""
    masked = np.asarray(model, dtype=float).copy()
    if active is None:
        return masked
    active = np.asarray(active, dtype=bool)
    if active.size != masked.size:
        raise ValueError(
            f"Active mask size {active.size} does not match model size {masked.size}"
        )
    masked[~active] = np.nan
    return masked


def save_model_vtk(
    path: str | Path,
    mesh,
    cell_data: dict[str, np.ndarray],
) -> Path:
    """Write a legacy ASCII VTK rectilinear grid readable by ParaView."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = (int(v) for v in mesh.shape_cells)
    n_cells = nx * ny * nz

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("MANTLE-3D inverted model\n")
        handle.write("ASCII\n")
        handle.write("DATASET RECTILINEAR_GRID\n")
        handle.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")

        handle.write(f"X_COORDINATES {nx + 1} double\n")
        _write_values(handle, np.asarray(mesh.nodes_x, dtype=float))
        handle.write(f"Y_COORDINATES {ny + 1} double\n")
        _write_values(handle, np.asarray(mesh.nodes_y, dtype=float))
        handle.write(f"Z_COORDINATES {nz + 1} double\n")
        _write_values(handle, np.asarray(mesh.nodes_z, dtype=float))

        handle.write(f"CELL_DATA {n_cells}\n")
        for name, values in cell_data.items():
            array = np.asarray(values, dtype=float)
            if array.size != n_cells:
                raise ValueError(
                    f"Cell array {name!r} has {array.size} values, expected {n_cells}"
                )
            handle.write(f"SCALARS {name} double 1\n")
            handle.write("LOOKUP_TABLE default\n")
            _write_values(handle, array)
    return path


def export_inverted_model(
    output_npz: str | Path,
    mesh,
    model_full: np.ndarray,
    model_name: str,
    active: np.ndarray | None = None,
    extra_cell_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Save the inverted 3-D model next to an existing NPZ result."""
    output_npz = Path(output_npz)
    stem = output_npz.with_suffix("")
    npy_path = stem.with_name(f"{stem.name}_model_3d.npy")
    vtk_path = stem.with_name(f"{stem.name}_model_3d.vtk")

    model_full = apply_active_mask(model_full, active)
    save_model_npy(npy_path, model_full, tuple(mesh.shape_cells))

    cell_data = {model_name: model_full}
    if active is not None:
        active = np.asarray(active, dtype=bool)
        cell_data["active"] = active.astype(float)
    if extra_cell_data:
        cell_data.update(
            {name: apply_active_mask(values, active) for name, values in extra_cell_data.items()}
        )
    save_model_vtk(vtk_path, mesh, cell_data)
    return {"npy": npy_path, "vtk": vtk_path}
