"""Persistence and integrity checks for the shared multiphysics mesh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from discretize import TensorMesh


MESH_DIR = Path("outputs/common_mesh")
MESH_FILE = "tensor_mesh.json"
GEOMETRY_FILE = "common_mesh_geometry.npz"
MANIFEST_FILE = "common_mesh_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_mesh_exists(root: str | Path) -> bool:
    directory = Path(root) / MESH_DIR
    return all((directory / name).exists() for name in (MESH_FILE, GEOMETRY_FILE, MANIFEST_FILE))


def save_common_mesh(
    mesh: TensorMesh,
    topography: np.ndarray,
    active: np.ndarray,
    root: str | Path,
    topography_source: str = "unknown",
    topography_path: str | None = None,
) -> dict[str, object]:
    """Persist the exact mesh, topography and active-cell mask with checksums."""
    directory = Path(root) / MESH_DIR
    directory.mkdir(parents=True, exist_ok=True)
    mesh_path = directory / MESH_FILE
    geometry_path = directory / GEOMETRY_FILE
    manifest_path = directory / MANIFEST_FILE

    mesh.save(mesh_path, verbose=False)
    np.savez_compressed(
        geometry_path,
        topography=np.asarray(topography),
        active=np.asarray(active, dtype=bool),
        crs=np.asarray("EPSG:9377"),
        z_positive=np.asarray("up"),
    )
    manifest = {
        "mesh_type": "TensorMesh",
        "crs": "EPSG:9377",
        "z_positive": "up",
        "shape_cells": list(mesh.shape_cells),
        "n_cells": int(mesh.n_cells),
        "n_active": int(np.asarray(active).sum()),
        "topography_source": topography_source,
        "topography_path": topography_path,
        "mesh_sha256": _sha256(mesh_path),
        "geometry_sha256": _sha256(geometry_path),
        "rule": "All independent inversions must load these files; do not rebuild the mesh.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_common_mesh(
    root: str | Path,
) -> tuple[TensorMesh, np.ndarray, np.ndarray, dict[str, object]]:
    """Load and verify the versioned common mesh artifacts."""
    directory = Path(root) / MESH_DIR
    mesh_path = directory / MESH_FILE
    geometry_path = directory / GEOMETRY_FILE
    manifest_path = directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    if _sha256(mesh_path) != manifest["mesh_sha256"]:
        raise ValueError("The common mesh JSON checksum does not match its manifest.")
    if _sha256(geometry_path) != manifest["geometry_sha256"]:
        raise ValueError("The common mesh geometry checksum does not match its manifest.")

    mesh = TensorMesh.deserialize(json.loads(mesh_path.read_text()))
    geometry = np.load(geometry_path)
    topography = geometry["topography"]
    active = geometry["active"].astype(bool)
    if active.size != mesh.n_cells:
        raise ValueError("Active-cell mask and mesh have incompatible sizes.")
    return mesh, topography, active, manifest
