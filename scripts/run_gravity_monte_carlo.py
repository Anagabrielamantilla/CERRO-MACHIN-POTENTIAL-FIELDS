"""New gravity Monte Carlo runs with the current smooth solver; archives remain untouched."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/monte_carlo/gravity_new"

import sys

sys.path.insert(0, str(ROOT / "src"))
from mantle3d.exports import save_model_npy  # noqa: E402
from mantle3d.gravity import load_config, load_gravity, run_inversion, spatial_holdout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo uncertainty analysis for the independent gravity inversion."
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of Monte Carlo realizations.")
    parser.add_argument("--seed", type=int, default=20260906, help="Random seed.")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the configured gravity uncertainty floor.",
    )
    parser.add_argument("--skip-vtk", action="store_true", help="Skip VTK export; pyvista is not required.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="New, empty destination (default: outputs/monte_carlo/gravity_new).")
    return parser.parse_args()


def model_cube(model_full: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(model_full, dtype=float).reshape(shape, order="F")


def save_active_vtk(path: Path, mesh, active: np.ndarray, cell_data: dict[str, np.ndarray]) -> None:
    import pyvista as pv

    grid_full = pv.RectilinearGrid(mesh.nodes_x, mesh.nodes_y, mesh.nodes_z)
    for name, values in cell_data.items():
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != mesh.n_cells:
            raise ValueError(f"{name} has {values.size} cells; expected {mesh.n_cells}")
        grid_full.cell_data[name] = values
    grid_full.cell_data["active"] = np.asarray(active, dtype=np.uint8)
    grid = grid_full.extract_cells(np.flatnonzero(active))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def plot_y_section(
    ax: plt.Axes,
    centers: np.ndarray,
    shape: tuple[int, int, int],
    values_full: np.ndarray,
    y_index: int,
    title: str,
    label: str,
    cmap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    cube = model_cube(values_full, shape)
    x = np.unique(centers[:, 0])
    z = np.unique(centers[:, 2])
    x_km = (x - x.min()) / 1000.0
    z_km = z / 1000.0
    image = ax.pcolormesh(x_km, z_km, cube[:, y_index, :].T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Distancia X (km)")
    ax.set_ylabel("Elevacion Z (km)")
    plt.colorbar(image, ax=ax, label=label, pad=0.02)


def save_figures(
    centers: np.ndarray,
    shape: tuple[int, int, int],
    active: np.ndarray,
    mean_kgm3: np.ndarray,
    std_kgm3: np.ndarray,
    p05_kgm3: np.ndarray,
    p95_kgm3: np.ndarray,
    metrics: list[dict[str, float]],
) -> None:
    y_index = shape[1] // 2
    active_std = std_kgm3[active]
    vmax_std = float(np.nanpercentile(active_std, 98)) if active_std.size else 1.0
    vmax_mean = float(np.nanpercentile(np.abs(mean_kgm3[active]), 98)) if np.any(active) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_y_section(axes[0], centers, shape, mean_kgm3, y_index, "Media Monte Carlo", "kg/m3", "RdBu_r", vmin=-vmax_mean, vmax=vmax_mean)
    plot_y_section(axes[1], centers, shape, std_kgm3, y_index, "Incertidumbre 1 sigma", "kg/m3", "magma", vmin=0, vmax=vmax_std)
    plot_y_section(axes[2], centers, shape, p95_kgm3 - p05_kgm3, y_index, "Intervalo P95-P05", "kg/m3", "plasma", vmin=0)
    fig.suptitle(f"Monte Carlo gravimetria: corte Y central (indice {y_index})")
    fig.savefig(OUTPUT_DIR / "gravity_monte_carlo_uncertainty_y_mid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(active_std, bins=50, color="#4C78A8", edgecolor="white")
    axes[0].set_xlabel("1 sigma density uncertainty (kg/m3)")
    axes[0].set_ylabel("Active cell count")
    axes[0].set_title("Uncertainty distribution")
    phi = np.asarray([row["phi_d"] for row in metrics], dtype=float)
    rmse_train = np.asarray([row["rmse_train_mgal"] for row in metrics], dtype=float)
    axes[1].scatter(phi, rmse_train, c=np.arange(phi.size), cmap="viridis", s=35)
    axes[1].set_xlabel("Final phi_d")
    axes[1].set_ylabel("Train RMSE (mGal)")
    axes[1].set_title("Experiment fit metrics")
    fig.savefig(OUTPUT_DIR / "gravity_monte_carlo_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global OUTPUT_DIR
    args = parse_args()
    if args.n < 2:
        raise ValueError("Monte Carlo requires at least two realizations.")

    if not np.isfinite(args.noise_scale) or args.noise_scale <= 0:
        raise ValueError("noise-scale must be finite and positive")
    if not args.skip_vtk:
        import importlib
        importlib.import_module("pyvista")  # Fail before inversion if VTK dependencies are absent.
    OUTPUT_DIR = args.output_dir.resolve()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    config = load_config(ROOT)
    frame = load_gravity(ROOT, config)
    observed = frame["value"].to_numpy(float)
    xyz = frame[["x", "y", "z"]].to_numpy(float)
    train_indices, _ = spatial_holdout(xyz, int(config["validation"]["holdout_stride"]))
    uncertainty_floor = max(
        float(config["uncertainty"]["absolute_floor_mgal"]),
        float(config["uncertainty"]["fraction_of_maximum"]) * np.max(np.abs(observed[train_indices])),
    )
    if not np.isfinite(uncertainty_floor) or uncertainty_floor <= 0:
        raise ValueError("Configured gravity uncertainty floor must be finite and positive")
    sigma = args.noise_scale * uncertainty_floor
    rng = np.random.default_rng(args.seed)

    models_kgm3: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    geometry_result = None

    for run_id in range(args.n):
        noise = rng.normal(loc=0.0, scale=sigma, size=observed.shape)
        perturbed = observed + noise
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_inversion(ROOT, observed_override=perturbed, uncertainty_floor_override=uncertainty_floor)
        (logs_dir / f"run_{run_id:03d}.log").write_text(buffer.getvalue())
        if geometry_result is None:
            geometry_result = result

        model_full_kgm3 = 1000.0 * np.asarray(result["model_full"], dtype=float)
        models_kgm3.append(model_full_kgm3)
        train = result["train"]
        holdout = result["holdout"]
        pred_train = np.asarray(result["predicted_train_field"], dtype=float)
        pred_holdout = np.asarray(result["predicted_holdout_field"], dtype=float)
        metrics.append(
            {
                "run_id": run_id,
                "noise_mean_mgal": float(noise.mean()),
                "noise_std_mgal": float(noise.std(ddof=1)),
                "phi_d": float(result["phi_d"]),
                "rmse_train_mgal": float(np.sqrt(np.mean((pred_train - perturbed[train]) ** 2))),
                "rmse_holdout_vs_original_mgal": float(np.sqrt(np.mean((pred_holdout - observed[holdout]) ** 2))),
                "density_min_kgm3": float(np.nanmin(model_full_kgm3)),
                "density_max_kgm3": float(np.nanmax(model_full_kgm3)),
            }
        )
        print(f"run {run_id + 1:03d}/{args.n}: phi_d={metrics[-1]['phi_d']:.3f}, noise_std={metrics[-1]['noise_std_mgal']:.3f} mGal")

    if geometry_result is None:
        raise RuntimeError("No Monte Carlo realizations were completed.")

    stack = np.stack(models_kgm3, axis=0)
    active = np.asarray(geometry_result["active"], dtype=bool)
    if not np.isfinite(stack[:, active]).all():
        raise ValueError("A realization contains nonfinite density in active mesh cells")
    mean_kgm3 = np.full(stack.shape[1], np.nan)
    std_kgm3 = np.full(stack.shape[1], np.nan)
    p05_kgm3 = np.full(stack.shape[1], np.nan)
    p50_kgm3 = np.full(stack.shape[1], np.nan)
    p95_kgm3 = np.full(stack.shape[1], np.nan)
    mean_kgm3[active] = np.mean(stack[:, active], axis=0)
    std_kgm3[active] = np.std(stack[:, active], axis=0, ddof=1)
    p05_kgm3[active] = np.percentile(stack[:, active], 5, axis=0)
    p50_kgm3[active] = np.percentile(stack[:, active], 50, axis=0)
    p95_kgm3[active] = np.percentile(stack[:, active], 95, axis=0)

    mesh = geometry_result["mesh"]
    shape = tuple(int(v) for v in mesh.shape_cells)
    centers = np.asarray(mesh.cell_centers, dtype=float)

    np.savez_compressed(
        OUTPUT_DIR / "gravity_monte_carlo_uncertainty_cube.npz",
        cell_centers=centers,
        mesh_shape=np.asarray(shape),
        active=active,
        density_models_kgm3=stack,
        density_mean_kgm3=mean_kgm3,
        density_std_kgm3=std_kgm3,
        density_p05_kgm3=p05_kgm3,
        density_p50_kgm3=p50_kgm3,
        density_p95_kgm3=p95_kgm3,
        metrics_json=np.array(json.dumps(metrics)),
        uncertainty_floor_mgal=np.array(uncertainty_floor),
        noise_sigma_mgal=np.array(sigma),
        seed=np.array(args.seed),
        n_realizations=np.array(args.n),
    )
    save_model_npy(
        OUTPUT_DIR / "gravity_monte_carlo_uncertainty_cube_model_3d.npy",
        std_kgm3,
        shape,
    )
    if not args.skip_vtk:
        save_active_vtk(
            OUTPUT_DIR / "gravity_monte_carlo_uncertainty_cube_model_3d.vtk",
            mesh,
            active,
            {
                "density_uncertainty_1sigma_kgm3": std_kgm3,
                "density_mean_kgm3": mean_kgm3,
                "density_p05_kgm3": p05_kgm3,
                "density_p50_kgm3": p50_kgm3,
                "density_p95_kgm3": p95_kgm3,
            },
        )
    save_figures(centers, shape, active, mean_kgm3, std_kgm3, p05_kgm3, p95_kgm3, metrics)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Gaussian field-data perturbations with the current smooth WeightedLeastSquares gravity solver; not the historical IRLS implementation.",
        "config": config,
        "regularization": "WeightedLeastSquares",
        "n_realizations": args.n,
        "seed": args.seed,
        "noise_sigma_mgal": sigma,
        "uncertainty_floor_mgal": uncertainty_floor,
        "density_units": "kg/m3",
        "bounds_kgm3": [
            1000.0 * float(config["inversion"]["density_lower_gcc"]),
            1000.0 * float(config["inversion"]["density_upper_gcc"]),
        ],
        "active_cells": int(active.sum()),
        "std_kgm3_summary": {
            "min": float(np.nanmin(std_kgm3[active])),
            "median": float(np.nanmedian(std_kgm3[active])),
            "mean": float(np.nanmean(std_kgm3[active])),
            "p90": float(np.nanpercentile(std_kgm3[active], 90)),
            "p95": float(np.nanpercentile(std_kgm3[active], 95)),
            "max": float(np.nanmax(std_kgm3[active])),
        },
        "outputs": {
            "npz": "gravity_monte_carlo_uncertainty_cube.npz",
            "vtk": "gravity_monte_carlo_uncertainty_cube_model_3d.vtk",
            "npy": "gravity_monte_carlo_uncertainty_cube_model_3d.npy",
            "figures": [
                "gravity_monte_carlo_uncertainty_y_mid.png",
                "gravity_monte_carlo_metrics.png",
            ],
            "logs": "logs/run_*.log",
        },
        "metrics": metrics,
    }
    if args.skip_vtk:
        summary["outputs"].pop("vtk")
    (OUTPUT_DIR / "gravity_monte_carlo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    description = f"""Monte Carlo de incertidumbre para la inversion gravimetrica independiente

Metodo
------
Se perturbaron los datos observados de gravedad (convencion de campo, mGal) con ruido Gaussiano independiente y media cero:

    d_i_mc = d_i + N(0, sigma)

Cada realizacion se invirtio con el mismo flujo de 01_gravity_inversion:
misfit L2 ponderado por incertidumbre, regularizacion WeightedLeastSquares,
pesos de sensibilidad, beta estimado por autovalores, enfriamiento beta y bounds
de contraste de densidad del archivo de configuracion.

Parametros del experimento
--------------------------
Fecha UTC: {summary['created_utc']}
Numero de realizaciones: {args.n}
Semilla aleatoria: {args.seed}
Piso de incertidumbre del notebook: {uncertainty_floor:.6f} mGal
Factor de ruido Monte Carlo: {args.noise_scale:.6f}
Sigma usado para perturbar datos: {sigma:.6f} mGal
Bounds contraste de densidad: {summary['bounds_kgm3'][0]:.6f} a {summary['bounds_kgm3'][1]:.6f} kg/m3
Malla: {shape}
Celdas activas usadas en el cubo final: {int(active.sum())}
Exportacion VTK solicitada: {not args.skip_vtk}

Productos generados
-------------------
gravity_monte_carlo_uncertainty_cube.npz:
  - density_models_kgm3: modelos de todas las realizaciones
  - density_mean_kgm3: media del ensamble
  - density_std_kgm3: cubo de incertidumbre 1 sigma
  - density_p05_kgm3, density_p50_kgm3, density_p95_kgm3: percentiles

gravity_monte_carlo_uncertainty_cube_model_3d.vtk:
  - density_uncertainty_1sigma_kgm3
  - density_mean_kgm3
  - density_p05_kgm3
  - density_p50_kgm3
  - density_p95_kgm3

Figuras:
  - gravity_monte_carlo_uncertainty_y_mid.png
  - gravity_monte_carlo_metrics.png

Resumen del cubo de incertidumbre 1 sigma en celdas activas
-----------------------------------------------------------
min: {summary['std_kgm3_summary']['min']:.12g} kg/m3
median: {summary['std_kgm3_summary']['median']:.12g} kg/m3
mean: {summary['std_kgm3_summary']['mean']:.12g} kg/m3
p90: {summary['std_kgm3_summary']['p90']:.12g} kg/m3
p95: {summary['std_kgm3_summary']['p95']:.12g} kg/m3
max: {summary['std_kgm3_summary']['max']:.12g} kg/m3

Detalles por realizacion
------------------------
"""
    for row in metrics:
        description += (
            f"run {int(row['run_id']):03d}: "
            f"noise_mean={row['noise_mean_mgal']:.6f} mGal, "
            f"noise_std={row['noise_std_mgal']:.6f} mGal, "
            f"phi_d={row['phi_d']:.6f}, "
            f"rmse_train={row['rmse_train_mgal']:.6f} mGal, "
            f"rmse_holdout_vs_original={row['rmse_holdout_vs_original_mgal']:.6f} mGal, "
            f"density_range=[{row['density_min_kgm3']:.12g}, {row['density_max_kgm3']:.12g}] kg/m3\n"
        )
    (OUTPUT_DIR / "gravity_monte_carlo_method_and_run.txt").write_text(description)

    print("OUTPUT_DIR", OUTPUT_DIR)
    print("NPZ", OUTPUT_DIR / "gravity_monte_carlo_uncertainty_cube.npz")
    if not args.skip_vtk:
        print("VTK", OUTPUT_DIR / "gravity_monte_carlo_uncertainty_cube_model_3d.vtk")
    print("STD median kg/m3", summary["std_kgm3_summary"]["median"])
    print("STD p95 kg/m3", summary["std_kgm3_summary"]["p95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
