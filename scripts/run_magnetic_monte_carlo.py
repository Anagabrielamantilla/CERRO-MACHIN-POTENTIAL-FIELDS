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
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/monte_carlo/magnetic"

import sys

sys.path.insert(0, str(ROOT / "src"))
from mantle3d.exports import save_model_npy  # noqa: E402
from mantle3d.magnetics import load_config, load_magnetics, run_inversion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo uncertainty analysis for the independent magnetic inversion."
    )
    parser.add_argument("--n", type=int, default=200, help="Number of Monte Carlo realizations.")
    parser.add_argument("--seed", type=int, default=20260906, help="Random seed.")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the configured magnetic uncertainty floor.",
    )
    return parser.parse_args()


def model_cube(model_full: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(model_full, dtype=float).reshape(shape, order="F")


def save_active_vtk(path: Path, mesh, active: np.ndarray, cell_data: dict[str, np.ndarray]) -> None:
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
    mean_si: np.ndarray,
    std_si: np.ndarray,
    p05_si: np.ndarray,
    p95_si: np.ndarray,
    metrics: list[dict[str, float]],
) -> None:
    y_index = shape[1] // 2
    active_std = std_si[active]
    vmax_std = float(np.nanpercentile(active_std, 98)) if active_std.size else 1.0
    vmax_mean = float(np.nanpercentile(mean_si[active], 98)) if np.any(active) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_y_section(axes[0], centers, shape, mean_si, y_index, "Media Monte Carlo", "SI", "viridis", vmin=0, vmax=vmax_mean)
    plot_y_section(axes[1], centers, shape, std_si, y_index, "Incertidumbre 1 sigma", "SI", "magma", vmin=0, vmax=vmax_std)
    plot_y_section(axes[2], centers, shape, p95_si - p05_si, y_index, "Intervalo P95-P05", "SI", "plasma", vmin=0)
    fig.suptitle(f"Monte Carlo magnetometria: corte Y central (indice {y_index})")
    fig.savefig(OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_y_mid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(active_std, bins=50, color="#4C78A8", edgecolor="white")
    axes[0].set_xlabel("1 sigma susceptibility uncertainty (SI)")
    axes[0].set_ylabel("Active cell count")
    axes[0].set_title("Uncertainty distribution")
    phi = np.asarray([row["phi_d"] for row in metrics], dtype=float)
    rmse_train = np.asarray([row["rmse_train_nt"] for row in metrics], dtype=float)
    axes[1].scatter(phi, rmse_train, c=np.arange(phi.size), cmap="viridis", s=35)
    axes[1].set_xlabel("Final phi_d")
    axes[1].set_ylabel("Train RMSE (nT)")
    axes[1].set_title("Experiment fit metrics")
    fig.savefig(OUTPUT_DIR / "magnetic_monte_carlo_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.n < 2:
        raise ValueError("Monte Carlo requires at least two realizations.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    config = load_config(ROOT)
    frame = load_magnetics(ROOT, config)
    observed = frame["Anomalia (nT)"].to_numpy(float)
    uncertainty_floor = max(
        float(config["uncertainty"]["absolute_floor_nt"]),
        float(config["uncertainty"]["fraction_of_maximum"]) * np.max(np.abs(observed)),
    )
    sigma = args.noise_scale * uncertainty_floor
    rng = np.random.default_rng(args.seed)

    models_si: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    geometry_result = None

    for run_id in range(args.n):
        noise = rng.normal(loc=0.0, scale=sigma, size=observed.shape)
        perturbed = observed + noise
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_inversion(ROOT, observed_override=perturbed)
        (logs_dir / f"run_{run_id:03d}.log").write_text(buffer.getvalue())
        if geometry_result is None:
            geometry_result = result

        model_full_si = np.asarray(result["model_full"], dtype=float)
        models_si.append(model_full_si)
        train = result["train"]
        holdout = result["holdout"]
        pred_train = np.asarray(result["predicted_train"], dtype=float)
        pred_holdout = np.asarray(result["predicted_holdout"], dtype=float)
        metrics.append(
            {
                "run_id": run_id,
                "noise_mean_nt": float(noise.mean()),
                "noise_std_nt": float(noise.std(ddof=1)),
                "phi_d": float(result["phi_d"]),
                "rmse_train_nt": float(np.sqrt(np.mean((pred_train - perturbed[train]) ** 2))),
                "rmse_holdout_vs_original_nt": float(np.sqrt(np.mean((pred_holdout - observed[holdout]) ** 2))),
                "susceptibility_min_si": float(np.nanmin(model_full_si)),
                "susceptibility_max_si": float(np.nanmax(model_full_si)),
            }
        )
        print(f"run {run_id + 1:03d}/{args.n}: phi_d={metrics[-1]['phi_d']:.3f}, noise_std={metrics[-1]['noise_std_nt']:.3f} nT")

    if geometry_result is None:
        raise RuntimeError("No Monte Carlo realizations were completed.")

    stack = np.stack(models_si, axis=0)
    active = np.isfinite(stack).all(axis=0)
    mean_si = np.full(stack.shape[1], np.nan)
    std_si = np.full(stack.shape[1], np.nan)
    p05_si = np.full(stack.shape[1], np.nan)
    p50_si = np.full(stack.shape[1], np.nan)
    p95_si = np.full(stack.shape[1], np.nan)
    mean_si[active] = np.mean(stack[:, active], axis=0)
    std_si[active] = np.std(stack[:, active], axis=0, ddof=1)
    p05_si[active] = np.percentile(stack[:, active], 5, axis=0)
    p50_si[active] = np.percentile(stack[:, active], 50, axis=0)
    p95_si[active] = np.percentile(stack[:, active], 95, axis=0)

    mesh = geometry_result["mesh"]
    shape = tuple(int(v) for v in mesh.shape_cells)
    centers = np.asarray(mesh.cell_centers, dtype=float)

    np.savez_compressed(
        OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_cube.npz",
        cell_centers=centers,
        mesh_shape=np.asarray(shape),
        active=active,
        susceptibility_models_si=stack,
        susceptibility_mean_si=mean_si,
        susceptibility_std_si=std_si,
        susceptibility_p05_si=p05_si,
        susceptibility_p50_si=p50_si,
        susceptibility_p95_si=p95_si,
        metrics_json=np.array(json.dumps(metrics)),
        uncertainty_floor_nt=np.array(uncertainty_floor),
        noise_sigma_nt=np.array(sigma),
        seed=np.array(args.seed),
        n_realizations=np.array(args.n),
    )
    save_model_npy(
        OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_cube_model_3d.npy",
        std_si,
        shape,
    )
    save_active_vtk(
        OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_cube_model_3d.vtk",
        mesh,
        active,
        {
            "susceptibility_uncertainty_1sigma_si": std_si,
            "susceptibility_mean_si": mean_si,
            "susceptibility_p05_si": p05_si,
            "susceptibility_p50_si": p50_si,
            "susceptibility_p95_si": p95_si,
        },
    )
    save_figures(centers, shape, active, mean_si, std_si, p05_si, p95_si, metrics)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Gaussian data perturbation Monte Carlo using the same 02_magnetic_inversion SimPEG workflow.",
        "n_realizations": args.n,
        "seed": args.seed,
        "noise_sigma_nt": sigma,
        "uncertainty_floor_nt": uncertainty_floor,
        "susceptibility_units": "SI",
        "bounds_si": [
            float(config["inversion"]["effective_susceptibility_lower_si"]),
            float(config["inversion"]["effective_susceptibility_upper_si"]),
        ],
        "active_cells": int(active.sum()),
        "std_si_summary": {
            "min": float(np.nanmin(std_si[active])),
            "median": float(np.nanmedian(std_si[active])),
            "mean": float(np.nanmean(std_si[active])),
            "p90": float(np.nanpercentile(std_si[active], 90)),
            "p95": float(np.nanpercentile(std_si[active], 95)),
            "max": float(np.nanmax(std_si[active])),
        },
        "outputs": {
            "npz": "magnetic_monte_carlo_uncertainty_cube.npz",
            "vtk": "magnetic_monte_carlo_uncertainty_cube_model_3d.vtk",
            "npy": "magnetic_monte_carlo_uncertainty_cube_model_3d.npy",
            "figures": [
                "magnetic_monte_carlo_uncertainty_y_mid.png",
                "magnetic_monte_carlo_metrics.png",
            ],
            "logs": "logs/run_*.log",
        },
        "metrics": metrics,
    }
    (OUTPUT_DIR / "magnetic_monte_carlo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    description = f"""Monte Carlo de incertidumbre para la inversion magnetometrica independiente

Metodo
------
Se perturbaron los datos observados de anomalia magnetica con ruido Gaussiano independiente y media cero:

    d_i_mc = d_i + N(0, sigma)

Cada realizacion se invirtio con el mismo flujo de 02_magnetic_inversion:
misfit L2 ponderado por incertidumbre, regularizacion WeightedLeastSquares,
pesos de sensibilidad, beta estimado por autovalores, enfriamiento beta y bounds
de susceptibilidad efectiva no negativa.

Parametros del experimento
--------------------------
Fecha UTC: {summary['created_utc']}
Numero de realizaciones: {args.n}
Semilla aleatoria: {args.seed}
Piso de incertidumbre del notebook: {uncertainty_floor:.6f} nT
Factor de ruido Monte Carlo: {args.noise_scale:.6f}
Sigma usado para perturbar datos: {sigma:.6f} nT
Bounds susceptibilidad: {summary['bounds_si'][0]:.6f} a {summary['bounds_si'][1]:.6f} SI
Malla: {shape}
Celdas activas usadas en el cubo final: {int(active.sum())}

Productos generados
-------------------
magnetic_monte_carlo_uncertainty_cube.npz:
  - susceptibility_models_si: modelos de todas las realizaciones
  - susceptibility_mean_si: media del ensamble
  - susceptibility_std_si: cubo de incertidumbre 1 sigma
  - susceptibility_p05_si, susceptibility_p50_si, susceptibility_p95_si: percentiles

magnetic_monte_carlo_uncertainty_cube_model_3d.vtk:
  - susceptibility_uncertainty_1sigma_si
  - susceptibility_mean_si
  - susceptibility_p05_si
  - susceptibility_p50_si
  - susceptibility_p95_si

Figuras:
  - magnetic_monte_carlo_uncertainty_y_mid.png
  - magnetic_monte_carlo_metrics.png

Resumen del cubo de incertidumbre 1 sigma en celdas activas
-----------------------------------------------------------
min: {summary['std_si_summary']['min']:.12g} SI
median: {summary['std_si_summary']['median']:.12g} SI
mean: {summary['std_si_summary']['mean']:.12g} SI
p90: {summary['std_si_summary']['p90']:.12g} SI
p95: {summary['std_si_summary']['p95']:.12g} SI
max: {summary['std_si_summary']['max']:.12g} SI

Detalles por realizacion
------------------------
"""
    for row in metrics:
        description += (
            f"run {int(row['run_id']):03d}: "
            f"noise_mean={row['noise_mean_nt']:.6f} nT, "
            f"noise_std={row['noise_std_nt']:.6f} nT, "
            f"phi_d={row['phi_d']:.6f}, "
            f"rmse_train={row['rmse_train_nt']:.6f} nT, "
            f"rmse_holdout_vs_original={row['rmse_holdout_vs_original_nt']:.6f} nT, "
            f"susceptibility_range=[{row['susceptibility_min_si']:.12g}, {row['susceptibility_max_si']:.12g}] SI\n"
        )
    (OUTPUT_DIR / "magnetic_monte_carlo_method_and_run.txt").write_text(description)

    print("OUTPUT_DIR", OUTPUT_DIR)
    print("NPZ", OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_cube.npz")
    print("VTK", OUTPUT_DIR / "magnetic_monte_carlo_uncertainty_cube_model_3d.vtk")
    print("STD median SI", summary["std_si_summary"]["median"])
    print("STD p95 SI", summary["std_si_summary"]["p95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
