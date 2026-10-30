"""Reproducible smooth 3-D gravity inversion used by notebook 01."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
from simpeg.potential_fields import gravity

from .common_mesh import common_mesh_exists, load_common_mesh, save_common_mesh
from .exports import export_inverted_model
from .topography import (
    resample_topography,
    resolve_topography_path,
    topography_reference_elevation,
)


def load_config(root: str | Path) -> dict:
    root = Path(root)
    return yaml.safe_load((root / "configs/gravity.yaml").read_text())


def load_gravity(root: str | Path, config: dict) -> pd.DataFrame:
    path = Path(root) / config["data"]["path"]
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                f"Expected gravity .npy data with shape (n, 4), got {values.shape}"
            )
        return pd.DataFrame(values, columns=["x", "y", "z", "value"])
    return pd.read_csv(path, sep=r"\s+", names=["x", "y", "z", "value"])


def spatial_holdout(xyz: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, spatially distributed train and holdout indices."""
    order = np.lexsort((xyz[:, 1], xyz[:, 0]))
    holdout = order[::stride]
    train = np.setdiff1d(np.arange(xyz.shape[0]), holdout)
    return train, holdout


def build_mesh_and_topography(
    xyz: np.ndarray, config: dict, root: str | Path = "."
) -> tuple[TensorMesh, np.ndarray, np.ndarray, str]:
    cfg = config["mesh"]
    h = float(cfg["core_cell_size_m"])
    margin = float(cfg["horizontal_margin_m"])
    nx = int(np.ceil((np.ptp(xyz[:, 0]) + margin) / h))
    ny = int(np.ceil((np.ptp(xyz[:, 1]) + margin) / h))
    nx += nx % 2
    ny += ny % 2

    npad = int(cfg["horizontal_padding_cells"])
    pfac = float(cfg["horizontal_padding_factor"])
    hx = [(h, npad, -pfac), (h, nx), (h, npad, pfac)]
    hy = [(h, npad, -pfac), (h, ny), (h, npad, pfac)]
    hz = [
        (h, int(cfg["vertical_padding_cells"]), -float(cfg["vertical_padding_factor"])),
        (h, int(cfg["vertical_core_cells"])),
    ]
    mesh = TensorMesh([hx, hy, hz], "CCN")
    mesh.origin += np.r_[
        xyz[:, 0].mean(),
        xyz[:, 1].mean(),
        topography_reference_elevation(xyz, config, root) + float(cfg["top_clearance_m"]),
    ]

    topography, topography_source = resample_topography(
        mesh, xyz, config, root, int(cfg["topography_grid_points"])
    )
    active = active_from_xyz(mesh, topography)
    return mesh, topography, active, topography_source


def _survey(locations: np.ndarray) -> gravity.survey.Survey:
    receivers = gravity.receivers.Point(locations, components="gz")
    source = gravity.sources.SourceField(receiver_list=[receivers])
    return gravity.survey.Survey(source)


def run_inversion(
    root: str | Path,
    observed_override: np.ndarray | None = None,
    uncertainty_floor_override: float | None = None,
) -> dict[str, object]:
    """Run the configured inversion and return models, predictions and geometry."""
    root = Path(root)
    config = load_config(root)
    frame = load_gravity(root, config)
    xyz = frame[["x", "y", "z"]].to_numpy(float)
    observed_field = frame["value"].to_numpy(float)
    if observed_override is not None:
        observed_field = np.asarray(observed_override, dtype=float)
        if observed_field.shape != (len(frame),) or not np.isfinite(observed_field).all():
            raise ValueError("observed_override must contain one finite field-convention mGal value per gravity station")
    sign_to_simpeg = float(config["data"]["observed_sign_to_simpeg"])
    observed_simpeg = sign_to_simpeg * observed_field
    train, holdout = spatial_holdout(xyz, int(config["validation"]["holdout_stride"]))
    expected_topography_source = (
        "dem_plus_stations" if config.get("topography", {}).get("path") else "stations"
    )
    resolved_topography_path = resolve_topography_path(root, config)
    expected_topography_path = (
        str(resolved_topography_path) if resolved_topography_path is not None else None
    )
    if common_mesh_exists(root):
        mesh, topography, active, mesh_manifest = load_common_mesh(root)
        mesh_source = "loaded"

    else:
        mesh, topography, active, topography_source = build_mesh_and_topography(
            xyz, config, root
        )
        mesh_manifest = save_common_mesh(
            mesh,
            topography,
            active,
            root,
            topography_source=topography_source,
            topography_path=expected_topography_path,
        )
        mesh_source = "created"

    receiver_offset = np.r_[0.0, 0.0, float(config["data"]["receiver_height_offset_m"])]
    train_survey = _survey(xyz[train] + receiver_offset)
    floor = max(
        float(config["uncertainty"]["absolute_floor_mgal"]),
        float(config["uncertainty"]["fraction_of_maximum"])
        * np.max(np.abs(observed_simpeg[train])),
    )
    if uncertainty_floor_override is not None:
        floor = float(uncertainty_floor_override)
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("uncertainty_floor_override must be finite and positive")
    standard_deviation = np.full(train.size, floor)
    data_object = data.Data(
        train_survey,
        dobs=observed_simpeg[train],
        standard_deviation=standard_deviation,
    )

    model_map = maps.IdentityMap(nP=int(active.sum()))
    simulation = gravity.simulation.Simulation3DIntegral(
        survey=train_survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=active,
        engine="choclo",
        store_sensitivities="ram",
    )
    misfit = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
    reference_model = np.zeros(int(active.sum()))
    reg = regularization.WeightedLeastSquares(
        mesh,
        active_cells=active,
        length_scale_x=1.0,
        length_scale_y=1.0,
        length_scale_z=1.0,
        reference_model=reference_model,
        reference_model_in_smooth=False,
    )
    inv_cfg = config["inversion"]
    opt = optimization.ProjectedGNCG(
        maxIter=int(inv_cfg["max_iterations"]),
        lower=float(inv_cfg["density_lower_gcc"]),
        upper=float(inv_cfg["density_upper_gcc"]),
        maxIterLS=20,
        cg_maxiter=int(inv_cfg["cg_max_iterations"]),
        cg_rtol=float(inv_cfg["cg_relative_tolerance"]),
        tolF=1e-8,
        tolX=1e-8,
        tolG=1e-8,
    )
    inv_problem = inverse_problem.BaseInvProblem(misfit, reg, opt)
    save_history = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list = [
        directives.UpdateSensitivityWeights(every_iteration=False),
        directives.UpdatePreconditioner(update_every_iteration=True),
        directives.BetaEstimate_ByEig(beta0_ratio=float(inv_cfg["beta0_ratio"])),
        directives.BetaSchedule(
            coolingFactor=float(inv_cfg["cooling_factor"]), coolingRate=1
        ),
        directives.TargetMisfit(chifact=float(inv_cfg["target_chi_factor"])),
        save_history,
    ]
    recovered = inversion.BaseInversion(inv_problem, directive_list).run(reference_model)
    predicted_train_simpeg = simulation.dpred(recovered)

    holdout_survey = _survey(xyz[holdout] + receiver_offset)
    holdout_simulation = gravity.simulation.Simulation3DIntegral(
        survey=holdout_survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=active,
        engine="choclo",
        store_sensitivities="forward_only",
    )
    predicted_holdout_simpeg = holdout_simulation.dpred(recovered)
    predicted_train_field = predicted_train_simpeg / sign_to_simpeg
    predicted_holdout_field = predicted_holdout_simpeg / sign_to_simpeg
    sensitivity = np.sqrt(np.maximum(simulation.getJtJdiag(recovered), 0.0))

    full_model = np.full(mesh.n_cells, np.nan)
    full_model[active] = recovered
    full_sensitivity = np.full(mesh.n_cells, np.nan)
    full_sensitivity[active] = sensitivity
    return {
        "config": config,
        "frame": frame,
        "xyz": xyz,
        "observed_field": observed_field,
        "observed_simpeg": observed_simpeg,
        "sign_to_simpeg": sign_to_simpeg,
        "train": train,
        "holdout": holdout,
        "mesh": mesh,
        "topography": topography,
        "topography_source": mesh_manifest.get("topography_source", expected_topography_source),
        "active": active,
        "mesh_manifest": mesh_manifest,
        "mesh_source": mesh_source,
        "model_active": recovered,
        "model_full": full_model,
        "sensitivity_active": sensitivity,
        "sensitivity_full": full_sensitivity,
        "predicted_train_simpeg": predicted_train_simpeg,
        "predicted_holdout_simpeg": predicted_holdout_simpeg,
        "predicted_train_field": predicted_train_field,
        "predicted_holdout_field": predicted_holdout_field,
        "uncertainty_floor": floor,
        "phi_d": float(inv_problem.phi_d),
        "history": {
            "beta": np.asarray(save_history.beta),
            "phi_d": np.asarray(save_history.phi_d),
            "phi_m": np.asarray(save_history.phi_m),
            "objective": np.asarray(save_history.phi),
        },
    }


def save_result(result: dict[str, object], root: str | Path) -> Path:
    """Save numerical results needed for later common-grid projection."""
    root = Path(root)
    mesh = result["mesh"]
    output = root / "outputs/gravity/gravity_inversion.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cell_centers=mesh.cell_centers,
        mesh_shape=np.asarray(mesh.shape_cells),
        mesh_origin=mesh.origin,
        active=result["active"],
        model_active_gcc=result["model_active"],
        model_full_gcc=result["model_full"],
        sensitivity_active=result["sensitivity_active"],
        sensitivity_full=result["sensitivity_full"],
        train_indices=result["train"],
        holdout_indices=result["holdout"],
        observed_field_mgal=result["observed_field"],
        observed_simpeg_mgal=result["observed_simpeg"],
        sign_to_simpeg=result["sign_to_simpeg"],
        predicted_train_field_mgal=result["predicted_train_field"],
        predicted_holdout_field_mgal=result["predicted_holdout_field"],
        predicted_train_simpeg_mgal=result["predicted_train_simpeg"],
        predicted_holdout_simpeg_mgal=result["predicted_holdout_simpeg"],
        uncertainty_floor_mgal=result["uncertainty_floor"],
        phi_d=result["phi_d"],
        history_beta=result["history"]["beta"],
        history_phi_d=result["history"]["phi_d"],
        history_phi_m=result["history"]["phi_m"],
        history_objective=result["history"]["objective"],
    )
    export_inverted_model(
        output,
        mesh,
        result["model_full"],
        "density_contrast_gcc",
        active=result["active"],
        extra_cell_data={"sensitivity": result["sensitivity_full"]},
    )
    return output
