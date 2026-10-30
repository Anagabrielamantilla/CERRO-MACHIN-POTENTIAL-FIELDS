"""Reproducible smooth 3-D magnetic inversion on the common MANTLE-3D mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from simpeg import data, data_misfit, directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg.potential_fields import magnetics

from .common_mesh import load_common_mesh
from .exports import export_inverted_model
from .gravity import spatial_holdout


def load_config(root: str | Path) -> dict:
    return yaml.safe_load((Path(root) / "configs/magnetics.yaml").read_text())


def load_magnetics(root: str | Path, config: dict) -> pd.DataFrame:
    frame = pd.read_csv(Path(root) / config["data"]["path"], encoding="utf-8-sig")
    cols = ["X", "Y", "Z", "Anomalia (nT)"]
    return frame[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def _survey(locations: np.ndarray, field: dict) -> magnetics.survey.Survey:
    receiver = magnetics.receivers.Point(locations, components="tmi")
    source = magnetics.sources.UniformBackgroundField(
        receiver_list=[receiver], amplitude=float(field["amplitude_nt"]),
        inclination=float(field["inclination_deg"]), declination=float(field["declination_deg"]),
    )
    return magnetics.survey.Survey(source)


def run_inversion(root: str | Path, observed_override: np.ndarray | None = None) -> dict[str, object]:
    root, config = Path(root), load_config(root)
    frame = load_magnetics(root, config)
    xyz = frame[["X", "Y", "Z"]].to_numpy(float)
    observed = frame["Anomalia (nT)"].to_numpy(float)
    if observed_override is not None:
        observed = np.asarray(observed_override, dtype=float)
        if observed.shape != (len(frame),) or not np.isfinite(observed).all():
            raise ValueError("observed_override must contain one finite value per magnetic station")
    train, holdout = spatial_holdout(xyz, int(config["validation"]["holdout_stride"]))
    mesh, topography, active, manifest = load_common_mesh(root)
    offset = np.r_[0.0, 0.0, float(config["data"]["receiver_height_offset_m"])]
    field = config["inducing_field"]
    survey = _survey(xyz[train] + offset, field)
    floor = max(float(config["uncertainty"]["absolute_floor_nt"]),
                float(config["uncertainty"]["fraction_of_maximum"]) * np.max(np.abs(observed[train])))
    data_object = data.Data(survey, dobs=observed[train], standard_deviation=np.full(train.size, floor))
    model_map = maps.IdentityMap(nP=int(active.sum()))
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey, mesh=mesh, chiMap=model_map, active_cells=active,
        model_type="scalar", engine="choclo", store_sensitivities="ram",
    )
    misfit = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
    # The inversion parameter is absolute/effective magnetic susceptibility, not
    # a signed contrast. Zero is the non-negative background/reference model;
    # positivity is enforced by ProjectedGNCG below.
    reference = np.zeros(int(active.sum()))
    reg = regularization.WeightedLeastSquares(
        mesh, active_cells=active, length_scale_x=1.0, length_scale_y=1.0,
        length_scale_z=1.0, reference_model=reference, reference_model_in_smooth=False,
    )
    cfg = config["inversion"]
    lower = float(cfg["effective_susceptibility_lower_si"])
    upper = float(cfg["effective_susceptibility_upper_si"])
    if lower < 0.0 or upper <= lower:
        raise ValueError("Magnetic susceptibility bounds must satisfy 0 <= lower < upper")
    initial_value = float(cfg["initial_susceptibility_si"])
    if not lower <= initial_value <= upper:
        raise ValueError("Initial magnetic susceptibility must lie within the inversion bounds")
    starting_model = np.full(int(active.sum()), initial_value)
    opt = optimization.ProjectedGNCG(
        maxIter=int(cfg["max_iterations"]), lower=lower, upper=upper, maxIterLS=20,
        cg_maxiter=int(cfg["cg_max_iterations"]), cg_rtol=float(cfg["cg_relative_tolerance"]),
        tolF=1e-8, tolX=1e-8, tolG=1e-8,
    )
    problem = inverse_problem.BaseInvProblem(misfit, reg, opt)
    history = directives.SaveOutputEveryIteration(on_disk=False)
    directive_list = [
        directives.UpdateSensitivityWeights(every_iteration=False),
        directives.UpdatePreconditioner(update_every_iteration=True),
        directives.BetaEstimate_ByEig(beta0_ratio=float(cfg["beta0_ratio"])),
        directives.BetaSchedule(coolingFactor=float(cfg["cooling_factor"]), coolingRate=1),
        directives.TargetMisfit(chifact=float(cfg["target_chi_factor"])), history,
    ]
    recovered = inversion.BaseInversion(problem, directive_list).run(starting_model)
    predicted_train = simulation.dpred(recovered)
    holdout_survey = _survey(xyz[holdout] + offset, field)
    holdout_simulation = magnetics.simulation.Simulation3DIntegral(
        survey=holdout_survey, mesh=mesh, chiMap=model_map, active_cells=active,
        model_type="scalar", engine="choclo", store_sensitivities="forward_only",
    )
    predicted_holdout = holdout_simulation.dpred(recovered)
    sensitivity = np.sqrt(np.maximum(simulation.getJtJdiag(recovered), 0.0))
    full_model, full_sensitivity = np.full(mesh.n_cells, np.nan), np.full(mesh.n_cells, np.nan)
    full_model[active], full_sensitivity[active] = recovered, sensitivity
    return dict(config=config, frame=frame, xyz=xyz, observed=observed, train=train, holdout=holdout,
                mesh=mesh, topography=topography, active=active, mesh_manifest=manifest,
                model_active=recovered, model_full=full_model, sensitivity_active=sensitivity,
                sensitivity_full=full_sensitivity, predicted_train=predicted_train,
                predicted_holdout=predicted_holdout, uncertainty_floor=floor, phi_d=float(problem.phi_d),
                history={"beta": np.asarray(history.beta), "phi_d": np.asarray(history.phi_d),
                         "phi_m": np.asarray(history.phi_m), "objective": np.asarray(history.phi)})


def save_result(result: dict[str, object], root: str | Path) -> Path:
    output = Path(root) / "outputs/magnetics/magnetic_inversion.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = result["mesh"]
    np.savez_compressed(output, cell_centers=mesh.cell_centers, mesh_shape=np.asarray(mesh.shape_cells),
        mesh_origin=mesh.origin, active=result["active"], model_active_effective_susceptibility_si=result["model_active"],
        model_full_effective_susceptibility_si=result["model_full"],
        sensitivity_active=result["sensitivity_active"], sensitivity_full=result["sensitivity_full"],
        train_indices=result["train"], holdout_indices=result["holdout"],
        observed_nt=result["observed"], predicted_train_nt=result["predicted_train"],
        predicted_holdout_nt=result["predicted_holdout"], uncertainty_floor_nt=result["uncertainty_floor"],
        phi_d=result["phi_d"], history_beta=result["history"]["beta"], history_phi_d=result["history"]["phi_d"],
        history_phi_m=result["history"]["phi_m"], history_objective=result["history"]["objective"])
    export_inverted_model(
        output,
        mesh,
        result["model_full"],
        "effective_susceptibility_si",
        active=result["active"],
        extra_cell_data={"sensitivity": result["sensitivity_full"]},
    )
    return output
