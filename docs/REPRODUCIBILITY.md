> Public edition: input observations and topography are withheld. Request the original data and geometry from the lead author before running inversions. Original provenance below describes the complete private archive.

# Reproduction and provenance notes

The source snapshot is MANTLE-3D commit `ae063f661a7be0b379d0103e543f1d9f4dd5d9d9`. Only independent gravity and magnetic code and their necessary geometry/export helpers were selected. The source repository was not modified.

## Changes to selected source code

- `gravity.run_inversion` always verifies and loads an existing common mesh; it does not rebuild an archived mesh in response to an absolute DEM-path mismatch. Mesh construction is still available when the mesh is absent.
- `magnetics.run_inversion` accepts an optional `observed_override` array with shape and finite-value validation, as required by the existing magnetic Monte Carlo script.
- Configurations and notebook prose use the included, relative topography path. The archived geometry and its hashes remain unchanged; the included corrected topography table is not asserted to be byte-identical to the historical external DEM.
- Package/environment descriptions cover independent potential fields; the torch extra is removed and a pyvista Monte Carlo extra is included. The Python import namespace stays `mantle3d` for compatibility.

These edits do not convert the current smooth gravity solver into the sparse historical implementation. The gravity Monte Carlo source driver, historical IRLS settings implementation, and an exact environment lockfile for the 2026-09-06 ensemble runs are unavailable. Neither independent inversion nor 1,000-run Monte Carlo ensembles were rerun when curating this repository.

## Local research outputs

Monte Carlo outputs originate from the author's `Downloads/monte_carlo/gravity` and `Downloads/monte_carlo/magnetic` folders. Their run summaries report 1,000 realizations each, seed 20260906, 71,390 active cells, noise sigma 0.7 mGal and 26.366116 nT, respectively. All per-run logs, summaries, method descriptions, NPY uncertainty cubes, and method figures are retained. Original full NPZ archives are stored as private release assets with exact checksums. Compact statistics NPZ archives omit only the full realization array; retained values are unchanged.

Additional SVG figures originate from `Downloads/Paraview_journal_imaging`. The combined convergence figure and statistics originate from `Documents/Codex/2026-09-25/en-x20/outputs`. Mixed-method interpretation figures and ParaView scene files with external/MT dependencies were excluded. The manuscript itself is not included because the requested collection is restricted to independent potential-field materials.

## Manuscript discrepancies

The supplied manuscript's Fig. 5 says 2,000 Monte Carlo gravity realizations; both available ensembles contain 1,000. The gravity Monte Carlo records describe sparse IRLS, depth weighting, and ±0.5 g/cm³ bounds; the source solver has smooth regularization and ±1 g/cm³ bounds. Keep archived outputs distinct from newly generated smooth-solver outputs until the historical implementation is recovered.

The archived combined convergence plot uses `running_m2 / i` for running variance while the ensemble sample variance uses `ddof=1`. This normalization mismatch affects comparisons; the reported convergence thresholds should be reassessed with a consistent estimator before a final paper revision.

## Empty VTK workspace

The `vtk/` directory has only a zero-byte `.gitkeep`, required to represent the directory in Git. No VTK datasets are included elsewhere in the initial collection. Export helpers may create VTK outputs during future inversion runs.

## Added gravity Monte Carlo runner

`scripts/run_gravity_monte_carlo.py` is a newly implemented driver for the current smooth gravity solver, with the same product schema as the archived gravity ensemble. It is not a recovered historical IRLS driver. The solver now accepts optional validated field-data and uncertainty-floor overrides. The runner freezes the unperturbed training-data uncertainty floor, preserves the spatial split, evaluates withheld predictions against original observations, and converts density to kg/m³ before ensemble statistics. It requires a new or empty output directory and defaults to `outputs/monte_carlo/gravity_new/`. Existing notebook calls retain their original behavior.

The field-only package initializer no longer imports the excluded inventory module. The new runner was checked with two real SimPEG realizations on a small synthetic mesh, including unit conversion, sample standard deviation, NPY ordering, fixed-seed noise, override validation, and rejection of an occupied destination. This smoke run used `--skip-vtk`; VTK export was not exercised. The full 1,000-realization field ensemble was not rerun.

Subsequent author uploads on GitHub added `vtk/MT.vtk`, `vtk/Magnetic.vtk`, and `vtk/gravity.vtk`. These files were preserved when integrating the gravity runner. The integrity manifest now includes them; the verifier no longer requires the VTK workspace to be empty.
