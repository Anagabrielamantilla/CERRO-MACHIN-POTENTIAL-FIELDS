"""Check archived data integrity, geometry, units, and independent validation splits."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / 'docs/archive_manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(digest == item['sha256'], f'Checksum mismatch: {path}')
    geometry = np.load(ROOT / 'outputs/common_mesh/common_mesh_geometry.npz', allow_pickle=False)
    active = geometry['active'].astype(bool)
    require(active.size == 90720 and active.sum() == 71390, 'Unexpected common mesh')
    for kind, stations, train_count, holdout_count in [('gravity', 187, 149, 38), ('magnetics', 165, 132, 33)]:
        path = next((ROOT / f'outputs/{kind}').glob('*inversion.npz'))
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z['mesh_shape'], [60, 54, 28]), f'{kind}: wrong shape')
            require(np.array_equal(z['active'], active), f'{kind}: active mask mismatch')
            train, holdout = z['train_indices'], z['holdout_indices']
            require(len(train) == train_count and len(holdout) == holdout_count, f'{kind}: split counts')
            require(np.intersect1d(train, holdout).size == 0, f'{kind}: validation leakage')
            require(np.array_equal(np.sort(np.r_[train, holdout]), np.arange(stations)), f'{kind}: incomplete split')
            model_key = 'model_full_gcc' if kind == 'gravity' else 'model_full_effective_susceptibility_si'
            require(np.isfinite(z[model_key][active]).all(), f'{kind}: nonfinite active model')
            if kind == 'magnetics':
                require((z[model_key][active] >= -1e-12).all(), 'Negative effective susceptibility')
            centers = z['cell_centers']
        mc_kind = 'gravity' if kind == 'gravity' else 'magnetic'
        with np.load(ROOT / f'outputs/monte_carlo/{mc_kind}/{mc_kind}_monte_carlo_statistics.npz', allow_pickle=False) as z:
            require(np.array_equal(z['active'], active), f'{kind}: MC active mask mismatch')
            require(np.allclose(z['cell_centers'], centers), f'{kind}: MC geometry mismatch')
            require(int(z['n_realizations']) == 1000 and int(z['seed']) == 20260906, f'{kind}: MC metadata')
            prefix, unit = ('density', 'kgm3') if kind == 'gravity' else ('susceptibility', 'si')
            require((z[f'{prefix}_std_{unit}'][active] >= 0).all(), f'{kind}: negative uncertainty')
            require((z[f'{prefix}_p05_{unit}'][active] <= z[f'{prefix}_p50_{unit}'][active]).all(), f'{kind}: percentile order')
            require((z[f'{prefix}_p50_{unit}'][active] <= z[f'{prefix}_p95_{unit}'][active]).all(), f'{kind}: percentile order')
        print(f'{kind}: data split, model, mesh, and Monte Carlo statistics verified')
    require((ROOT / 'vtk/.gitkeep').stat().st_size == 0, 'VTK placeholder must be empty')
    print(f'Archive integrity verified: {len(manifest["files"])} files')


if __name__ == '__main__':
    main()
