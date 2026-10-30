"""Retrieve the original Monte Carlo ensembles from the private GitHub release."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / 'docs/ensemble_assets.json').read_text())
    for asset in metadata['assets']:
        destination = ROOT / asset['destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            subprocess.run(['gh', 'release', 'download', metadata['release'],
                            '--repo', metadata['repository'], '--pattern', asset['filename'],
                            '--dir', str(destination.parent)], check=True)
        with destination.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != asset['sha256']:
            raise ValueError(f'Checksum mismatch: {destination}; remove this file before retrying')
        print(f'Verified {asset["filename"]}: {digest}')


if __name__ == '__main__':
    main()
