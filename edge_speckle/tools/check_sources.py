"""Verify preserved origins and allow only documented packaging transformations."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    locations = p.add_mutually_exclusive_group(required=True)
    locations.add_argument('--project', type=Path, help='Root containing scratch/orin_cuda_20261010')
    locations.add_argument('--original-root', type=Path, help='Orin directory containing original wxst_orin.py/edge_adaptation_20261011')
    args = p.parse_args()
    entries = json.loads((ROOT / 'docs/source_origins.json').read_text())
    rows = []
    for entry in entries:
        origin = args.project / entry['original'] if args.project else args.original_root / entry['original'].split('orin_cuda_20261010/', 1)[1]
        raw = origin.read_bytes()
        transformed = raw
        for change in entry['changes']:
            transformed = transformed.replace(change['old'].encode(), change['new'].encode())
        packaged = (ROOT / entry['packaged']).read_bytes()
        row = dict(file=entry['packaged'], original_unchanged=hashlib.sha256(raw).hexdigest() == entry['original_sha256'],
                   packaged_matches_documented_transform=packaged == transformed)
        rows.append(row)
    report = dict(pass_check=all(r['original_unchanged'] and r['packaged_matches_documented_transform'] for r in rows), files=rows)
    print(json.dumps(report, indent=2))
    if not report['pass_check']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
