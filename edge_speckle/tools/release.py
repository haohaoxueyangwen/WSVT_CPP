"""Create a source-only distribution with a per-file manifest; exclude data/binaries."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in ('__pycache__', 'build', 'dist', 'runs', '.git', '_native', 'results') or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.suffix in ('.so', '.dll', '.pyc', '.npz', '.npy', '.tif', '.tiff', '.h5', '.zip'):
            continue
        if path.name != 'release_manifest.json':
            yield path


def main():
    files = list(source_files())
    manifest = dict(package_version='0.1.0', distribution='source only; native binaries are built on target',
                    files_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    manifest_path = ROOT / 'docs/release_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    destination = ROOT / 'dist'
    destination.mkdir(exist_ok=True)
    archive = destination / 'speckle_edge-0.1.0-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files + [manifest_path]:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print(json.dumps(dict(archive=str(archive), file_count=len(files) + 1, bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
