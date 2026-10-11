import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def implementation_receipt():
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob('*.py')) + sorted((root / '_native').glob('*.so'))
    receipt = {str(path.relative_to(root)): sha256(path) for path in files}
    build_file = root / '_native/build_info.json'
    return dict(runtime_files_sha256=receipt,
                native_build=json.loads(build_file.read_text()) if build_file.exists() else None)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
