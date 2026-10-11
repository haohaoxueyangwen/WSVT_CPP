from __future__ import annotations

from pathlib import Path


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


sha256 = file_sha256
sha256_file = file_sha256
sha = file_sha256

