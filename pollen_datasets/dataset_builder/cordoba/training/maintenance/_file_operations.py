"""Verified file replacement shared by dataset maintenance commands."""

import hashlib
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_verified(source, staged, *, partial, backup, original_stat, root):
    """Copy, verify, archive, and replace; restore the original if rename fails."""
    root = root.resolve()
    if source.resolve().parent != partial.resolve().parent:
        raise ValueError("Replacement must be beside the source")
    if not all(path.resolve().is_relative_to(root) for path in (source, partial, backup)):
        raise ValueError("Replacement paths outside target folder")
    if partial.exists():
        raise FileExistsError(partial)
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copyfile(staged, partial)
    checksum = sha256(staged)
    if sha256(partial) != checksum:
        raise ValueError(f"Copy hash mismatch: {source.name}")
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
        raise ValueError(f"Source changed before replacement: {source}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    source.rename(backup)
    try:
        partial.rename(source)
    except BaseException:
        backup.rename(source)
        raise
    return checksum
