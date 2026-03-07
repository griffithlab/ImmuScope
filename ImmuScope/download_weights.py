"""Download and install ImmuScope model weights.

This is intentionally implemented with stdlib-only dependencies so users can run
it right after `pip install immuscope`.

Default install location follows the XDG base directory spec:
- $XDG_DATA_HOME/ImmuScope/weights
- ~/.local/share/ImmuScope/weights

The ImmuScope wrapper supports overriding the weights directory via:
- `--weights-dir`
- `IMMU_SCOPE_WEIGHTS_DIR` environment variable
"""

from __future__ import annotations
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

DEFAULT_URL = (
    "https://zenodo.org/records/14810445/files/ImmuScope-weights.tar.gz?download=1"
)


def default_destination_dir() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else (Path.home() / ".local" / "share")
    return base / "ImmuScope" / "weights"


def _weights_already_installed(dest: Path) -> bool:
    return (dest / "IM").is_dir()


def _download_to_file(url: str, out_path: Path) -> None:
    with urlopen(url) as resp, open(out_path, "wb") as out_fh:
        shutil.copyfileobj(resp, out_fh)


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract a tarfile into path (prevents path traversal)."""

    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path) + os.sep) and member_path != path:
            raise RuntimeError(f"Unsafe tar member path: {member.name}")
    tar.extractall(path=path)


def _find_extracted_weights_root(extract_dir: Path) -> Path:
    """Find a directory under extract_dir that looks like the weights root.

    Expected structure: <root>/IM/<checkpoint>.pt
    """

    candidates = [extract_dir]
    candidates.extend([p for p in extract_dir.iterdir() if p.is_dir()])

    for cand in candidates:
        if (cand / "IM").is_dir():
            return cand

    raise RuntimeError(
        "Downloaded archive did not contain an 'IM/' directory at the expected location. "
        "Please inspect the extracted contents and move them into the desired weights directory."
    )


def main() -> int:
    dest = default_destination_dir().expanduser().resolve()
    if _weights_already_installed(dest):
        print(f"ImmuScope weights already installed at: {dest}")
        return 0

    if dest.exists() and not _weights_already_installed(dest):
        print(f"Destination exists but doesn't look like an ImmuScope weights directory: {dest}")
        print("Please remove it and re-run immuscope-download-weights.")
        return 2

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="immuscope-weights-") as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / "ImmuScope-weights.tar.gz"
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading weights from: {DEFAULT_URL}")
        _download_to_file(DEFAULT_URL, archive_path)

        print("Extracting archive")
        with tarfile.open(archive_path, mode="r:gz") as tf:
            _safe_extract(tf, extract_dir)

        weights_root = _find_extracted_weights_root(extract_dir)

        print(f"Installing into: {dest}")
        shutil.copytree(weights_root, dest)

    print("Done")
    print(f"Weights installed at: {dest}")
    return 0
