"""The SELDnet manifest in the package's config/weights.yaml: classes, training array, and the checkpoint and scaler fetched into the data dir."""

from __future__ import annotations

import hashlib
import os
import typing
from pathlib import Path

from arena_auditory.shared import ArraySpec, load_array_spec

if typing.TYPE_CHECKING:
    from arena_auditory.assets import SoundLibrary


def data_dir() -> Path:
    """Raises RuntimeError when ARENA_DATA_DIR is unset."""
    root = os.environ.get("ARENA_DATA_DIR")
    if not root:
        raise RuntimeError("ARENA_DATA_DIR is not set, source arena or export it to the Arena data directory")
    return Path(root) / "auditory" / "seld"


def _load() -> dict:
    import yaml
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory("arena_auditory")) / "config" / "weights.yaml"
    with open(path) as f:
        return dict(yaml.safe_load(f) or {})


def manifest() -> list[dict]:
    return list(_load().get("files", []))


def classes(library: SoundLibrary) -> tuple[str, ...]:
    """SELDnet output classes in model order. Raises ValueError unless each is a detect kind of library."""
    names = tuple(str(name) for name in _load().get("classes", []))
    detect = {name for name, kind in library.kinds().items() if kind.detect}
    unknown = [name for name in names if name not in detect]
    if not names or unknown:
        raise ValueError(f"weights.yaml classes {list(names)} must be detect kinds of the sound library {sorted(detect)}")
    return names


def array_spec() -> ArraySpec:
    """The microphone array the weights were trained on, channel order included."""
    return load_array_spec(str(_load()["array"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(root: Path | None = None) -> dict[str, str]:
    """Return ``{role: local path}`` for every manifest entry, downloading what is missing from Hugging Face."""
    root = root or data_dir()
    root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for entry in manifest():
        dest = root / entry["dest"]
        if not dest.is_file():
            from huggingface_hub import hf_hub_download

            cached = Path(hf_hub_download(repo_id=entry["repo"], filename=entry["filename"]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            dest.symlink_to(cached)
        actual = _sha256(dest)
        if actual != entry["sha256"]:
            raise RuntimeError(f"{dest}: sha256 {actual} does not match weights.yaml ({entry['sha256']})")
        resolved[entry["role"]] = str(dest)
    return resolved


def main() -> None:
    for role, path in ensure().items():
        print(f"{role}: {path}")
