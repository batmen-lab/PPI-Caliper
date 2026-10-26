"""Repository paths, independent of the caller's working directory."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def data_path(path: Path) -> str:
    """Return a portable data/ identifier for experiment input manifests."""
    return "data/" + path.relative_to(DATA_DIR).as_posix()


def set_data_root(root: Path) -> None:
    """Use downloaded/preprocessed data in an explicitly selected directory."""
    global DATA_DIR
    DATA_DIR = Path(root).resolve()
