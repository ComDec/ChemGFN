import os
import sys
from pathlib import Path

def find_root(indicator: str = ".project-root") -> Path:
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / indicator).exists():
            return parent
    raise FileNotFoundError(f"Indicator {indicator} not found from {path}")

def setup_root(__file__, indicator: str = ".project-root", pythonpath: bool = False) -> Path:
    root = find_root(indicator)
    if pythonpath and str(root) not in sys.path:
        sys.path.append(str(root))
    os.environ.setdefault("PROJECT_ROOT", str(root))
    return root

__all__ = ["find_root", "setup_root"]
