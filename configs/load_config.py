from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_project_root(cfg: dict, cfg_file: Path) -> Path:
    # Resolve relative to config file location, then normalize
    pr = Path(cfg["paths"].get("project_root", "."))
    if not pr.is_absolute():
        # config lives in core/configs/, so move up to repo root sensibly
        # adjust if your config location differs
        base = cfg_file.resolve().parents[2]  # repo root if cfg is core/configs/infer.yaml
        pr = (base / pr).resolve()
    return pr
