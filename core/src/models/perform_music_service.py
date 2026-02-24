from __future__ import annotations

from pathlib import Path

import torch

from configs.load_config import load_yaml
from src.models import perform_music as pm  # ✅ NOT core.src.models


# Coarse API guardrail (as requested)
MIN_START_INDEX = 0
MAX_START_INDEX = 32000


def _find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find configs/music_generator.yaml.
    Avoid brittle .parents[N] assumptions.
    """
    for p in [start] + list(start.parents):
        if (p / "configs" / "music_generator.yaml").exists():
            return p
    raise FileNotFoundError(
        "Could not locate repo root (missing configs/music_generator.yaml). "
        f"Started from: {start}"
    )


def _resolve_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_arrangement_for_api(seconds: float = 20.0, start_index: int = 4000) -> str:
    """
    DB-only generation. No upload path.

    API-exposed knobs:
      - seconds
      - start_index (guardrailed to [0, 32000])

    Everything else is driven by configs/music_generator.yaml.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())

    # --- load YAML ---
    cfg_file = repo_root / "configs" / "music_generator.yaml"
    cfg = load_yaml(cfg_file)

    # --- API overrides (only these two) ---
    cfg.setdefault("generation", {})
    cfg["generation"]["seconds"] = float(seconds)

    # Guardrail start_index: clamp to [0, 32000]
    start_index = int(start_index)
    start_index = max(MIN_START_INDEX, min(start_index, MAX_START_INDEX))
    cfg["generation"]["manual_start_index"] = start_index

    # Keep API outputs separate (optional, but nice)
    cfg.setdefault("io", {})
    cfg["io"]["run_subdir"] = "api_runs"

    # --- resolve features folder ---
    features_candidates = cfg.get("paths", {}).get("features_candidates", [])
    candidates = [repo_root / p for p in features_candidates]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Could not find features folder. Looked in:\n" + "\n".join(str(p) for p in candidates)
        )

    # --- determine required instruments ---
    leader = cfg.get("instruments", {}).get("leader", "Guitar")
    add_drums = bool(cfg.get("instruments", {}).get("add_drums", False))

    # Prefer YAML followers list; fallback to checkpoint keys
    followers = cfg.get("instruments", {}).get("arranger_followers") or list(
        cfg.get("checkpoints", {}).get("models", {}).keys()
    )

    # --- validate required npy files using YAML templates ---
    req = cfg.get("validation", {}).get("require_files", {})
    needed_files: list[str] = []

    leader_templates = req.get("leader", [])
    needed_files += [s.format(leader=leader) for s in leader_templates]

    for inst in followers:
        if inst not in req:
            raise KeyError(
                f"Missing validation.require_files entry for instrument '{inst}'. "
                f"Add it to YAML under validation.require_files."
            )
        needed_files += list(req[inst])

    if add_drums:
        if "Drums" not in req:
            raise KeyError(
                "YAML has instruments.add_drums=true but validation.require_files.Drums is missing."
            )
        needed_files += list(req["Drums"])

    missing = [f for f in needed_files if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required feature files in {data_path}:\n" + "\n".join(missing)
        )

    # --- resolve device at runtime ---
    dev = _resolve_device(cfg["generation"].get("device", "auto"))

    # --- patch perform_music module state (explicit + minimal) ---
    project_root = (repo_root / cfg.get("paths", {}).get("project_root", ".")).resolve()
    ckpt_dir = project_root / cfg.get("paths", {}).get("checkpoints_dir", "core/checkpoints")

    pm.cfg = cfg
    pm.project_root = project_root

    pm.LEADER = leader
    pm.ADD_DRUMS = add_drums
    pm.MANUAL_START_INDEX = int(cfg["generation"]["manual_start_index"])
    pm.GENERATE_SECONDS = float(cfg["generation"]["seconds"])

    pm.ckpt_dir = ckpt_dir
    pm.ARRANGER_CKPT = ckpt_dir / cfg.get("checkpoints", {}).get("arranger", "arranger.ckpt")
    pm.MODEL_REGISTRY = {
        k: str(ckpt_dir / v) for k, v in cfg.get("checkpoints", {}).get("models", {}).items()
    }

    # --- run and return output path ---
    out_path = pm.run_complete_arrangement(root=pm.project_root, dev=dev)
    return str(out_path)
