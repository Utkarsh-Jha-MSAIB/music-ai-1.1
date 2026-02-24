from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import sys
import json
import time
import io
import zipfile
import numpy as np
import soundfile as sf

# For analytics (fast-ish)
import librosa
import os
os.environ["MPLBACKEND"] = "Agg"

# Ensure `src.*` and `configs.*` imports work
ROOT = Path(__file__).resolve().parents[1]      # C:\Music_AI_1.1
sys.path.insert(0, str(ROOT / "core"))          # makes core/src importable as `src`

from configs.load_config import load_yaml
from core.inference import MusicGenerator, GenerateRequest


app = FastAPI(title="Music AI API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gen = MusicGenerator()


def _cfg():
    cfg_file = ROOT / "configs" / "music_generator.yaml"
    return load_yaml(cfg_file)


def _runs_base_dir(cfg: dict) -> Path:
    out_dir = cfg["paths"].get("outputs_dir", "outputs")
    run_subdir = cfg.get("io", {}).get("run_subdir", "api_runs")
    return (ROOT / out_dir / run_subdir).resolve()


def _safe_join(base: Path, *parts: str) -> Path:
    p = (base / Path(*parts)).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _list_run_files(run_dir: Path) -> dict:
    files = {}
    for p in run_dir.glob("*"):
        if p.is_file():
            files[p.name] = f"/runs/{run_dir.name}/files/{p.name}"
    return files


def _pick_default_wav(files: dict) -> str | None:
    if not files:
        return None
    keys = list(files.keys())
    def pick(substr: str):
        for k in keys:
            if k.lower().endswith(".wav") and substr in k.lower():
                return k
        return None
    return (
        pick("complete_arrangement")
        or pick("final")
        or pick("complete")
        or pick("reference")
        or next((k for k in keys if k.lower().endswith(".wav")), None)
    )

def _calculate_scores(f0: np.ndarray, rms_norm: np.ndarray) -> dict:
    """
    KPI summary that matches the *plots*:

    - AvgLoudness_100: mean of loudness curve points (rms_norm) scaled to 0..100
    - Loudness_CV: coefficient of variation of loudness curve (std/mean) over "active" frames
    - MedianPitch_Hz: median of pitch curve points (Hz) over voiced frames

    Inputs:
    - rms_norm: normalized RMS used in the UI plot (0..1)
    - f0: pitch array in Hz (may contain zeros or small values)
    """
    rms_norm = np.asarray(rms_norm, dtype=np.float32)
    f0 = np.asarray(f0, dtype=np.float32)

    # --- loudness active frames (avoid silence dominating) ---
    if rms_norm.size:
        peak = float(np.max(rms_norm)) if np.isfinite(np.max(rms_norm)) else 0.0
        thr = 0.01 * peak  # 1% of peak loudness
        active = rms_norm > thr
        x = rms_norm[active] if np.any(active) else rms_norm
    else:
        x = rms_norm

    # Avg loudness on the SAME curve you plot (scaled 0..100)
    if x.size:
        mean_l = float(np.mean(x))
        std_l = float(np.std(x))
    else:
        mean_l = 0.0
        std_l = 0.0

    avg_loud_100 = float(np.clip(mean_l * 100.0, 0.0, 100.0))

    # Loudness CV = std/mean (unitless). If mean ~ 0, define as 0 to avoid explosions.
    if mean_l > 1e-8:
        loud_cv = float(std_l / mean_l)
    else:
        loud_cv = 0.0

    # --- pitch median (voiced only) ---
    voiced = f0[f0 > 40.0]  # simple voiced threshold
    median_pitch = float(np.median(voiced)) if voiced.size else 0.0

    return {
        "AvgLoudness_100": avg_loud_100,
        "Loudness_CV": loud_cv,
        "MedianPitch_Hz": median_pitch,
    }


def _downsample_series(y: np.ndarray, t: np.ndarray, max_points: int = 1400):
    if y is None or t is None or len(y) == 0 or len(t) == 0:
        return {"y": [], "t": []}
    n = min(len(y), len(t))
    if n <= max_points:
        return {"y": y[:n].tolist(), "t": t[:n].tolist()}
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return {"y": y[idx].tolist(), "t": t[idx].tolist()}


class GenerateBody(BaseModel):
    seconds: float = 20.0
    start_index: int = 1855


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/readiness")
def readiness():
    cfg = _cfg()
    candidates = [ROOT / p for p in cfg["paths"]["features_candidates"]]
    features_path = next((p for p in candidates if p.exists()), None)

    if features_path is None:
        return {"ready": False, "features_path": None, "missing": ["features folder not found"], "corrupt": {}}

    leader = cfg["instruments"]["leader"]
    add_drums = bool(cfg["instruments"].get("add_drums", False))
    models = list(cfg["checkpoints"]["models"].keys())
    req = cfg["validation"]["require_files"]

    needed = []
    needed += [s.format(leader=leader) for s in req["leader"]]
    for inst in models:
        needed += req[inst]
    if add_drums:
        needed += req["Drums"]

    missing = []
    corrupt = {}

    for name in needed:
        p = features_path / name
        if not p.exists():
            missing.append(str(p))
            continue
        # lightweight npy check (header + file size sanity)
        try:
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            expected_bytes = int(np.prod(arr.shape) * arr.dtype.itemsize)
            actual_bytes = p.stat().st_size
            if actual_bytes < expected_bytes:
                corrupt[str(p)] = f"TRUNCATED: expected >= {expected_bytes} bytes, got {actual_bytes}"
        except Exception as e:
            corrupt[str(p)] = f"UNREADABLE: {e}"

    return {
        "ready": (len(missing) == 0 and len(corrupt) == 0),
        "features_path": str(features_path),
        "missing": missing,
        "corrupt": corrupt,
    }


@app.post("/generate")
def generate_run(body: GenerateBody):
    try:
        req = GenerateRequest(seconds=body.seconds, start_index=body.start_index)
        bundle = gen.generate(req)

        run_id = bundle["run_id"]
        cfg = _cfg()
        run_dir = _safe_join(_runs_base_dir(cfg), run_id)

        # Expand meta for the dashboard: what the run actually used
        meta = {
            "requested_seconds": body.seconds,
            "requested_start_index": body.start_index,
            "actual_seconds": bundle.get("actual_seconds") or bundle.get("actual_length_seconds"),
            "runtime_sec": bundle.get("runtime_sec"),
            "requested_start_index": bundle.get("requested_start_index"),
            "runtime_sec": bundle.get("runtime_sec"),
            # snapshot of config (nice for UI)
            "paths": cfg.get("paths", {}),
            "checkpoints": cfg.get("checkpoints", {}),
            "instruments": cfg.get("instruments", {}),
            "generation": cfg.get("generation", {}),
            "audio": cfg.get("audio", {}),
            "mix": cfg.get("mix", {}),
            "io": cfg.get("io", {}),
        }
        (run_dir / "meta_dashboard.json").write_text(json.dumps(meta, indent=2))

        files = _list_run_files(run_dir)

        return {
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_dir.stat().st_mtime)),
            "runtime_sec": bundle.get("runtime_sec"),
            "meta": meta,
            "files": files,
            "download_zip_url": f"/runs/{run_id}/download.zip",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs")
def list_runs():
    cfg = _cfg()
    base = _runs_base_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue

        meta_dash = _read_json(d / "meta_dashboard.json") or {}
        meta_pm = _read_json(d / "meta.json") or {}

        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime))

        runs.append({
            "run_id": d.name,
            "created_at": created_at,
            "seconds": meta_dash.get("actual_seconds") or meta_pm.get("seconds"),
            "start_idx": meta_dash.get("requested_start_index") or meta_pm.get("start_idx"),
            "leader": (meta_dash.get("instruments", {}) or {}).get("leader") or meta_pm.get("leader"),
        })

    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    cfg = _cfg()
    run_dir = _safe_join(_runs_base_dir(cfg), run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    files = _list_run_files(run_dir)

    meta_dash_path = run_dir / "meta_dashboard.json"
    meta_pm_path = run_dir / "meta.json"

    meta_dash = _read_json(meta_dash_path)
    meta_pm = _read_json(meta_pm_path) or {}

    # Ensure requested fields exist (UI expects these keys)
    if meta_pm:
        if not meta_dash.get("requested_seconds"):
            meta_dash["requested_seconds"] = meta_pm.get("seconds")
        if not meta_dash.get("actual_seconds"):
            meta_dash["actual_seconds"] = meta_pm.get("seconds")
        if meta_dash.get("requested_start_index") is None:
            meta_dash["requested_start_index"] = meta_pm.get("start_idx")

    # Save back so older runs get “healed”
    meta_dash_path.write_text(json.dumps(meta_dash, indent=2))

    # If this run predates meta_dashboard.json, synthesize it once and save it.
    if not meta_dash:
        # Start from current YAML snapshot so we always have audio/generation/mix/instruments
        meta_dash = {
            "requested_seconds": meta_pm.get("seconds"),
            "actual_seconds": meta_pm.get("seconds"),
            "requested_start_index": meta_pm.get("start_idx"),
            "runtime_sec": None,

            "paths": cfg.get("paths", {}),
            "checkpoints": cfg.get("checkpoints", {}),
            "instruments": cfg.get("instruments", {}),
            "generation": cfg.get("generation", {}),
            "audio": cfg.get("audio", {}),
            "mix": cfg.get("mix", {}),
            "io": cfg.get("io", {}),
        }

        # Override with what the run actually used (from meta.json)
        if "leader" in meta_pm:
            meta_dash.setdefault("instruments", {})
            meta_dash["instruments"]["leader"] = meta_pm["leader"]
        if "seed" in meta_pm:
            meta_dash.setdefault("generation", {})
            meta_dash["generation"]["seed"] = meta_pm["seed"]
        if "device" in meta_pm:
            meta_dash.setdefault("generation", {})
            meta_dash["generation"]["device"] = meta_pm["device"]

        meta_dash_path.write_text(json.dumps(meta_dash, indent=2))

    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_dir.stat().st_mtime)),
        "meta": meta_dash,
        "files": files,
        "download_zip_url": f"/runs/{run_id}/download.zip",
        "default_track": _pick_default_wav(files),

        # Optional: expose runtime directly too (your App.jsx looks for it)
        "runtime_sec": meta_dash.get("runtime_sec"),
    }

@app.get("/runs/{run_id}/files/{filename}")
def run_file(run_id: str, filename: str):
    cfg = _cfg()
    run_dir = _safe_join(_runs_base_dir(cfg), run_id)
    p = _safe_join(run_dir, filename)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    name = p.name.lower()
    if name.endswith(".wav"):
        return FileResponse(p, media_type="audio/wav", filename=p.name)
    if name.endswith(".json"):
        return FileResponse(p, media_type="application/json", filename=p.name)
    if name.endswith(".png"):
        return FileResponse(p, media_type="image/png", filename=p.name)
    return FileResponse(p, filename=p.name)


@app.get("/runs/{run_id}/download.zip")
def run_zip(run_id: str):
    cfg = _cfg()
    run_dir = _safe_join(_runs_base_dir(cfg), run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.glob("*"):
            if p.is_file():
                z.write(p, arcname=p.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'},
    )


@app.get("/runs/{run_id}/analysis")
def run_analysis(
    run_id: str,
    file: str | None = Query(default=None, description="wav filename inside the run directory"),
):
    cfg = _cfg()
    run_dir = _safe_join(_runs_base_dir(cfg), run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    files = _list_run_files(run_dir)

    wav_name = file or _pick_default_wav(files)
    if not wav_name:
        raise HTTPException(status_code=404, detail="No wav file available for analysis")

    if not wav_name.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="analysis supports .wav files only")

    wav_path = _safe_join(run_dir, wav_name)
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {wav_name}")

    target_sr = int(cfg.get("audio", {}).get("sample_rate", 16000))
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)

    duration = float(librosa.get_duration(y=y, sr=sr))

    # Waveform proxy: abs amplitude downsampled
    wave = np.abs(y).astype(np.float32, copy=False)
    t_wave = np.linspace(0, duration, num=len(wave), endpoint=False)

    # RMS envelope
    frame_len = 2048
    hop = int(cfg.get("audio", {}).get("hop_samples", 64))

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0].astype(np.float32, copy=False)
    rms_norm = rms / (float(np.max(rms)) + 1e-7)
    t_rms = librosa.times_like(rms, sr=sr, hop_length=hop).astype(np.float32, copy=False)

    # Pitch
    f0 = librosa.yin(y, fmin=40, fmax=2000, sr=sr, frame_length=frame_len, hop_length=hop)
    f0 = np.nan_to_num(f0).astype(np.float32, copy=False)
    t_f0 = librosa.times_like(f0, sr=sr, hop_length=hop).astype(np.float32, copy=False)

    # ✅ IMPORTANT: pass raw rms, not rms_norm; remove extra kwargs
    scores = _calculate_scores(f0, rms_norm)

    return {
      "file": wav_name,
      "duration": duration,
      "scores": scores,
      "wave": _downsample_series(wave, t_wave, max_points=1600),
      "rms": _downsample_series(rms_norm, t_rms, max_points=1400),
      "f0": _downsample_series(f0, t_f0, max_points=1400),
    }
