# api/rag_routes.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import io
import json
import time
import zipfile
import numpy as np
import librosa
import os
import re
import sys
import shutil
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "core"))      # makes core/src importable as `src`

RAG_BASE = (ROOT / "outputs" / "rag_uploads").resolve()
RAG_BASE.mkdir(parents=True, exist_ok=True)


def _safe_join(base: Path, *parts: str) -> Path:
    p = (base / Path(*parts)).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


def _downsample_series(y: np.ndarray, t: np.ndarray, max_points: int = 1400):
    if y is None or t is None or len(y) == 0 or len(t) == 0:
        return {"y": [], "t": []}
    n = min(len(y), len(t))
    if n <= max_points:
        return {"y": y[:n].tolist(), "t": t[:n].tolist()}
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return {"y": y[idx].tolist(), "t": t[idx].tolist()}


def _calculate_scores(f0: np.ndarray, rms_norm: np.ndarray) -> dict:
    avg_energy = float(np.mean(rms_norm))
    energy_score = int(np.clip(avg_energy * 100, 0, 100))

    dyn = float(np.max(rms_norm) - np.min(rms_norm))
    dynamics_score = int(np.clip(dyn * 100, 0, 100))

    active = f0[f0 > 40]
    if active.size:
        pitch_std = float(np.std(active))
        complexity = int(np.clip((pitch_std / 100.0) * 100, 0, 100))
    else:
        complexity = 0

    return {"Energy": energy_score, "Dynamics": dynamics_score, "Complexity": complexity}


def _guess_sr_from_wav(wav_path: Path) -> int:
    # cheap: librosa load header-ish
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return int(info.samplerate)
    except Exception:
        return 16000


def _list_files(upload_dir: Path) -> dict:
    files = {}
    for p in upload_dir.glob("*"):
        if p.is_file():
            files[p.name] = f"/rag/{upload_dir.name}/files/{p.name}"
    return files


def _parse_rank_idx_score(name: str):
    """
    Supports:
      Rank1_IDX_9132_LEN_20.00s_Score_0.8421_extension.wav
      Rank1_IDX_9132_LEN_20.00s_Score_0.8421_stitched.wav
      Rank1_IDX_9132_LEN_20.00s_Score_0.8421.wav   (arranged)
      Rank1_full.wav
      Rank1_arranged.wav
    """
    m = re.search(
        r"Rank(\d+)_IDX_(\d+)_LEN_([^_]+)_Score_([0-9.]+)(?:_(extension|stitched))?\.wav$",
        name,
        re.IGNORECASE,
    )
    if m:
        tag = m.group(5)  # extension / stitched / None
        kind = tag.lower() if tag else "arranged"
        return {
            "rank": int(m.group(1)),
            "db_idx": int(m.group(2)), 
            "len_tag": m.group(3),
            "score": float(m.group(4)),
            "kind": kind,
        }

    # Non-scored artifacts from /render
    m2 = re.search(r"Rank(\d+)_(full|arranged)\.wav$", name, re.IGNORECASE)
    if m2:
        return {
            "rank": int(m2.group(1)),
            "db_id_4s": None,
            "len_tag": None,
            "score": None,
            "kind": m2.group(2).lower(),  # "full" or "arranged"
        }

    return None


class RagSearchBody(BaseModel):
    upload_id: str
    target_duration_sec: float = 20.0
    top_k: int = 5


@router.post("/upload")
async def rag_upload(file: UploadFile = File(...)):
    # hard reject non-wav (keep it simple)
    fname = (file.filename or "").lower()
    if not fname.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    upload_id = f"u_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    up_dir = _safe_join(RAG_BASE, upload_id)
    up_dir.mkdir(parents=True, exist_ok=True)

    in_path = up_dir / "input.wav"
    data = await file.read()
    in_path.write_bytes(data)

    # quick stats
    sr = _guess_sr_from_wav(in_path)
    try:
        y, _sr = librosa.load(str(in_path), sr=sr, mono=True)
        dur = float(librosa.get_duration(y=y, sr=sr))
    except Exception:
        sr = 16000
        dur = None

    meta = {
        "upload_id": upload_id,
        "filename": file.filename,
        "saved_as": str(in_path),
        "sample_rate": sr,
        "duration_sec": dur,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (up_dir / "upload_meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "ok": True,
        "upload_id": upload_id,
        "filename": file.filename,
        "duration_sec": dur,
        "sample_rate": sr,
        "files": _list_files(up_dir),
    }


@router.post("/search")
def rag_search(body: RagSearchBody):
    up_dir = _safe_join(RAG_BASE, body.upload_id)
    if not up_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")

    input_wav = up_dir / "input.wav"
    if not input_wav.exists():
        raise HTTPException(status_code=404, detail="input.wav missing for this upload")

    try:
        from src.models.audio_RAG_service import run_audio_rag_service
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot import src.models.audio_RAG_service.run_audio_rag_service. Error: {e}",
        )

    out = run_audio_rag_service(
        input_wav=str(input_wav),
        target_duration_sec=float(body.target_duration_sec),
        top_k=int(body.top_k),
        output_dir=str(up_dir),
        repo_root=ROOT,
    )

    if not out.get("ok", False):
        raise HTTPException(status_code=500, detail=out.get("error", "RAG search failed"))

    files = _list_files(up_dir)
    rank_map = {}  # rank -> dict we return

    wav_files = [k for k in files.keys() if k.lower().endswith(".wav") and k.lower().startswith("rank")]
    wav_files.sort()
    
    # ---- NEW: merge model-returned metrics (coherence/relevance/etc) into UI results ----
    metrics_by_rank = {}
    for r in out.get("results", []):
        rk = int(r.get("rank", 0) or 0)
        if rk > 0:
            metrics_by_rank[rk] = r

    for name in wav_files:
        info = _parse_rank_idx_score(name)
        if not info:
            continue

        rank = info.get("rank")
        db_idx = info.get("db_idx")          # from filename parse (may exist)
        db_id_4s = info.get("db_id_4s")      # only exists for render artifacts in your current parser
        score = info.get("score")
        kind = info.get("kind")

        if rank not in rank_map:
            rank_map[rank] = {
                "rank": rank,
                "db_id_4s": db_id_4s,   # will likely be None initially
                "db_idx": db_idx,       # new: filename IDX
                "score": score,
                "coherence": None,
                "relevance": None,
                "lookback": None,

                "extension_wav": None,
                "arranged_wav": None,
                "stitched_wav": None,  # backward compat

                "extension_wav_url": None,
                "arranged_wav_url": None,
                "stitched_wav_url": None,

                "analysis_extension_url": None,
                "analysis_arranged_url": None,
                "analysis_stitched_url": None,

                # from /render (optional presence)
                "full_wav": None,
                "full_wav_url": None,
                "analysis_full_url": None,
            }
            
        # NEW: hydrate coherence/relevance/lookback/etc from model output
        m = metrics_by_rank.get(rank)
        if m:
            # ALWAYS trust model-computed db_id_4s over filename parse
            if m and m.get("db_id_4s") is not None:
                rank_map[rank]["db_id_4s"] = int(m["db_id_4s"])
            if rank_map[rank].get("score") is None and m.get("score") is not None:
                rank_map[rank]["score"] = float(m["score"])

            rank_map[rank]["coherence"] = float(m.get("coherence", 0.0)) if m.get("coherence") is not None else None
            rank_map[rank]["relevance"] = float(m.get("relevance", 0.0)) if m.get("relevance") is not None else None
            rank_map[rank]["lookback"] = float(m.get("lookback", 0.0)) if m.get("lookback") is not None else None    

        # Fill db_id_4s/score if we didn't have it yet
        if rank_map[rank].get("db_id_4s") is None and db_id_4s is not None:
            rank_map[rank]["db_id_4s"] = db_id_4s
        if rank_map[rank].get("score") is None and score is not None:
            rank_map[rank]["score"] = score
            
        if rank_map[rank].get("db_idx") is None and db_idx is not None:
            rank_map[rank]["db_idx"] = int(db_idx)

        url = files.get(name)

        if kind == "extension":
            rank_map[rank]["extension_wav"] = name
            rank_map[rank]["extension_wav_url"] = url
            rank_map[rank]["analysis_extension_url"] = f"/rag/{body.upload_id}/analysis?file={name}"

        elif kind == "stitched":
            rank_map[rank]["stitched_wav"] = name
            rank_map[rank]["stitched_wav_url"] = url
            rank_map[rank]["analysis_stitched_url"] = f"/rag/{body.upload_id}/analysis?file={name}"

        elif kind == "full":
            rank_map[rank]["full_wav"] = name
            rank_map[rank]["full_wav_url"] = url
            rank_map[rank]["analysis_full_url"] = f"/rag/{body.upload_id}/analysis?file={name}"

        else:  # arranged
            # This could be either scored arranged (Rank..IDX..Score..wav) or render arranged (RankX_arranged.wav)
            rank_map[rank]["arranged_wav"] = name
            rank_map[rank]["arranged_wav_url"] = url
            rank_map[rank]["analysis_arranged_url"] = f"/rag/{body.upload_id}/analysis?file={name}"

    results = [rank_map[k] for k in sorted(rank_map.keys())]

    manifest = {
        "ok": True,
        "upload_id": body.upload_id,
        "target_duration_sec": float(body.target_duration_sec),
        "top_k": int(body.top_k),
        "results": results,
        "files": files,
        "download_zip_url": f"/rag/{body.upload_id}/download.zip",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (up_dir / "rag_results.json").write_text(json.dumps(manifest, indent=2))
    return manifest


@router.get("/{upload_id}/files/{filename}")
def rag_file(upload_id: str, filename: str):
    up_dir = _safe_join(RAG_BASE, upload_id)
    if not up_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")

    # ✅ Files are saved directly under up_dir (NOT up_dir/files)
    p = _safe_join(up_dir, filename)
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


@router.get("/{upload_id}/download.zip")
def rag_download_zip(upload_id: str):
    up_dir = _safe_join(RAG_BASE, upload_id)
    if not up_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in up_dir.glob("*"):
            if p.is_file():
                z.write(p, arcname=p.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{upload_id}.zip"'},
    )


class RagRenderBody(BaseModel):
    upload_id: str
    rank: int
    do_arrange: bool = False


@router.post("/render")
def rag_render(body: RagRenderBody):
    up_dir = _safe_join(RAG_BASE, body.upload_id)
    if not up_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")

    input_wav = up_dir / "input.wav"
    if not input_wav.exists():
        raise HTTPException(status_code=404, detail="input.wav missing")

    # Find extension file for rank
    ext_path = None
    for p in up_dir.glob(f"Rank{body.rank}_IDX_*_extension.wav"):
        ext_path = p
        break
    if ext_path is None:
        raise HTTPException(status_code=404, detail=f"extension wav not found for rank {body.rank}")

    # Load both at same SR
    target_sr = 16000
    y_in, _ = librosa.load(str(input_wav), sr=target_sr, mono=True)
    y_ext, _ = librosa.load(str(ext_path), sr=target_sr, mono=True)

    # Crossfade
    crossfade_sec = 0.15
    L = int(crossfade_sec * target_sr)
    L = min(L, len(y_in), len(y_ext))

    if L > 0:
        fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, L, dtype=np.float32)

        tail = (y_in[-L:].astype(np.float32) * fade_out) + (y_ext[:L].astype(np.float32) * fade_in)
        y_full = np.concatenate([y_in[:-L], tail, y_ext[L:]]).astype(np.float32)
    else:
        y_full = np.concatenate([y_in, y_ext]).astype(np.float32)

    # Write full
    import soundfile as sf

    full_name = f"Rank{body.rank}_full.wav"
    full_path = up_dir / full_name
    sf.write(str(full_path), y_full, target_sr)

    arranged_name = None
    arranged_url = None
    arranged_png_url = None
    analysis_arranged_url = None

    if body.do_arrange:
        # Load YAML config so we can compute features consistently and find temp_leader_name
        try:
            from configs.load_config import load_yaml
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot import configs.load_config.load_yaml: {e}")

        cfg = load_yaml(ROOT / "configs" / "audio_rag.yaml")
        project_root = (ROOT / cfg["paths"].get("project_root", ".")).resolve()

        core_dir = (project_root / cfg["paths"]["core_dir"]).resolve()
        data_dir = (project_root / cfg["paths"]["features_dir"]).resolve()

        # ensure src import works
        if str(core_dir) not in sys.path:
            sys.path.insert(0, str(core_dir))

        try:
            from src.models import perform_music as neural_band
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot import src.models.perform_music: {e}")

        pyin_hop = int(cfg["librosa"]["pyin_hop_length"])
        pyin_fmin = float(cfg["librosa"]["pyin_fmin"])
        pyin_fmax = float(cfg["librosa"]["pyin_fmax"])

        rms_frame = int(cfg["librosa"]["rms_frame_length"])
        rms_hop = int(cfg["librosa"]["rms_hop_length"])

        if rms_hop != pyin_hop:
            raise HTTPException(
                status_code=500,
                detail=f"Config mismatch: rms_hop({rms_hop}) != pyin_hop({pyin_hop})",
            )

        # Compute pitch + loudness for y_full
        final_pitch, _, _ = librosa.pyin(
            y_full, fmin=pyin_fmin, fmax=pyin_fmax, sr=target_sr, hop_length=pyin_hop
        )
        final_pitch = np.nan_to_num(final_pitch)

        final_loud = librosa.feature.rms(
            y=y_full, frame_length=rms_frame, hop_length=rms_hop, center=True
        )[0]
        final_loud = np.nan_to_num(final_loud)

        N_frames = min(len(final_pitch), len(final_loud))
        if N_frames <= 0:
            raise HTTPException(status_code=500, detail="No frames available for arrangement")

        temp_leader = str(cfg["models"]["temp_leader_name"])
        temp_name = f"{temp_leader}_Rank{body.rank}"

        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"pitch_{temp_name}.npy", final_pitch[:N_frames].reshape(1, N_frames))
        np.save(data_dir / f"loudness_{temp_name}.npy", final_loud[:N_frames].reshape(1, N_frames))
        np.save(data_dir / f"audio_{temp_name}.npy", y_full.reshape(1, len(y_full)))

        # Drive perform_music
        neural_band.LEADER = temp_name
        neural_band.GENERATE_SECONDS = float(len(y_full) / target_sr)
        neural_band.MANUAL_START_INDEX = 0

        neural_band.run_complete_arrangement()

        # Robustly find produced wav (avoid hardcoding Unison_0)
        produced = list(up_dir.glob(f"Smart_Arrangement_{temp_name}_*.wav"))
        if not produced:
            # last resort: search by mtime among Smart_Arrangement_*.wav
            produced_any = list(up_dir.glob("Smart_Arrangement_*.wav"))
            if produced_any:
                produced_any.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                produced = [produced_any[0]]

        if not produced:
            raise HTTPException(
                status_code=500,
                detail="Arrangement ran but no Smart_Arrangement_*.wav was found in the upload output folder. "
                       "Check perform_music output directory / naming.",
            )

        produced_wav = produced[0]

        arranged_name = f"Rank{body.rank}_arranged.wav"
        arranged_path = up_dir / arranged_name
        shutil.move(str(produced_wav), str(arranged_path))

        arranged_url = f"/rag/{body.upload_id}/files/{arranged_name}"
        analysis_arranged_url = f"/rag/{body.upload_id}/analysis?file={arranged_name}"

        # Move png if it exists
        produced_png = produced_wav.with_suffix(".png")
        if produced_png.exists():
            arranged_png = up_dir / f"Rank{body.rank}_arranged.png"
            shutil.move(str(produced_png), str(arranged_png))
            arranged_png_url = f"/rag/{body.upload_id}/files/{arranged_png.name}"

        # Cleanup temp feature files
        for f in [
            data_dir / f"pitch_{temp_name}.npy",
            data_dir / f"loudness_{temp_name}.npy",
            data_dir / f"audio_{temp_name}.npy",
        ]:
            try:
                if f.exists():
                    os.remove(f)
            except Exception:
                pass

    return {
        "ok": True,
        "upload_id": body.upload_id,
        "rank": body.rank,
        "full_wav": full_name,
        "full_wav_url": f"/rag/{body.upload_id}/files/{full_name}",
        "analysis_full_url": f"/rag/{body.upload_id}/analysis?file={full_name}",
        "arranged_wav": arranged_name,
        "arranged_wav_url": arranged_url,
        "arranged_png_url": arranged_png_url,
        "analysis_arranged_url": analysis_arranged_url,
    }


@router.get("/{upload_id}/analysis")
def rag_analysis(
    upload_id: str,
    file: str | None = Query(default=None, description="wav filename inside upload dir"),
):
    up_dir = _safe_join(RAG_BASE, upload_id)
    if not up_dir.exists():
        raise HTTPException(status_code=404, detail="upload_id not found")

    # Treat "null"/"undefined"/empty as missing
    if file is None or str(file).strip().lower() in {"", "null", "undefined"}:
        wav_name = "input.wav"
    else:
        wav_name = file

    if not wav_name.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="analysis supports .wav files only")

    wav_path = _safe_join(up_dir, wav_name)
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {wav_name}")

    target_sr = 16000
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    wave = np.abs(y).astype(np.float32, copy=False)
    t_wave = np.linspace(0, duration, num=len(wave), endpoint=False)

    frame_len = 2048
    hop = 64  # keep consistent with perform UI (or later read from a yaml)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0].astype(np.float32, copy=False)
    rms_norm = rms / (float(np.max(rms)) + 1e-7)
    t_rms = librosa.times_like(rms_norm, sr=sr, hop_length=hop).astype(np.float32, copy=False)

    f0 = librosa.yin(y, fmin=40, fmax=2000, sr=sr, frame_length=frame_len, hop_length=hop)
    f0 = np.nan_to_num(f0).astype(np.float32, copy=False)
    t_f0 = librosa.times_like(f0, sr=sr, hop_length=hop).astype(np.float32, copy=False)

    scores = _calculate_scores(f0, rms_norm)

    return {
        "file": wav_name,
        "duration": duration,
        "scores": scores,
        "wave": _downsample_series(wave, t_wave, max_points=1600),
        "rms": _downsample_series(rms_norm, t_rms, max_points=1400),
        "f0": _downsample_series(f0, t_f0, max_points=1400),
    }
