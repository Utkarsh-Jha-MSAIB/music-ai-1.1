# core/src/models/audio_RAG_service.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.load_config import load_yaml

# IMPORTANT: this is your existing big pipeline function (unchanged)
from src.models.audio_RAG import run_audio_rag


def run_audio_rag_service(
    *,
    input_wav: str,
    target_duration_sec: float,
    top_k: int,
    output_dir: str,
    repo_root: Path,
    do_arrange: bool = False,   # ✅ NEW
) -> dict[str, Any]:
    """
    Wrapper that runs your Audio RAG pipeline.

    ✅ do_arrange=False (UI mode):
        - should produce Top-K candidate/stitched WAVs into output_dir
        - must NOT run perform_music arrangement for all K

    ✅ do_arrange=True (batch/CLI mode):
        - runs your original behavior (if you still want it)
    """

    cfg = load_yaml(repo_root / "configs" / "audio_rag.yaml")

    # Ensure nested dicts exist
    cfg.setdefault("io", {})
    cfg.setdefault("rag", {})
    cfg.setdefault("search", {})
    cfg.setdefault("paths", {})

    # Overrides
    cfg["io"]["input_filename"] = str(input_wav)
    cfg["rag"]["target_duration_sec"] = float(target_duration_sec)
    cfg["search"]["top_k_results"] = int(top_k)

    # Force outputs into upload folder
    cfg["paths"]["outputs_dir"] = str(output_dir)

    # ✅ NEW: control arrangement behavior from API/UI
    cfg["rag"]["do_arrange"] = bool(do_arrange)

    return run_audio_rag(
        input_filename=str(input_wav),
        target_duration=float(target_duration_sec),
        top_k=int(top_k),
        cfg_override=cfg,      # full config
        repo_root=repo_root,
    )


def arrange_one_service(
    *,
    upload_dir: str,
    stitched_wav_name: str,
    repo_root: Path,
) -> dict[str, Any]:
    """
    Arrange ONE selected stitched wav (after user picks a Top-K card).

    This intentionally DOES NOT touch /generate or your existing perform UI.
    It just uses the same perform_music stack for a single file.

    Expected outputs placed into upload_dir:
      - arranged.wav (or a name you choose)
      - arranged.png (if your pipeline produces it)
    """

    # You already rely on being able to import src.* via core/ on sys.path in API layer,
    # but we keep it defensive in case this is used elsewhere.
    import sys
    CORE = (repo_root / "core").resolve()
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))

    from src.models import perform_music as neural_band

    up = Path(upload_dir).resolve()
    stitched_path = (up / stitched_wav_name).resolve()
    if not stitched_path.exists():
        return {"ok": False, "error": f"Stitched wav not found: {stitched_path}"}

    # --- YOU have two options here ---
    #
    # Option A (clean): save features for this stitched wav using YOUR existing feature-extraction logic,
    # then set LEADER=tempname and call neural_band.run_complete_arrangement().
    #
    # Option B (minimal): if perform_music can accept a direct wav input (many setups can't),
    # call that path.
    #
    # Since your current RAG script uses temp npy files (pitch_/loudness_/audio_),
    # we'll keep that same strategy here: create ONE temp leader and run arrangement ONCE.

    import numpy as np
    import librosa

    # load config so we use consistent params (sr/hops)
    cfg = load_yaml(repo_root / "configs" / "audio_rag.yaml")

    sample_rate = int(cfg["audio"]["sample_rate"])
    frame_rate = int(cfg["audio"]["frame_rate"])

    pyin_hop = int(cfg["librosa"]["pyin_hop_length"])
    rms_frame = int(cfg["librosa"]["rms_frame_length"])
    rms_hop = int(cfg["librosa"]["rms_hop_length"])
    pyin_fmin = float(cfg["librosa"]["pyin_fmin"])
    pyin_fmax = float(cfg["librosa"]["pyin_fmax"])

    # features dir where perform_music expects temp npys
    project_root = (repo_root / cfg["paths"].get("project_root", ".")).resolve()
    data_dir = (project_root / cfg["paths"]["features_dir"]).resolve()

    temp_leader_base = str(cfg["models"].get("temp_leader_name", "temp_leader"))
    temp_name = f"{temp_leader_base}_UI"

    # load stitched wav
    y, _ = librosa.load(str(stitched_path), sr=sample_rate, mono=True)
    duration_sec = float(len(y) / sample_rate)

    # extract pitch/loudness
    pitch_arr, _, _ = librosa.pyin(
        y,
        fmin=pyin_fmin,
        fmax=pyin_fmax,
        sr=sample_rate,
        hop_length=pyin_hop,
    )
    pitch_arr = np.nan_to_num(pitch_arr)

    loud_arr = librosa.feature.rms(
        y=y,
        frame_length=rms_frame,
        hop_length=rms_hop,
        center=True,
    )[0]
    loud_arr = np.nan_to_num(loud_arr)

    N_frames = int(min(len(pitch_arr), len(loud_arr)))
    if N_frames <= 0:
        return {"ok": False, "error": "No frames extracted for stitched audio"}

    # build the 1xN npys like your pipeline does
    N_chunks = 1
    np.save(data_dir / f"pitch_{temp_name}.npy", pitch_arr[:N_frames].reshape(N_chunks, N_frames))
    np.save(data_dir / f"loudness_{temp_name}.npy", loud_arr[:N_frames].reshape(N_chunks, N_frames))
    np.save(data_dir / f"audio_{temp_name}.npy", y.reshape(N_chunks, len(y)))

    # run arrangement ONCE
    neural_band.LEADER = temp_name
    neural_band.GENERATE_SECONDS = duration_sec
    neural_band.MANUAL_START_INDEX = 0

    try:
        neural_band.run_complete_arrangement()
    except Exception as e:
        return {"ok": False, "error": f"perform_music failed: {e}"}

    # rename output into upload dir (same trick as your pipeline)
    # Update these names if your perform_music writes different defaults.
    default_out_name = f"Smart_Arrangement_{temp_name}_Unison_0.wav"
    default_out_path = (project_root / cfg["paths"]["outputs_dir"] / default_out_name).resolve()

    # if your perform outputs elsewhere, adjust this path to match where it actually writes
    if not default_out_path.exists():
        return {"ok": False, "error": f"Expected perform output missing: {default_out_path}"}

    arranged_name = f"ARRANGED_{stitched_wav_name}"
    arranged_path = (up / arranged_name).resolve()

    # move wav
    import shutil
    shutil.move(str(default_out_path), str(arranged_path))

    # move png if present
    default_png = default_out_path.with_suffix(".png")
    arranged_png = arranged_path.with_suffix(".png")
    if default_png.exists():
        shutil.move(str(default_png), str(arranged_png))

    # cleanup temp feature npys
    for f in [
        data_dir / f"pitch_{temp_name}.npy",
        data_dir / f"loudness_{temp_name}.npy",
        data_dir / f"audio_{temp_name}.npy",
    ]:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    return {
        "ok": True,
        "arranged_wav": arranged_name,
        "arranged_png": arranged_png.name if arranged_png.exists() else None,
    }
