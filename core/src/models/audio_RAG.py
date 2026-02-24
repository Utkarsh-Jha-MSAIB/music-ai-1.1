"""
Audio RAG (clean + robust top-K) — UPDATED MODULE (pitch + loudness continuity + gate toggles + stats logging)

Change requested (implemented here):
✅ Keep your polyphonic-safe “pitch coherence” via CHROMA PROGRESSION at the seam
✅ Improve query embedding to include chroma motion (mean + std + delta-mean) -> less “same extension”
✅ Make chroma floors stricter (defaults: strict=0.65, fallback=0.50) to avoid generic tonal matches
✅ Add diversity selection (penalize near-duplicate harmonic beds) on top of ban_radius
✅ Enforce TRUE pitch-contour continuity (F0 seam gate + pitch contour similarity) so extensions don’t jump register/melody
✅ Optionally match a longer loudness head window (beyond tiny seam) to keep energy consistent
✅ NEW: Gate toggles (disable/soften RMS prefilter and/or pitch register gate)
✅ NEW: Per-attempt stats dict + return it in output + print it

Everything else is unchanged.
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from configs.load_config import load_yaml


# ---------- helpers ----------

def fmt(n: float, d: int = 4) -> str:
    return f"{float(n):.{d}f}"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    if a.shape[0] != b.shape[0]:
        # dimension mismatch -> cannot compare meaningfully
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def safe_nan(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def hz_to_cents(hz: np.ndarray, ref_hz: float) -> np.ndarray:
    """Convert Hz -> cents relative to ref_hz. hz<=0 -> 0."""
    hz = hz.astype(np.float32, copy=False)
    out = np.zeros_like(hz, dtype=np.float32)
    mask = (hz > 0) & (ref_hz > 0)
    if np.any(mask):
        out[mask] = 1200.0 * np.log2(hz[mask] / float(ref_hz))
    return out


def robust_median_hz(f0: np.ndarray) -> float:
    """Median over voiced frames only. Returns 0 if no voiced frames."""
    f0 = safe_nan(f0).astype(np.float32, copy=False)
    v = f0[f0 > 0]
    if v.size == 0:
        return 0.0
    return float(np.median(v))


def pitch_contour_sim(input_f0: np.ndarray, cand_f0: np.ndarray) -> float:
    """
    Seam pitch similarity:
    - absolute register match (median cents distance)
    - motion match (delta-cents cosine)
    Returns [0,1].
    """
    eps = 1e-9
    A = safe_nan(input_f0).astype(np.float32, copy=False)
    B = safe_nan(cand_f0).astype(np.float32, copy=False)

    F = min(A.shape[0], B.shape[0])
    if F <= 1:
        return 0.0

    A = A[:F]
    B = B[:F]

    medA = robust_median_hz(A)
    medB = robust_median_hz(B)
    if medA <= 0 or medB <= 0:
        return 0.0

    cents_med = abs(1200.0 * np.log2(medB / medA))
    abs_sim = float(np.exp(-(cents_med / 200.0)))

    Ac = hz_to_cents(A, medA)
    Bc = hz_to_cents(B, medB)

    dA = np.diff(Ac)
    dB = np.diff(Bc)

    m = (np.abs(dA) + np.abs(dB)) > 1e-3
    if np.sum(m) < 2:
        motion_sim = abs_sim
    else:
        x = dA[m]
        y = dB[m]
        nx = float(np.linalg.norm(x) + eps)
        ny = float(np.linalg.norm(y) + eps)
        motion_cos = float(np.clip(np.dot(x, y) / (nx * ny), -1.0, 1.0))
        motion_sim = 0.5 * (motion_cos + 1.0)

    return 0.60 * abs_sim + 0.40 * motion_sim


def audio_to_chroma_frames(audio_waveform: np.ndarray, *, sr: int, chroma_hop: int) -> np.ndarray:
    """Return chroma frames (12, T) using CQT chroma."""
    C = librosa.feature.chroma_cqt(y=audio_waveform, sr=sr, hop_length=chroma_hop).astype(np.float32)
    return C


def chroma_mean_vec(frames_12xT: np.ndarray) -> np.ndarray:
    """Normalized mean chroma vector (12,)."""
    eps = 1e-9
    if frames_12xT.size == 0 or frames_12xT.shape[1] == 0:
        return np.zeros(12, dtype=np.float32)
    v = np.mean(frames_12xT, axis=1).astype(np.float32)
    v = v / (float(np.linalg.norm(v)) + eps)
    return v


def chroma_embed_motion(frames_12xT: np.ndarray) -> np.ndarray:
    """
    Motion-aware chroma embedding: [mean(12), std(12), delta-mean(12)] -> 36D
    Polyphonic-safe, cheap, and much more discriminative than mean-only.
    """
    eps = 1e-9
    if frames_12xT.size == 0 or frames_12xT.shape[1] == 0:
        return np.zeros(36, dtype=np.float32)

    C = frames_12xT.astype(np.float32, copy=False)
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + eps)

    mu = np.mean(Cn, axis=1)
    sd = np.std(Cn, axis=1)

    if Cn.shape[1] >= 2:
        dmu = np.mean(np.diff(Cn, axis=1), axis=1)
    else:
        dmu = np.zeros(12, dtype=np.float32)

    emb = np.concatenate([mu, sd, dmu]).astype(np.float32)
    emb = emb / (float(np.linalg.norm(emb)) + eps)
    return emb


def chroma_progression_sim(A: np.ndarray, B: np.ndarray) -> float:
    """
    Polyphonic-safe “pitch coherence” proxy.
    Compares seam-window chroma frames + their first differences (motion).
    A, B: (12, F)
    """
    eps = 1e-9
    if A.size == 0 or B.size == 0:
        return 0.0

    F = min(A.shape[1], B.shape[1])
    if F <= 0:
        return 0.0

    A = A[:, :F].astype(np.float32)
    B = B[:, :F].astype(np.float32)

    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)

    frame_sim = float(np.mean(np.sum(An * Bn, axis=0)))

    if F >= 2:
        dA = np.diff(An, axis=1)
        dB = np.diff(Bn, axis=1)
        dAn = dA / (np.linalg.norm(dA, axis=0, keepdims=True) + eps)
        dBn = dB / (np.linalg.norm(dB, axis=0, keepdims=True) + eps)
        motion_sim = float(np.mean(np.sum(dAn * dBn, axis=0)))
    else:
        motion_sim = frame_sim

    return 0.65 * frame_sim + 0.35 * motion_sim


def crossfade_audio(audio_out: np.ndarray, audio_in: np.ndarray, length_samples: int) -> int:
    """
    In-place crossfade:
      - fades out the last L samples of audio_out
      - fades in the first L samples of audio_in
    Returns L actually used.
    """
    L = min(int(length_samples), len(audio_out), len(audio_in))
    if L <= 0:
        return 0

    fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, L, dtype=np.float32)

    audio_out[-L:] *= fade_out
    audio_in[:L] *= fade_in
    return L


def get_gru_embedding(
    audio_segment: np.ndarray,
    *,
    gru_ckpt: Path,
    sr: int,
    chroma_hop: int,
) -> np.ndarray:
    """
    Placeholder upgraded:
    Motion-aware chroma embedding (36D) instead of mean-only (12D).
    Keeps signature so you can later drop in the real GRU model.
    """
    _ = gru_ckpt
    C = audio_to_chroma_frames(audio_segment, sr=sr, chroma_hop=chroma_hop)
    return chroma_embed_motion(C)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-9)
    return (x - mu) / sd


def shape_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cheap "trend" similarity for 1D curves (no DTW).
    - z-score normalize
    - compare both value shape + first-derivative shape
    Returns roughly [0,1].
    """
    A = zscore(a)
    B = zscore(b)
    F = min(A.size, B.size)
    if F < 8:
        return 0.0
    A = A[:F]
    B = B[:F]

    # cosine on normalized values
    eps = 1e-9
    na = float(np.linalg.norm(A) + eps)
    nb = float(np.linalg.norm(B) + eps)
    cos0 = float(np.clip(np.dot(A, B) / (na * nb), -1.0, 1.0))
    sim0 = 0.5 * (cos0 + 1.0)

    # derivative similarity (captures peak/fall/rise behavior)
    dA = np.diff(A)
    dB = np.diff(B)
    ndA = float(np.linalg.norm(dA) + eps)
    ndB = float(np.linalg.norm(dB) + eps)
    cos1 = float(np.clip(np.dot(dA, dB) / (ndA * ndB), -1.0, 1.0))
    sim1 = 0.5 * (cos1 + 1.0)

    return 0.55 * sim0 + 0.45 * sim1


def loud_trend_sim(a_loud: np.ndarray, b_loud: np.ndarray) -> float:
    """
    Loudness trend similarity: log-compress then shape-sim.
    """
    A = np.log1p(np.maximum(np.asarray(a_loud, dtype=np.float32), 0.0))
    B = np.log1p(np.maximum(np.asarray(b_loud, dtype=np.float32), 0.0))
    return shape_sim_1d(A, B)


def pitch_trend_sim(a_f0: np.ndarray, b_f0: np.ndarray) -> float:
    """
    Pitch trend similarity:
    - convert each segment to cents relative to its own median (register-invariant)
    - compare shape + derivative
    """
    A = safe_nan(np.asarray(a_f0, dtype=np.float32))
    B = safe_nan(np.asarray(b_f0, dtype=np.float32))
    F = min(A.size, B.size)
    if F < 8:
        return 0.0
    A = A[:F]
    B = B[:F]

    medA = robust_median_hz(A)
    medB = robust_median_hz(B)
    if medA <= 0 or medB <= 0:
        return 0.0

    Ac = hz_to_cents(A, medA)
    Bc = hz_to_cents(B, medB)
    return shape_sim_1d(Ac, Bc)


def dtw_cost(a: np.ndarray, b: np.ndarray, *, band: int = 12) -> float:
    """
    Simple banded DTW cost for 1D sequences (O(N*band)).
    Lower is better.
    band is in samples (after downsampling).
    """
    A = np.asarray(a, dtype=np.float32).reshape(-1)
    B = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(A.size, B.size)
    if n < 8:
        return 1e9
    A = A[:n]; B = B[:n]

    INF = 1e18
    # dp[j] = cost up to (i, j)
    dp = np.full(n + 1, INF, dtype=np.float64)
    dp[0] = 0.0

    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(n, i + band)
        new = np.full(n + 1, INF, dtype=np.float64)
        for j in range(j0, j1 + 1):
            cost = float((A[i - 1] - B[j - 1]) ** 2)
            new[j] = cost + min(new[j - 1], dp[j], dp[j - 1])
        dp = new

    return float(dp[n] / n)


# ---------- main ----------

def run_audio_rag(
    *,
    input_filename: Optional[str] = None,
    target_duration: Optional[float] = None,
    top_k: Optional[int] = None,
    cfg_override: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> dict:
    here = Path(__file__).resolve()
    repo_root = repo_root or next(p for p in here.parents if (p / "configs").exists())
    cfg = cfg_override or load_yaml(repo_root / "configs" / "audio_rag.yaml")
    project_root = (repo_root / cfg["paths"].get("project_root", ".")).resolve()

    if input_filename is not None:
        cfg.setdefault("io", {})
        cfg["io"]["input_filename"] = input_filename
    if target_duration is not None:
        cfg.setdefault("rag", {})
        cfg["rag"]["target_duration_sec"] = float(target_duration)
    if top_k is not None:
        cfg.setdefault("search", {})
        cfg["search"]["top_k_results"] = int(top_k)

    # ---------- NEW: gate toggles ----------
    gate_cfg = cfg.get("gates", {}) if isinstance(cfg.get("gates", {}), dict) else {}
    ENABLE_RMS_PREFILTER = bool(gate_cfg.get("enable_rms_prefilter", True))
    ENABLE_PITCH_REGISTER_GATE = bool(gate_cfg.get("enable_pitch_register_gate", True))
    PITCH_REGISTER_AS_PENALTY = bool(gate_cfg.get("pitch_register_as_penalty", False))
    PITCH_REGISTER_PENALTY_W = float(gate_cfg.get("pitch_register_penalty_w", 0.35))

    # Optional: return printed logs
    LOG_STATS = bool(gate_cfg.get("log_stats", True))
    # --------------------------------------

    core_dir = (project_root / cfg["paths"]["core_dir"]).resolve()
    data_dir = (project_root / cfg["paths"]["features_dir"]).resolve()
    output_dir = (project_root / cfg["paths"]["outputs_dir"]).resolve()

    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    input_path = Path(cfg["io"]["input_filename"])
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    # RAG params
    target_duration = float(cfg["rag"]["target_duration_sec"])
    base_chunk_duration = float(cfg["rag"]["base_chunk_duration_sec"])
    lookback_options = list(cfg["rag"]["lookback_options_sec"])
    seam_frame_count = int(cfg["rag"]["seam_frame_count"])

    # scoring weights
    alpha = float(cfg["scoring"]["alpha"])
    beta = float(cfg["scoring"]["beta"])
    coherence_min_threshold = float(cfg["scoring"]["coherence_min_threshold"])

    # filters/search
    rms_tolerance_pct = float(cfg["filters"]["rms_tolerance_pct"])
    top_k_results = int(cfg["search"]["top_k_results"])
    stride = int(cfg["search"]["stride"])
    attempts = int(cfg["search"].get("attempts", 2))
    ban_radius = int(cfg["search"].get("ban_radius", 20))

    # diversity
    diverse_sim_max = float(cfg["search"].get("diverse_sim_max", 0.92))
    diverse_enable = bool(cfg["search"].get("diverse_enable", True))

    # pitch continuity defaults
    pitch_cents_max_default = 250.0
    pitch_floor_default = 0.55
    reject_unvoiced_seam_default = True

    # audio
    sample_rate = int(cfg["audio"]["sample_rate"])
    frame_rate = int(cfg["audio"]["frame_rate"])
    crossfade_time_sec = float(cfg["audio"]["crossfade_time_sec"])
    crossfade_time_samples = int(crossfade_time_sec * sample_rate)

    # librosa
    pyin_hop = int(cfg["librosa"]["pyin_hop_length"])
    rms_frame = int(cfg["librosa"]["rms_frame_length"])
    rms_hop = int(cfg["librosa"]["rms_hop_length"])
    chroma_hop = int(cfg["librosa"]["chroma_hop_length"])
    pyin_fmin = float(cfg["librosa"]["pyin_fmin"])
    pyin_fmax = float(cfg["librosa"]["pyin_fmax"])

    # DTW params
    dtw_enable = bool(cfg["search"].get("dtw_enable", False))
    dtw_rerank_n = int(cfg["search"].get("dtw_rerank_n", max(top_k_results * 10, 80)))
    dtw_downsample = int(cfg["search"].get("dtw_downsample", 3))
    dtw_band = int(cfg["search"].get("dtw_band", 12))
    w_dtw = float(cfg["scoring"].get("w_dtw", 0.25))

    if rms_hop != pyin_hop:
        raise ValueError(
            f"Config mismatch: rms_hop({rms_hop}) != pyin_hop({pyin_hop}). "
            "This pipeline slices pitch/loudness on the same frame axis."
        )
    expected_fps = sample_rate / rms_hop
    if abs(frame_rate - expected_fps) > 1e-6:
        raise ValueError(
            f"Config mismatch: frame_rate={frame_rate} vs expected sample_rate/rms_hop={expected_fps}"
        )

    # DB paths
    db_instrument = str(cfg["database"]["instrument"])
    pitch_path = (data_dir / cfg["database"]["pitch_file"].format(instrument=db_instrument)).resolve()
    loud_path = (data_dir / cfg["database"]["loudness_file"].format(instrument=db_instrument)).resolve()
    audio_path = (data_dir / cfg["database"]["audio_file"].format(instrument=db_instrument)).resolve()
    chroma_frames_path = (data_dir / cfg["database"]["chroma_frames_file"].format(instrument=db_instrument)).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    for p in (pitch_path, loud_path, audio_path, chroma_frames_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing DB file: {p}")

    try:
        print(f"Starting Audio RAG Pipeline (Target: {target_duration}s)")
        y_in, _sr = librosa.load(str(input_path), sr=sample_rate)

        # input loudness
        input_loud = librosa.feature.rms(y=y_in, frame_length=rms_frame, hop_length=rms_hop, center=True)[0]
        input_loud = np.nan_to_num(input_loud)

        input_duration = len(y_in) / sample_rate
        if target_duration <= input_duration:
            raise ValueError(
                f"Target duration ({target_duration}s) must be > input duration ({input_duration:.2f}s)"
            )

        extend_sec = target_duration - input_duration
        extend_len_samples = max(1, int(extend_sec * sample_rate))
        print(f"Input: {input_duration:.2f}s, need extension: {extend_sec:.2f}s")

        # seam config (for chroma)
        seam_sec = float(cfg["rag"].get("seam_sec", 2.5))
        seam_samples = max(1, int(seam_sec * sample_rate))
        input_tail = y_in[-seam_samples:] if len(y_in) >= seam_samples else y_in

        # seam pitch (pyin) — computed once
        seam_pitch_sec = float(cfg["rag"].get("seam_sec", 2.5))
        seam_pitch_frames = max(4, int(seam_pitch_sec * frame_rate))

        f0_in, _, _ = librosa.pyin(
            y_in,
            fmin=pyin_fmin,
            fmax=pyin_fmax,
            sr=sample_rate,
            frame_length=rms_frame,
            hop_length=pyin_hop,
            center=True,
        )
        f0_in = safe_nan(np.asarray(f0_in, dtype=np.float32))

        input_pitch_tail = f0_in[-seam_pitch_frames:] if f0_in.size >= seam_pitch_frames else f0_in
        if input_pitch_tail.size < seam_pitch_frames:
            if input_pitch_tail.size == 0:
                input_pitch_tail = np.zeros(seam_pitch_frames, dtype=np.float32)
            else:
                last = float(input_pitch_tail[-1])
                pad = np.full(seam_pitch_frames - input_pitch_tail.size, last, dtype=np.float32)
                input_pitch_tail = np.concatenate([input_pitch_tail, pad]).astype(np.float32)

        input_pitch_med = robust_median_hz(input_pitch_tail)

        # seam chroma vector
        input_seam_frames_full = audio_to_chroma_frames(input_tail, sr=sample_rate, chroma_hop=chroma_hop)
        input_seam_chroma = chroma_mean_vec(input_seam_frames_full)

        # seam frames for progression coherence
        seam_c_frames = max(2, int((seam_sec * sample_rate) / chroma_hop))
        if input_seam_frames_full.shape[1] >= seam_c_frames:
            input_seam_frames = input_seam_frames_full[:, -seam_c_frames:]
        else:
            pad = seam_c_frames - input_seam_frames_full.shape[1]
            if input_seam_frames_full.shape[1] == 0:
                input_seam_frames = np.zeros((12, seam_c_frames), dtype=np.float32)
            else:
                last = input_seam_frames_full[:, [-1]]
                input_seam_frames = np.concatenate(
                    [input_seam_frames_full, np.repeat(last, pad, axis=1)],
                    axis=1,
                ).astype(np.float32)

        # RMS reference
        avg_in = float(np.mean(input_loud))
        std_in = float(np.std(input_loud))

        # DB load (memmap)
        db_pitch = np.load(pitch_path, mmap_mode="r", allow_pickle=False).reshape(-1)
        db_loud = np.load(loud_path, mmap_mode="r", allow_pickle=False).reshape(-1)
        db_audio = np.load(audio_path, mmap_mode="r", allow_pickle=False).reshape(-1)
        db_chroma_frames = np.load(chroma_frames_path, mmap_mode="r", allow_pickle=False)  # (12, T)

        if db_chroma_frames.ndim != 2 or db_chroma_frames.shape[0] != 12:
            raise ValueError(f"Expected chroma_frames (12, T), got {db_chroma_frames.shape}")

        # scanning geometry
        base_chunk_frames = int(base_chunk_duration * frame_rate)
        base_chunk_samples = int(base_chunk_duration * sample_rate)
        if base_chunk_frames <= 0 or base_chunk_samples <= 0:
            raise ValueError("base_chunk_duration_sec too small -> 0 frames/samples")

        db_chunk_sec = float(cfg["search"].get("db_chunk_sec", 4.0))
        group = max(1, int(round(db_chunk_sec / base_chunk_duration)))

        # query embeddings
        user_gru_queries: Dict[float, np.ndarray] = {}
        gru_ckpt = (project_root / cfg["models"]["gru_model_checkpoint"]).resolve()

        for dur in lookback_options:
            dur = float(dur)
            if dur <= input_duration:
                start = len(y_in) - int(dur * sample_rate)
                user_gru_queries[dur] = get_gru_embedding(
                    y_in[start:], gru_ckpt=gru_ckpt, sr=sample_rate, chroma_hop=chroma_hop
                )

        if not user_gru_queries:
            raise ValueError("No lookback queries possible. Reduce lookback_options_sec or use longer input audio.")

        user_durs = sorted(user_gru_queries.keys())
        max_lookback_sec = float(user_durs[-1])
        score_len_frames = max(1, int(max_lookback_sec * frame_rate))
        max_lookback_chroma_frames = max(1, int((max_lookback_sec * sample_rate) / chroma_hop))

        # ---------- NEW: precompute input trend windows per lookback dur ----------
        # IMPORTANT: these are ONLY the lookback periods (dur in user_durs)
        input_loud_by_dur: Dict[float, np.ndarray] = {}
        input_pitch_by_dur: Dict[float, np.ndarray] = {}

        for dur in user_durs:
            dur = float(dur)
            n_frames = max(8, int(dur * frame_rate))

            # loudness: last n_frames
            if input_loud.size >= n_frames:
                input_loud_by_dur[dur] = input_loud[-n_frames:].astype(np.float32, copy=False)
            else:
                pad = n_frames - input_loud.size
                last = float(input_loud[-1]) if input_loud.size else 0.0
                input_loud_by_dur[dur] = np.concatenate(
                    [input_loud.astype(np.float32, copy=False), np.full(pad, last, dtype=np.float32)]
                )

            # pitch: last n_frames
            if f0_in.size >= n_frames:
                input_pitch_by_dur[dur] = f0_in[-n_frames:].astype(np.float32, copy=False)
            else:
                pad = n_frames - f0_in.size
                last = float(f0_in[-1]) if f0_in.size else 0.0
                input_pitch_by_dur[dur] = np.concatenate(
                    [f0_in.astype(np.float32, copy=False), np.full(pad, last, dtype=np.float32)]
                )
        # ------------------------------------------------------------------------

        # sampling range
        num_base_chunks = max(1, int(max_lookback_sec / base_chunk_duration))
        num_db_base_chunks = len(db_loud) // base_chunk_frames
        sampling_range = num_db_base_chunks - num_base_chunks
        if sampling_range <= 0:
            raise RuntimeError("Database too short for scoring window. Increase DB length or reduce lookbacks.")

        sample_indices = range(0, sampling_range, stride)

        heap_mult = int(cfg["search"].get("heap_mult", 20))
        heap_k = max(top_k_results * heap_mult, top_k_results * 50)

        # ---------- NEW: debug stats returned to caller ----------
        debug: dict = {
            "gates": {
                "enable_rms_prefilter": ENABLE_RMS_PREFILTER,
                "enable_pitch_register_gate": ENABLE_PITCH_REGISTER_GATE,
                "pitch_register_as_penalty": PITCH_REGISTER_AS_PENALTY,
                "pitch_register_penalty_w": PITCH_REGISTER_PENALTY_W,
            },
            "attempts": [],
        }
        # ------------------------------------------------------

        def collect_candidates(
            *,
            current_alpha: float,
            current_beta: float,
            coherence_floor: float,
            chroma_floor: float,
            rms_tol_mult: float,
            use_tqdm: bool,
            label: str,
        ) -> list[dict]:
            top_heap: list[tuple[float, int, dict]] = []
            counter = 0

            stats = dict(
                total=0,
                rms=0,
                loud_seam=0,
                loud_ext=0,
                pitch_med=0,
                pitch_jump=0,
                pitch_floor=0,
                chroma=0,
                coh=0,
                kept=0,
                pitch_len=0,
                pitch_unvoiced=0,
                trend_sum=0.0,
                trend_loud_sum=0.0,
                trend_pitch_sum=0.0,
                trend_n=0,              
            )

            avg_min = avg_in * (1 - rms_tolerance_pct * rms_tol_mult)
            avg_max = avg_in * (1 + rms_tolerance_pct * rms_tol_mult)
            std_min = std_in * (1 - rms_tolerance_pct * rms_tol_mult)
            std_max = std_in * (1 + rms_tolerance_pct * rms_tol_mult)

            it = tqdm(sample_indices, desc=label) if use_tqdm else sample_indices
            best_seen = -1e9

            loud_head_mult = float(cfg["search"].get("loud_head_mult", 1.0))

            pitch_cents_max = float(cfg["search"].get("pitch_cents_max", pitch_cents_max_default))
            pitch_floor = float(cfg["search"].get("pitch_floor", pitch_floor_default))
            reject_unvoiced_seam = bool(cfg["search"].get("reject_unvoiced_seam", reject_unvoiced_seam_default))

            w_loud = float(cfg["scoring"].get("w_loud", 0.45))
            w_pitch = float(cfg["scoring"].get("w_pitch", 0.35))
            w_chroma_prog = float(cfg["scoring"].get("w_chroma_prog", 0.20))

            # ---- NEW: trend knobs (loggable) ----
            w_trend_loud = float(cfg["scoring"].get("w_trend_loud", 0.20))
            w_trend_pitch = float(cfg["scoring"].get("w_trend_pitch", 0.10))
            trend_downsample = int(cfg["search"].get("trend_downsample", 2))

            # (optional) if you later enable DTW:
            use_dtw_rerank = bool(cfg["search"].get("use_dtw_rerank", False))
            dtw_band = int(cfg["search"].get("dtw_band", 12))
            dtw_downsample = int(cfg["search"].get("dtw_downsample", 3))
            dtw_top_m = int(cfg["search"].get("dtw_top_m", 50))
            dtw_w_loud = float(cfg["scoring"].get("w_dtw_loud", 0.10))
            dtw_w_pitch = float(cfg["scoring"].get("w_dtw_pitch", 0.10))

            # log the exact parameters used this attempt
            debug_params = {
                "label": label,
                "current_alpha": float(current_alpha),
                "current_beta": float(current_beta),
                "coherence_floor": float(coherence_floor),
                "chroma_floor": float(chroma_floor),
                "rms_tol_mult": float(rms_tol_mult),
                "rms_tolerance_pct": float(rms_tolerance_pct),
                "avg_in": float(avg_in),
                "std_in": float(std_in),
                "loud_head_mult": float(loud_head_mult),
                "pitch_cents_max": float(pitch_cents_max),
                "pitch_floor": float(pitch_floor),
                "reject_unvoiced_seam": bool(reject_unvoiced_seam),
                "w_loud": float(w_loud),
                "w_pitch": float(w_pitch),
                "w_chroma_prog": float(w_chroma_prog),
                "seam_frame_count": int(seam_frame_count),
                "seam_pitch_frames": int(seam_pitch_frames),
                "seam_c_frames": int(seam_c_frames),
                "score_len_frames": int(score_len_frames),
                "stride": int(stride),
                "heap_k": int(heap_k),
                "w_trend_loud": float(w_trend_loud),
                "w_trend_pitch": float(w_trend_pitch),
                "trend_downsample": int(trend_downsample),
                "use_dtw_rerank": bool(use_dtw_rerank),
                "dtw_band": int(dtw_band),
                "dtw_downsample": int(dtw_downsample),
                "dtw_top_m": int(dtw_top_m),
                "w_dtw_loud": float(dtw_w_loud),
                "w_dtw_pitch": float(dtw_w_pitch),
            }

            for base_idx in it:
                stats["total"] += 1

                start_f = base_idx * base_chunk_frames
                start_s = base_idx * base_chunk_samples

                if start_s + extend_len_samples > len(db_audio):
                    continue

                seg_loud = db_loud[start_f: start_f + score_len_frames]
                if len(seg_loud) < score_len_frames:
                    continue

                # ----- Step 1: RMS mean/std prefilter (toggleable) -----
                a_cand = float(np.mean(seg_loud))
                s_cand = float(np.std(seg_loud))

                if ENABLE_RMS_PREFILTER:
                    if not (-3*avg_min <= a_cand <= 3*avg_max):
                        stats["rms"] += 1
                        continue

                    # # only apply std gate if input is not near-constant
                    # if std_in > 1e-4:
                    #     if not (std_min <= s_cand <= std_max):
                    #         stats["rms"] += 1
                    #         continue

                # ----- Step 2: loudness seam/head coherence (SOFT GATE + SCORE) -----
                head_n = int(max(seam_frame_count, seam_frame_count * loud_head_mult))
                head_n = min(head_n, len(seg_loud), len(input_loud))
                if head_n <= 0:
                    stats["loud_seam"] += 1
                    continue

                cand_loud0 = seg_loud[:head_n]
                inp_loud0  = input_loud[-head_n:]
                if cand_loud0.size < head_n or inp_loud0.size < head_n:
                    stats["loud_seam"] += 1
                    continue

                loud_floor = float(cfg["search"].get("loud_floor", 0.0))

                cand_head_mean = float(np.mean(cand_loud0))
                inp_tail_mean  = float(np.mean(inp_loud0))
                inp_tail_p90   = float(np.percentile(inp_loud0, 90))
                cand_head_p90  = float(np.percentile(cand_loud0, 90))

                # 1) Absolute head-energy sanity check:
                # Only reject if candidate head is *truly dead* relative to the input tail.
                # Use AND (both mean and p90 dead) and soften with a constant.
                energy_floor = 0.25 * loud_floor   # loud_floor=0.25 -> 0.0625
                if (cand_head_mean < energy_floor * inp_tail_mean) and (cand_head_p90 < energy_floor * inp_tail_p90):
                    stats["loud_seam"] += 1
                    continue

                # 2) Similarity-based loudness coherence: keep it, but do NOT hard-reject at loud_floor.
                loud_pen = float(np.mean((cand_loud0 - inp_loud0) ** 2))
                loud_coh = float(np.exp(-loud_pen * 25.0))

                # Optional: kill only catastrophic mismatches (constant threshold, not yaml)
                if loud_coh < 0.02:
                    stats["loud_seam"] += 1
                    continue

                # Candidate extension waveform (exactly what you'll save later)
                cand_ext = np.asarray(db_audio[start_s : start_s + extend_len_samples], dtype=np.float32)

                # RMS on that waveform with YOUR same params
                cand_ext_loud = librosa.feature.rms(
                    y=cand_ext, frame_length=rms_frame, hop_length=rms_hop, center=True
                )[0]

                frames_per_sec = max(8, int(round(float(sample_rate) / float(rms_hop))))
                ABS_SILENT_MEAN = 0.01

                bad_secs = 0
                for i in range(0, cand_ext_loud.size, frames_per_sec):
                    w = cand_ext_loud[i:i+frames_per_sec]
                    if w.size < 8: break
                    if float(np.mean(w)) < ABS_SILENT_MEAN:
                        bad_secs += 1

                if bad_secs >= 1:
                    stats["loud_ext"] += 1
                    continue

                # ----- Step 3: pitch seam checks -----
                cand_pitch0 = db_pitch[start_f: start_f + seam_pitch_frames]
                if cand_pitch0.size < seam_pitch_frames or input_pitch_tail.size < seam_pitch_frames:
                    stats["pitch_len"] += 1
                    continue

                cand_pitch_med = robust_median_hz(cand_pitch0)

                # if unvoiced seam and reject flag, reject
                if not (input_pitch_med > 0 and cand_pitch_med > 0):
                    if reject_unvoiced_seam:
                        stats["pitch_unvoiced"] += 1
                        continue

                # pitch register jump in cents (toggleable / penalty-capable)
                cents_jump = None
                pitch_reg_pen = 0.0

                if input_pitch_med > 0 and cand_pitch_med > 0:
                    cents_jump = abs(1200.0 * np.log2(cand_pitch_med / input_pitch_med))

                    if ENABLE_PITCH_REGISTER_GATE and (not PITCH_REGISTER_AS_PENALTY):
                        if cents_jump > pitch_cents_max:
                            stats["pitch_jump"] += 1
                            continue

                    if PITCH_REGISTER_AS_PENALTY:
                        pitch_reg_pen = float(np.clip(cents_jump / max(pitch_cents_max, 1e-6), 0.0, 2.0))

                pitch_coh = pitch_contour_sim(input_pitch_tail, cand_pitch0)
                if pitch_coh < pitch_floor:
                    stats["pitch_floor"] += 1
                    continue

                # ----- Step 4: chroma seam gate -----
                start_c = start_s // chroma_hop
                T = db_chroma_frames.shape[1]
                if start_c + max_lookback_chroma_frames > T:
                    continue

                cand_frames = db_chroma_frames[:, start_c: start_c + max_lookback_chroma_frames]
                if cand_frames.shape[1] == 0:
                    continue

                seam_block = cand_frames[:, :min(seam_c_frames, cand_frames.shape[1])]
                cand_chroma = chroma_mean_vec(seam_block)

                chroma_sim = float(np.dot(input_seam_chroma, cand_chroma))
                if chroma_sim < chroma_floor:
                    stats["chroma"] += 1
                    continue

                pitchish_coh = chroma_progression_sim(input_seam_frames, seam_block)

                coherence = (w_loud * loud_coh) + (w_pitch * pitch_coh) + (w_chroma_prog * pitchish_coh)
                if coherence < coherence_floor:
                    stats["coh"] += 1
                    continue

                # ----- Step 5: relevance aggregation (PLUS lookback-only trend match) -----
                rels = []
                wts = []
                best_dur = 0.0
                best_rel = -1.0
                best_trend_loud = 0.0
                best_trend_pitch = 0.0

                # knobs (add to yaml if you want; defaults here are sane)
                w_trend_loud = float(cfg["scoring"].get("w_trend_loud", 0.20))
                w_trend_pitch = float(cfg["scoring"].get("w_trend_pitch", 0.10))
                trend_downsample = int(cfg["search"].get("trend_downsample", 2))

                for dur, uemb in user_gru_queries.items():
                    dur = float(dur)

                    # ---- existing chroma-motion relevance (same as before) ----
                    nF = max(1, int((dur * sample_rate) / chroma_hop))
                    nF = min(nF, cand_frames.shape[1])
                    cemb = chroma_embed_motion(cand_frames[:, :nF])
                    rel_embed = cosine_sim(cemb, uemb)

                    # ---- NEW: trend match over the SAME lookback duration ----
                    n_frames = max(8, int(dur * frame_rate))

                    # candidate windows are the FIRST n_frames after start_f (because extension begins there)
                    cand_loud_d = seg_loud[:n_frames] if seg_loud.size >= n_frames else seg_loud
                    cand_pitch_d = db_pitch[start_f: start_f + n_frames]

                    # input windows are the LAST n_frames (precomputed)
                    inp_loud_d = input_loud_by_dur[dur]
                    inp_pitch_d = input_pitch_by_dur[dur]

                    # downsample for speed (keeps peaks/falls shape but cheaper)
                    if trend_downsample > 1:
                        cand_loud_d = cand_loud_d[::trend_downsample]
                        inp_loud_d = inp_loud_d[::trend_downsample]
                        cand_pitch_d = cand_pitch_d[::trend_downsample]
                        inp_pitch_d = inp_pitch_d[::trend_downsample]

                    t_loud = loud_trend_sim(inp_loud_d, cand_loud_d) if (inp_loud_d.size >= 8 and cand_loud_d.size >= 8) else 0.0
                    t_pitch = pitch_trend_sim(inp_pitch_d, cand_pitch_d) if (inp_pitch_d.size >= 8 and cand_pitch_d.size >= 8) else 0.0

                    trend = (w_trend_loud * t_loud) + (w_trend_pitch * t_pitch)

                    stats["trend_sum"] += float((w_trend_loud * t_loud) + (w_trend_pitch * t_pitch))
                    stats["trend_loud_sum"] += float(t_loud)
                    stats["trend_pitch_sum"] += float(t_pitch)
                    stats["trend_n"] += 1

                    # combine: relevance for this dur includes embedding relevance + trend bonus
                    r = float(rel_embed + trend)

                    rels.append(r)
                    wts.append(1.0 / max(dur, 1e-6))

                    if r > best_rel:
                        best_rel = r
                        best_dur = dur
                        best_trend_loud = float(t_loud)
                        best_trend_pitch = float(t_pitch)

                wts = np.asarray(wts, dtype=np.float32)
                wts = wts / (float(wts.sum()) + 1e-9)
                rel_agg = float(np.dot(wts, np.asarray(rels, dtype=np.float32)))

                score = (current_alpha * rel_agg) + (current_beta * coherence) + (0.30 * chroma_sim)

                # apply optional pitch register penalty
                if PITCH_REGISTER_AS_PENALTY and (cents_jump is not None):
                    score = score - (PITCH_REGISTER_PENALTY_W * pitch_reg_pen)

                if score > best_seen:
                    best_seen = score
                    if use_tqdm:
                        it.set_postfix(best=fmt(best_seen, 4))

                stats["kept"] += 1

                counter += 1
                item = {
                    "score": score,
                    "db_idx": int(base_idx),
                    "start_s_db": int(start_s),

                    "coherence": float(coherence),
                    "relevance": float(rel_agg),
                    "chroma": float(chroma_sim),
                    "lookback": float(best_dur),

                    # diagnostics
                    "pitchish": float(pitchish_coh),
                    "loud_coh": float(loud_coh),
                    "pitch_coh": float(pitch_coh),
                    "pitch_med_in": float(input_pitch_med),
                    "pitch_med_cand": float(cand_pitch_med),
                    "cents_jump": float(cents_jump) if cents_jump is not None else None,
                    "pitch_reg_pen": float(pitch_reg_pen),

                    # diversify later
                    "_div_vec": cand_chroma.astype(np.float32),
                    "trend_loud": float(best_trend_loud),
                    "trend_pitch": float(best_trend_pitch),
                }

                tup = (score, counter, item)
                if len(top_heap) < heap_k:
                    heapq.heappush(top_heap, tup)
                else:
                    if score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, tup)

            # store attempt logs
            attempt_log = {
                "params": debug_params,
                "stats": stats,
            }
            debug["attempts"].append(attempt_log)

            if stats["trend_n"] > 0:
                stats["trend_mean"] = float(stats["trend_sum"] / stats["trend_n"])
                stats["trend_loud_mean"] = float(stats["trend_loud_sum"] / stats["trend_n"])
                stats["trend_pitch_mean"] = float(stats["trend_pitch_sum"] / stats["trend_n"])

            if LOG_STATS:
                print(f"[{label}] params:", debug_params)
                print(f"[{label}] stats:", stats)

            return [t[2] for t in sorted(top_heap, key=lambda x: x[0], reverse=True)]

        use_tqdm = bool(cfg.get("search", {}).get("use_tqdm", False))

        all_candidates: list[dict] = []

        for attempt in range(attempts):
            if attempt == 0:
                params = dict(
                    current_alpha=alpha,
                    current_beta=beta,
                    coherence_floor=coherence_min_threshold,
                    chroma_floor=float(cfg["search"].get("chroma_floor_strict", 0.65)),
                    rms_tol_mult=1.0,
                    label="Strict search",
                )
            else:
                params = dict(
                    current_alpha=float(cfg["search"].get("fallback_alpha", 0.7)),
                    current_beta=float(cfg["search"].get("fallback_beta", 0.3)),
                    coherence_floor=float(cfg["search"].get("fallback_coherence_floor", 0.0)),
                    chroma_floor=float(cfg["search"].get("chroma_floor_fallback", 0.50)),
                    rms_tol_mult=float(cfg["search"].get("fallback_rms_mult", 3.0)),
                    label="Fallback search",
                )

            cand = collect_candidates(use_tqdm=use_tqdm, **params)

            # Keep strict candidates; add fallback candidates instead of overwriting
            all_candidates.extend(cand)

            # If strict already has plenty, don't waste time on fallback
            if attempt == 0 and len(all_candidates) >= top_k_results * 2:
                break

        # Dedupe by db_idx (keep best score), then sort by score descending
        if all_candidates:
            best_by_idx: dict[int, dict] = {}
            for m in all_candidates:
                idx = int(m["db_idx"])
                if (idx not in best_by_idx) or (float(m["score"]) > float(best_by_idx[idx]["score"])):
                    best_by_idx[idx] = m
            all_candidates = sorted(best_by_idx.values(), key=lambda x: float(x["score"]), reverse=True)

        post = dict(
            requested=int(top_k_results),
            all_candidates=int(len(all_candidates)),        # after strict+fallback merge + dedupe
            selected_initial=0,                              # after apply_ban/diverse
            selected_final=0,                                # after fill + slice
            returned=0,                                      # results_manifest length
            ban_radius=int(ban_radius),
            diverse_enable=bool(diverse_enable),
            diverse_sim_max=float(diverse_sim_max),
            ban_skipped=0,
            diverse_skipped=0,
            fill_added=0,
        )

        # ---------- NEW: DTW reranker (runs only on shortlist) ----------
        if dtw_enable and all_candidates:
            # shortlist
            all_candidates_short = all_candidates[:dtw_rerank_n]

            for m in all_candidates_short:
                dur = float(m.get("lookback", 0.0))
                if dur <= 0:
                    continue

                n_frames = max(8, int(dur * frame_rate))

                # input: last dur (precomputed ONLY for lookback periods)
                inp_l = input_loud_by_dur.get(dur, None)
                if inp_l is None:
                    continue

                # candidate: first dur starting at db_idx
                start_f = int(m["db_idx"]) * base_chunk_frames
                cand_seg = db_loud[start_f: start_f + score_len_frames]
                cand_l = cand_seg[:n_frames] if cand_seg.size >= n_frames else cand_seg
                if cand_l.size < 8:
                    continue

                # downsample + log + zscore
                inp = zscore(np.log1p(np.maximum(inp_l[::dtw_downsample], 0.0)))
                cand = zscore(np.log1p(np.maximum(cand_l[::dtw_downsample], 0.0)))

                cost = dtw_cost(inp, cand, band=dtw_band)
                dtw_gamma = float(cfg["search"].get("dtw_gamma", 0.35))
                dtw_sim = float(np.exp(-dtw_gamma * cost))

                m["dtw_loud"] = dtw_sim
                m["score"] = float(m["score"] + w_dtw * dtw_sim)

            # re-sort after DTW bump
            all_candidates = sorted(all_candidates, key=lambda x: float(x["score"]), reverse=True)
        # ---------------------------------------------------------------

        if not all_candidates:
            post["returned"] = 0
            debug["post"] = post
            return {
                "ok": False,
                "error": (
                    "FAILURE: no candidates passed filters. "
                    "Try: increase filters.rms_tolerance_pct, lower chroma floors, "
                    "increase search.pitch_cents_max, lower search.pitch_floor, or widen fallback multipliers."
                ),
                "debug": debug,
            }        
        
        def apply_ban(cands: list[dict], ban: int, k: int) -> list[dict]:
            out: list[dict] = []
            for m in cands:
                if any(abs(m["db_idx"] - x["db_idx"]) <= ban for x in out):
                    continue
                out.append(m)
                if len(out) >= k:
                    break
            return out

        def apply_ban_diverse(cands: list[dict], ban: int, k: int, sim_max: float, post: dict) -> list[dict]:
            out: list[dict] = []
            out_vecs: list[np.ndarray] = []

            for m in cands:
                idx = int(m["db_idx"])

                # ban radius
                if any(abs(idx - int(x["db_idx"])) <= ban for x in out):
                    post["ban_skipped"] += 1
                    continue

                # diversity
                v = m.get("_div_vec", None)
                if v is not None and out_vecs:
                    # NOTE: dot assumes vectors are normalized; your chroma_mean_vec is normalized, so OK
                    if any(float(np.dot(v, ov)) > sim_max for ov in out_vecs):
                        post["diverse_skipped"] += 1
                        continue

                out.append(m)
                if v is not None:
                    out_vecs.append(v)

                if len(out) >= k:
                    break

            return out

        if diverse_enable:
            selected = apply_ban_diverse(all_candidates, ban_radius, top_k_results, diverse_sim_max, post)
        else:
            selected = apply_ban(all_candidates, ban_radius, top_k_results)

        post["selected_initial"] = int(len(selected))

        # Fill to K but still respect ban + diversity
        if len(selected) < top_k_results:
            picked = {int(m["db_idx"]) for m in selected}
            out_vecs = [m.get("_div_vec", None) for m in selected if m.get("_div_vec", None) is not None]

            for m in all_candidates:
                idx = int(m["db_idx"])
                if idx in picked:
                    continue

                if any(abs(idx - int(x["db_idx"])) <= ban_radius for x in selected):
                    continue

                v = m.get("_div_vec", None)
                if diverse_enable and v is not None and out_vecs:
                    if any(float(np.dot(v, ov)) > diverse_sim_max for ov in out_vecs):
                        continue

                selected.append(m)
                post["fill_added"] += 1   # ✅ count it
                picked.add(idx)
                if v is not None:
                    out_vecs.append(v)

                if len(selected) >= top_k_results:
                    break

        # Never overwrite this with ban=0 logic
        selected = selected[:top_k_results]
        selected = sorted(selected, key=lambda x: float(x["score"]), reverse=True)

        post["selected_final"] = int(len(selected))

        print(f"\nRetrieval Complete. Returning {len(selected)} matches (requested {top_k_results}).")

        output_dir.mkdir(parents=True, exist_ok=True)

        results_manifest: list[dict] = []

        for rank, match in enumerate(selected, start=1):
            start_s_db = int(match["start_s_db"])
            retrieved = np.array(db_audio[start_s_db: start_s_db + extend_len_samples], dtype=np.float32)

            inp = np.array(y_in, dtype=np.float32).copy()
            ret = retrieved.copy()

            L = crossfade_audio(inp, ret, crossfade_time_samples)

            if L == 0:
                _ = np.concatenate([inp, ret])
            else:
                _ = np.concatenate([inp[:-L], inp[-L:] + ret[:L], ret[L:]])

            final_len_s = (len(y_in) + len(ret)) / sample_rate
            len_tag = f"{final_len_s:.2f}s"

            match["db_id_4s"] = int(match["db_idx"]) // group

            extension_name = (
                f"Rank{rank}_IDX_{match['db_id_4s']}_LEN_{len_tag}_Score_{match['score']:.4f}_extension.wav"
            )
            extension_path = output_dir / extension_name
            sf.write(str(extension_path), ret, sample_rate)

            results_manifest.append(
                {
                    "rank": int(rank),
                    "db_idx": int(match["db_idx"]),
                    "db_id_4s": int(match["db_idx"]) // group,
                    "start_s_db": int(start_s_db),
                    "score": float(match["score"]),
                    "coherence": float(match.get("coherence", 0.0)),
                    "relevance": float(match.get("relevance", 0.0)),
                    "chroma": float(match.get("chroma", 0.0)),
                    "lookback": float(match.get("lookback", 0.0)),
                    "pitchish": float(match.get("pitchish", 0.0)),
                    "loud_coh": float(match.get("loud_coh", 0.0)),
                    "pitch_coh": float(match.get("pitch_coh", 0.0)),
                    "pitch_med_in": float(match.get("pitch_med_in", 0.0)),
                    "pitch_med_cand": float(match.get("pitch_med_cand", 0.0)),
                    "cents_jump": match.get("cents_jump", None),
                    "pitch_reg_pen": match.get("pitch_reg_pen", 0.0),
                    "extension_wav": extension_name,
                    "stitched_wav": None,
                    "arranged_wav": None,
                    "arranged_png": None,
                    "trend_loud": float(match.get("trend_loud", 0.0)),
                    "trend_pitch": float(match.get("trend_pitch", 0.0)),
                    "dtw_loud": float(match.get("dtw_loud", 0.0)),
                }
            )

        # ✅ ADD HERE (after results_manifest is built, before return)
        post["returned"] = int(len(results_manifest))
        debug["post"] = post    

        return {
            "ok": True,
            "input": str(input_path),
            "target_duration": float(target_duration),
            "top_k": len(results_manifest),
            "output_dir": str(output_dir),
            "results": results_manifest,
            "debug": debug,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    print(run_audio_rag())