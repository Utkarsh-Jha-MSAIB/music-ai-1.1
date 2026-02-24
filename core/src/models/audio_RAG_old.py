import numpy as np
import librosa
import sys
from pathlib import Path
import os
import shutil
from tqdm import tqdm
import heapq
import soundfile as sf
from typing import Optional, Dict, Any

#warnings.filterwarnings("ignore", module="librosa")


try:
    import torchcrepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

# Input file location & other parameters to always update

from configs.load_config import load_yaml


# Start for the function

def audio_to_chroma(audio_waveform, *, sr: int, chroma_hop: int) -> np.ndarray:
    chromagram = librosa.feature.chroma_cqt(y=audio_waveform, sr=sr, hop_length=chroma_hop)
    chroma_vector = np.mean(chromagram, axis=1)
    norm = np.linalg.norm(chroma_vector)
    return (chroma_vector / norm + 1e-9) if norm > 0 else (chroma_vector + 1e-9)


def pitch_to_chroma(f0_curve):
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12) + 1e-9
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return (chroma_vector / norm + 1e-9) if norm > 0 else (chroma_vector + 1e-9)


def gru_embedding_relevance(p_embed, q_embed):
    """Calculates cosine similarity between GRU embedding vectors"""
    p = p_embed.flatten()
    q = q_embed.flatten()

    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)

    if norm_p == 0 or norm_q == 0:
        return 0.0

    return np.dot(p, q) / (norm_p * norm_q)


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

    dtype = audio_out.dtype
    fade_out = np.linspace(1.0, 0.0, L, dtype=dtype)
    fade_in  = np.linspace(0.0, 1.0, L, dtype=dtype)

    audio_out[-L:] *= fade_out
    audio_in[:L]   *= fade_in
    return L



def cleanup_temp_files(temp_name: str, *, data_dir: Path) -> None:
    temp_files = [
        data_dir / f'pitch_{temp_name}.npy',
        data_dir / f'loudness_{temp_name}.npy',
        data_dir / f'audio_{temp_name}.npy'
    ]
    cleaned = 0
    for f in temp_files:
        if f.exists():
            os.remove(f)
            cleaned += 1
    if cleaned > 0:
        pass


def get_gru_embedding(
    audio_segment: np.ndarray,
    *,
    gru_ckpt: Path,
    sr: int,
    chroma_hop: int,
    ) -> np.ndarray:
    # currently fallback-to-chroma always (until you implement real GRU)
    chroma = audio_to_chroma(audio_segment, sr=sr, chroma_hop=chroma_hop)
    return chroma[:12]


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
    cfg_local = cfg_override or load_yaml(repo_root / "configs" / "audio_rag.yaml")
    project_root = (repo_root / cfg_local["paths"].get("project_root", ".")).resolve()

    # apply overrides
    if input_filename is not None:
        cfg_local.setdefault("io", {})
        cfg_local["io"]["input_filename"] = input_filename
    if target_duration is not None:
        cfg_local.setdefault("rag", {})
        cfg_local["rag"]["target_duration_sec"] = float(target_duration)
    if top_k is not None:
        cfg_local.setdefault("search", {})
        cfg_local["search"]["top_k_results"] = int(top_k)

   # --- derive paths from cfg_local ---
    core_dir = (project_root / cfg_local["paths"]["core_dir"]).resolve()
    data_dir = (project_root / cfg_local["paths"]["features_dir"]).resolve()
    output_dir = (project_root / cfg_local["paths"]["outputs_dir"]).resolve()

    # sys.path for src imports (must happen before importing/using src.*)
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    #from src.models import perform_music as neural_band

    # IO
    input_path = Path(cfg_local["io"]["input_filename"])
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    # RAG params
    target_duration = float(cfg_local["rag"]["target_duration_sec"])
    base_chunk_duration = float(cfg_local["rag"]["base_chunk_duration_sec"])
    temp_leader_name = str(cfg_local["models"]["temp_leader_name"])

    db_chunk_sec = 4.0
    group = max(1, int(round(db_chunk_sec / base_chunk_duration)))

    # model paths
    gru_ckpt = (project_root / cfg_local["models"]["gru_model_checkpoint"]).resolve()

    # audio params
    sample_rate = int(cfg_local["audio"]["sample_rate"])
    frame_rate  = int(cfg_local["audio"]["frame_rate"])
    crossfade_time_samples = int(float(cfg_local["audio"]["crossfade_time_sec"]) * sample_rate)

    # seam / lookback
    lookback_options = list(cfg_local["rag"]["lookback_options_sec"])
    seam_frame_count = int(cfg_local["rag"]["seam_frame_count"])

    # scoring
    alpha = float(cfg_local["scoring"]["alpha"])
    beta  = float(cfg_local["scoring"]["beta"])
    coherence_min_threshold = float(cfg_local["scoring"]["coherence_min_threshold"])

    # filters / search
    rms_tolerance_pct = float(cfg_local["filters"]["rms_tolerance_pct"])
    top_k_results = int(cfg_local["search"]["top_k_results"])
    stride = int(cfg_local["search"]["stride"])
    attempts = int(cfg_local["search"].get("attempts", 2))

    # librosa params
    pyin_hop = int(cfg_local["librosa"]["pyin_hop_length"])
    rms_frame = int(cfg_local["librosa"]["rms_frame_length"])
    rms_hop   = int(cfg_local["librosa"]["rms_hop_length"])
    chroma_hop = int(cfg_local["librosa"]["chroma_hop_length"])
    pyin_fmin = float(cfg_local["librosa"]["pyin_fmin"])
    pyin_fmax = float(cfg_local["librosa"]["pyin_fmax"])

    # ---- Consistency check: shared frame grid ----
    # This pipeline assumes loudness + pitch are indexed on the SAME frame axis.
    if rms_hop != pyin_hop:
        raise ValueError(
            f"Config mismatch: rms_hop({rms_hop}) != pyin_hop({pyin_hop}).\n"
            f"This pipeline slices db_loudness and db_pitches using the SAME frame indices,\n"
            f"so set librosa.rms_hop_length == librosa.pyin_hop_length."
        )

    expected_fps = sample_rate / rms_hop
    if abs(frame_rate - expected_fps) > 1e-6:
        raise ValueError(
            f"Config mismatch:\n"
            f"  frame_rate={frame_rate}\n"
            f"  expected=sample_rate/rms_hop={expected_fps}\n"
            f"Fix audio.frame_rate or hop lengths to be consistent."
        )

    # database
    db_instrument = str(cfg_local["database"]["instrument"])
    pitch_path = (data_dir / cfg_local["database"]["pitch_file"].format(instrument=db_instrument)).resolve()
    loud_path  = (data_dir / cfg_local["database"]["loudness_file"].format(instrument=db_instrument)).resolve()
    audio_path = (data_dir / cfg_local["database"]["audio_file"].format(instrument=db_instrument)).resolve()
 
    print(f"Starting Audio RAG Pipeline (Target: {target_duration}s)...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    for p in (pitch_path, loud_path, audio_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing DB file: {p}")

    try:
        # 1. Load user input
        y_user_input, sr = librosa.load(str(input_path), sr=sample_rate)

        # --- INPUT seam features (this is the “G” part in RAG: grounding to input) ---
        input_pitch, _, _ = librosa.pyin(y_user_input, fmin=pyin_fmin, fmax=pyin_fmax, sr=sample_rate, hop_length=pyin_hop)
        input_pitch = np.nan_to_num(input_pitch)

        input_loud = librosa.feature.rms(y=y_user_input, frame_length=rms_frame, hop_length=rms_hop, center=True)[0]
        input_loud = np.nan_to_num(input_loud)

        # last SEAM frames of input
        last_pitch_end = input_pitch[-seam_frame_count:]
        last_loudness_end = input_loud[-seam_frame_count:]

        # ---- input seam chroma (computed once) ----
        seam_sec = 1.5
        seam_samples = int(seam_sec * sample_rate)
        input_tail = y_user_input[-seam_samples:] if len(y_user_input) >= seam_samples else y_user_input
        input_seam_chroma = audio_to_chroma(input_tail, sr=sample_rate, chroma_hop=chroma_hop)  # (12,)

        input_duration = len(y_user_input) / sample_rate

        if target_duration <= input_duration:
            raise ValueError(
                f"Target duration ({target_duration}s) must be greater than input duration ({input_duration:.1f}s)"
                )

        search_duration = target_duration - input_duration

        if search_duration <= 0:
            raise ValueError("search_duration must be > 0")

        if int(search_duration * sample_rate) < 1:
            raise ValueError("search_duration too small; produces 0 samples")

        print(
            f"   Input duration: {input_duration:.1f}s. Searching for single {search_duration:.1f}s extension segment")

        # Calculate Reference RMS for Dynamic Energy Filter
        avg_input_rms = float(np.mean(input_loud))
        std_input_rms = float(np.std(input_loud))

        # Define Tolerance Bands for both metrics
        loud_avg_min = avg_input_rms * (1 - rms_tolerance_pct)
        loud_avg_max = avg_input_rms * (1 + rms_tolerance_pct)
        loud_std_min = std_input_rms * (1 - rms_tolerance_pct)
        loud_std_max = std_input_rms * (1 + rms_tolerance_pct)

        print(f"   Avg Loudness Range: [{loud_avg_min:.4f}, {loud_avg_max:.4f}]")
        print(f"   STD Loudness Range: [{loud_std_min:.4f}, {loud_std_max:.4f}]")

        # Database Loading (MEMMAP)
        db_pitches  = np.load(pitch_path, mmap_mode="r", allow_pickle=False).reshape(-1)
        db_loudness = np.load(loud_path,  mmap_mode="r", allow_pickle=False).reshape(-1)
        db_audio    = np.load(audio_path, mmap_mode="r", allow_pickle=False).reshape(-1)

        chroma_frames_path = (data_dir / cfg_local["database"]["chroma_frames_file"].format(instrument=db_instrument)).resolve()
        if not chroma_frames_path.exists():
            raise FileNotFoundError(f"Missing DB chroma frames file: {chroma_frames_path}")

        # ---- DB consistency checks (catches "quietly worse retrieval") ----
        expected_frames_from_audio = (len(db_audio) + rms_hop - 1) // rms_hop  # ceil

        def _warn_or_raise(name, got, exp, tol):
            if abs(got - exp) > tol:
                raise ValueError(
                    f"DB mismatch for {name}:\n"
                    f"  got={got}\n"
                    f"  expected~={exp} (ceil(len(db_audio)/hop))\n"
                    f"This usually means the DB .npy files were precomputed with a different hop_length.\n"
                    f"Regenerate DB features with the same sample_rate + hop_length as YAML."
                )

        _warn_or_raise("db_loudness (frames)", len(db_loudness), expected_frames_from_audio, tol=10)
        _warn_or_raise("db_pitches (frames)",  len(db_pitches),  expected_frames_from_audio, tol=20)

        db_chroma_frames = np.load(chroma_frames_path, mmap_mode="r", allow_pickle=False)  # shape (12, T)
        
        print("Loaded db_chroma_frames:", db_chroma_frames.shape, db_chroma_frames.dtype)
        if db_chroma_frames.shape[0] != 12:
            raise ValueError(f"Expected chroma frames shape (12, T), got {db_chroma_frames.shape}")

        expected_T = (len(db_audio) + chroma_hop - 1) // chroma_hop
        print("Expected chroma frames (approx):", expected_T)

        # allow a tiny tolerance because librosa framing can differ by a frame or two
        if abs(db_chroma_frames.shape[1] - expected_T) > 5:
            print("WARNING: chroma_frames length doesn't match audio length/hop. Check precompute hop_length and SR.")

        # 2. Analyze user input (we still use input_duration for lengths),
        # but we DO NOT take seam features from DB using user offsets.

        # (DB-internal coherence) seam reference will be computed per-candidate
        # inside the retrieval loop, from the DB region right before that candidate.

        # Multiple Relevance Queries (Generate GRU embeddings)
        user_gru_queries = {}
        for dur in lookback_options:
            if dur <= input_duration:
                start_sample = len(y_user_input) - int(dur * sample_rate)
                query_audio = y_user_input[start_sample:]
                user_gru_queries[dur] = get_gru_embedding(
                    query_audio,
                    gru_ckpt=gru_ckpt,
                    sr=sample_rate,
                    chroma_hop=chroma_hop,
                    )

        if not user_gru_queries:
            raise ValueError("No GRU queries could be formed (lookback options exceed input duration).")

        # 3. Single-Step Retrieval Search
        # search_len_samples = max(1, int(search_duration * sample_rate))
        # search_len_frames  = max(1, int(search_duration * frame_rate))
        # base_chunk_frames = int(base_chunk_duration * frame_rate)

        # 3. Single-Step Retrieval Search
        # We will SCORE candidates using lookback-matched windows,
        # but we will OUTPUT the full extension to hit target_duration.

        extend_len_samples = max(1, int(search_duration * sample_rate))   # final extension length (output)

        # prepare lookback frame lengths for candidate-side matching
        user_durs = sorted(float(d) for d in user_gru_queries.keys())     # only durs that actually exist
        max_lookback_sec = user_durs[-1]
        max_lookback_frames = max(1, int((max_lookback_sec * sample_rate) / chroma_hop))

        # For pitch/loudness candidate-side scoring window, use the same max lookback duration (in frames)
        score_len_frames = max(1, int(max_lookback_sec * frame_rate))

        base_chunk_frames = int(base_chunk_duration * frame_rate)
        if base_chunk_frames <= 0:
            raise ValueError("base_chunk_duration_sec too small; base_chunk_frames became 0")

        # IMPORTANT: sampling range is about choosing start points.
        # Use score window (not full extension) so you don't artificially shrink the search space.
        num_base_chunks = max(1, int((max_lookback_sec) / base_chunk_duration))
        num_db_base_chunks = len(db_pitches) // base_chunk_frames
        sampling_range = num_db_base_chunks - num_base_chunks
        if sampling_range <= 0:
            raise FileNotFoundError(f"ERROR: Database is too short to find a {max_lookback_sec:.1f}s scoring window")
            
        # if base_chunk_frames <= 0:
        #     raise ValueError("base_chunk_duration_sec too small; base_chunk_frames became 0")

        # num_base_chunks = int(search_duration / base_chunk_duration)
        # num_db_base_chunks = len(db_pitches) // base_chunk_frames

        # # Try Exhaustive Search
        # sampling_range = num_db_base_chunks - num_base_chunks
        # if sampling_range <= 0: raise FileNotFoundError(f"ERROR: Database is too short to find a {search_duration:.1f}s segment")
        # sample_indices = range(sampling_range) 
        sample_indices = range(0, sampling_range, stride)

        # Initialize Top K Match Tracker
        # top_k_matches = []
        # best_composite_score = -1.0
        top_k_heap = []  # min-heap of (score, counter, item)
        best_composite_score = -1.0

        # Keep more than top_k_results so ban_radius doesn't shrink you to 2-3
        HEAP_MULT = int(cfg_local.get("search", {}).get("heap_mult", 6))  # add to yaml if you want
        HEAP_K = max(top_k_results * HEAP_MULT, top_k_results)

        counter = 0
        def push_top_k(item, k=HEAP_K):
            nonlocal counter
            counter += 1
            s = item["score"]
            tup = (s, counter, item)
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, tup)
            else:
                if s > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, tup)

        for attempt in range(attempts):
            top_k_heap = []
            best_composite_score = -1.0
            top_k_matches = []

            if attempt == 0:
                current_alpha = alpha
                current_beta = beta
                coherence_floor = coherence_min_threshold
                chroma_floor = 0.65
                voiced_frac_req = 0.25
                rms_tol_mult = 1.0
                desc_label = "Strict Search (GRU/Coherence)"
            else:
                current_alpha = 0.7
                current_beta = 0.3   #Since finding higher coherence is bigger issue
                coherence_floor = 0.0
                chroma_floor = 0.45          # was 0.65
                voiced_frac_req = 0.10       # was 0.25
                rms_tol_mult = 2.0           # widen rms window
                desc_label = "Fallback Search"
                print(f"      Strict search failed. Retrying with α={current_alpha}, β={current_beta}")

            use_tqdm = bool(cfg_local.get("search", {}).get("use_tqdm", False))
            search_iterator = tqdm(sample_indices, desc=desc_label) if use_tqdm else sample_indices

            for base_idx in search_iterator:
                start_f_db = base_idx * base_chunk_frames
                start_s_db = base_idx * int(base_chunk_duration * sample_rate)

                # Retrieve features and audio for the candidate segment
                #db_segment_audio = db_audio[start_s_db: start_s_db + search_len_samples]
                # 0) bounds check first (no slicing yet)
                #if start_s_db + search_len_samples > len(db_audio):
                if start_s_db + extend_len_samples > len(db_audio):
                    continue

                # 1) Slice loudness once (needed for RMS filter + loudness seam)
                # db_segment_loudness = db_loudness[start_f_db: start_f_db + search_len_frames]
                # if len(db_segment_loudness) < search_len_frames:
                #     continue

                # cand_loud = db_segment_loudness
                # avg_candidate_loud = float(np.mean(cand_loud))
                # std_candidate_loud = float(np.std(cand_loud))

                # if not (loud_avg_min <= avg_candidate_loud <= loud_avg_max):
                #     continue
                # if not (loud_std_min <= std_candidate_loud <= loud_std_max):
                #     continue

                # # 2) Only now slice pitch (needed for seam coherence)
                # db_segment_pitch = db_pitches[start_f_db: start_f_db + search_len_frames]

                # 1) Slice loudness for SCORING window (near seam), not full extension
                db_segment_loudness = db_loudness[start_f_db: start_f_db + score_len_frames]
                if len(db_segment_loudness) < score_len_frames:
                    continue

                avg_candidate_loud = float(np.mean(db_segment_loudness))
                std_candidate_loud = float(np.std(db_segment_loudness))

                # if not (loud_avg_min <= avg_candidate_loud <= loud_avg_max):
                #     continue
                # if not (loud_std_min <= std_candidate_loud <= loud_std_max):
                #     continue

                # --- RMS filter (attempt-aware widening) ---
                avg_min = avg_input_rms * (1 - rms_tolerance_pct * rms_tol_mult)
                avg_max = avg_input_rms * (1 + rms_tolerance_pct * rms_tol_mult)
                std_min = std_input_rms * (1 - rms_tolerance_pct * rms_tol_mult)
                std_max = std_input_rms * (1 + rms_tolerance_pct * rms_tol_mult)

                if not (avg_min <= avg_candidate_loud <= avg_max):
                    continue
                if not (std_min <= std_candidate_loud <= std_max):
                    continue

                # 2) Slice pitch for SCORING window (near seam)
                db_segment_pitch = db_pitches[start_f_db: start_f_db + score_len_frames]
                if len(db_segment_pitch) < score_len_frames:
                    continue
                # if len(db_segment_pitch) < search_len_frames:
                #     continue

                candidate_pitch_start = db_segment_pitch[:seam_frame_count]
                candidate_loudness_start = db_segment_loudness[:seam_frame_count]
                if (
                    len(candidate_pitch_start) < seam_frame_count
                    or len(candidate_loudness_start) < seam_frame_count
                    or len(last_pitch_end) < seam_frame_count
                    or len(last_loudness_end) < seam_frame_count
                ):
                    continue

                # pitch_penalty = np.mean((candidate_pitch_start - last_pitch_end) ** 2)
                # loudness_penalty = np.mean((candidate_loudness_start - last_loudness_end) ** 2)
                
                # ---- weighted seam penalties (emphasize the seam moment) ----
                w = np.linspace(1.0, 0.2, seam_frame_count)
                w = w / (w.sum() + 1e-9)

                dp = (candidate_pitch_start - last_pitch_end)
                dl = (candidate_loudness_start - last_loudness_end)

                pitch_penalty = float(np.sum(w * (dp * dp)))
                loudness_penalty = float(np.sum(w * (dl * dl)))

                #pitch_coherence = np.exp(-pitch_penalty * 50)

                # ---- pitch coherence OFF (pyin is unreliable for polyphonic guitar) ----
                pitch_coherence = 1.0

                # loudness seam continuity only
                loudness_penalty = float(np.mean((candidate_loudness_start - last_loudness_end) ** 2))
                loudness_coherence = float(np.exp(-loudness_penalty * 30))

                coherence_score = loudness_coherence

                # Coherence Floor Guard
                if coherence_score < coherence_floor:
                    continue

                # 2. Relevance Score - GRU Embedding Match (Also tried cosine similarity logic here)
                # candidate_gru_embed = get_gru_embedding(
                #     db_segment_audio,
                #     gru_ckpt=gru_ckpt,
                #     sr=sample_rate,
                #     chroma_hop=chroma_hop,
                # )

                # --- FAST relevance: use precomputed chroma frames (no librosa per-candidate) ---
                # start_c = start_s_db // chroma_hop
                # end_c = (start_s_db + search_len_samples + chroma_hop - 1) // chroma_hop  # ceil

                # # Clamp
                # T = db_chroma_frames.shape[1]
                # if start_c >= T:
                #     continue
                # end_c = min(end_c, T)
                # if end_c <= start_c:
                #     continue

                # cand_frames = db_chroma_frames[:, start_c:end_c]  # (12, Tc)
                # if cand_frames.shape[1] == 0:
                #     continue
                
                # # ---- chroma seam gate (avoid harmonic/key clashes) ----
                # seam_c_frames = max(1, int((seam_sec * sample_rate) / chroma_hop))
                # cand_seam = cand_frames[:, :min(seam_c_frames, cand_frames.shape[1])]
                # cand_chroma = np.mean(cand_seam, axis=1)
                # cand_chroma = cand_chroma / (np.linalg.norm(cand_chroma) + 1e-9)

                # chroma_sim = float(np.dot(input_seam_chroma, cand_chroma))
                # if chroma_sim < 0.65:
                #     continue

                # candidate_gru_embed = np.mean(cand_frames, axis=1)
                # norm = np.linalg.norm(candidate_gru_embed)
                # candidate_gru_embed = (candidate_gru_embed / norm + 1e-9) if norm > 0 else (candidate_gru_embed + 1e-9)


                # max_relevance = -1.0
                # best_lookback_dur = 0.0

                # for dur, user_gru_embed in user_gru_queries.items():
                #     relevance = gru_embedding_relevance(candidate_gru_embed, user_gru_embed)
                #     if relevance > max_relevance:
                #         max_relevance = relevance
                #         best_lookback_dur = dur

                # # 3. Calculating a composite score
                # # composite_score = (current_alpha * max_relevance) + (current_beta * coherence_score)
                # composite_score = (current_alpha * max_relevance) + (current_beta * coherence_score) + (0.15 * chroma_sim)

                # --- FAST relevance: use precomputed chroma frames (no librosa per-candidate) ---
                start_c = start_s_db // chroma_hop
                T = db_chroma_frames.shape[1]
                if start_c + max_lookback_frames > T:
                    continue

                cand_frames_max = db_chroma_frames[:, start_c : start_c + max_lookback_frames]  # (12, max_lookback_frames)
                if cand_frames_max.shape[1] == 0:
                    continue

                # ---- chroma seam gate (avoid harmonic/key clashes) ----
                seam_c_frames = max(1, int((seam_sec * sample_rate) / chroma_hop))
                cand_seam = cand_frames_max[:, :min(seam_c_frames, cand_frames_max.shape[1])]
                cand_chroma = np.mean(cand_seam, axis=1)
                cand_chroma = cand_chroma / (np.linalg.norm(cand_chroma) + 1e-9)

                chroma_sim = float(np.dot(input_seam_chroma, cand_chroma))
                if chroma_sim < chroma_floor:
                    continue

                # ---- relevance: compare SAME duration windows (input lookback dur vs candidate first dur) ----
                rels = []
                weights = []
                best_lookback_dur = 0.0
                best_rel = -1.0

                for dur, user_gru_embed in user_gru_queries.items():
                    dur = float(dur)
                    nF = max(1, int((dur * sample_rate) / chroma_hop))
                    nF = min(nF, cand_frames_max.shape[1])

                    cand_embed = np.mean(cand_frames_max[:, :nF], axis=1)
                    cand_embed = cand_embed / (np.linalg.norm(cand_embed) + 1e-9)

                    rel = float(gru_embedding_relevance(cand_embed, user_gru_embed))
                    rels.append(rel)

                    # weight shorter lookbacks more (seam relevance)
                    weights.append(1.0 / max(dur, 1e-6))

                    if rel > best_rel:
                        best_rel = rel
                        best_lookback_dur = dur

                weights = np.asarray(weights, dtype=np.float32)
                weights = weights / (weights.sum() + 1e-9)
                relevance_agg = float(np.dot(weights, np.asarray(rels, dtype=np.float32)))

                # 3. composite score
                composite_score = (current_alpha * relevance_agg) + (current_beta * coherence_score) + (0.15 * chroma_sim)

                # Track Top K Matches
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    if use_tqdm:
                        search_iterator.set_postfix(score=f"{best_composite_score:.4f}")
        
                # Add the result to the list of top matches
                push_top_k({
                    "score": composite_score,
                    "db_idx": base_idx,
                    "start_s_db": start_s_db,                            # ✅ exact audio start sample used
                    "db_id_4s": base_idx // group,                       # ✅ original 4s ID (4/2=2)
                    "coherence": coherence_score,
                    "relevance": relevance_agg,
                    "lookback": best_lookback_dur
                })

            # After each attempt, sort all matches found so far and prune to TOP_K
            # top_k_matches.sort(key=lambda x: x['score'], reverse=True)
            # top_k_matches = top_k_matches[:top_k_results]

            #top_k_matches = [it for (_, _, it) in sorted(top_k_heap, key=lambda x: x[0], reverse=True)]
            top_k_matches = [t[2] for t in sorted(top_k_heap, key=lambda x: x[0], reverse=True)]

            ban = int(cfg_local.get("search", {}).get("ban_radius", 20))  # or just ban=50
            filtered = []
            for m in top_k_matches:
                if any(abs(m["db_idx"] - x["db_idx"]) <= ban for x in filtered):
                    continue
                filtered.append(m)
                if len(filtered) >= top_k_results:
                    break
            top_k_matches = filtered

            if top_k_matches and top_k_matches[0]['db_idx'] != -1:
                if attempt == 0: break

        # 4. Final concatenation & Loop through Top K results
        if not top_k_matches:
            raise RuntimeError("FAILURE: No candidates passed filters. Increase rms_tolerance_pct, lower chroma_floor, or lower coherence_min_threshold.")

        print(f"\nRetrieval Complete. Found Top {len(top_k_matches)} extension segments")

        # UI toggle (default True for CLI behavior)
        do_arrange = bool(cfg_local.get("rag", {}).get("do_arrange", False))

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect results for UI
        results_manifest = []

        y_user_input_clean = y_user_input

        for rank, match in enumerate(top_k_matches, start=1):
            # best_idx = match["db_idx"]
            # start_s_db = best_idx * int(base_chunk_duration * sample_rate)
            best_idx = match["db_idx"]
            start_s_db = match["start_s_db"]   # ✅ use exact start used in scoring

            # Retrieve the specific audio segment for this index
            # retrieved_audio = db_audio[start_s_db : start_s_db + search_len_samples]
            retrieved_audio = db_audio[start_s_db : start_s_db + extend_len_samples]

            # Stitch input + retrieval
            input_audio_copy = y_user_input_clean.copy()
            retrieved_audio_copy = retrieved_audio.copy()

            L = crossfade_audio(input_audio_copy, retrieved_audio_copy, crossfade_time_samples)

            if L == 0:
                y_final_audio = np.concatenate([input_audio_copy, retrieved_audio_copy])
            else:
                y_final_audio = np.concatenate(
                    [
                        input_audio_copy[:-L],
                        input_audio_copy[-L:] + retrieved_audio_copy[:L],
                        retrieved_audio_copy[L:],
                    ]
                )

            final_audio_len_s = len(y_final_audio) / sample_rate
            len_tag = f"{final_audio_len_s:.2f}s"

            # ✅ ALWAYS write a stitched preview wav for UI (incase we want base + extension)
            # stitched_name = (
            #     f"Rank{rank}_IDX_{best_idx}_LEN_{len_tag}_Score_{match['score']:.4f}_stitched.wav"
            # )
            # stitched_path = output_dir / stitched_name

            # import soundfile as sf
            # sf.write(str(stitched_path), y_final_audio, sample_rate)

            # ✅ NEW: write EXTENSION ONLY for all Top-K
            # Note: retrieved_audio_copy already has fade-in applied in its first L samples
            # because crossfade_audio() modifies it in-place.
            stitched_name = None   # ✅ ALWAYS defined now
            arranged_name = None

            extension_only = retrieved_audio_copy

            extension_name = (
                f"Rank{rank}_IDX_{best_idx}_LEN_{len_tag}_Score_{match['score']:.4f}_extension.wav"
            )
            extension_path = output_dir / extension_name
            sf.write(str(extension_path), extension_only, sample_rate)

            item = {
                "rank": rank,
                "db_idx": int(best_idx),
                "db_id_4s": int(match["db_id_4s"]),      # ✅ original 4s DB chunk id
                "start_s_db": int(match["start_s_db"]),  # ✅ debug / verification
                "score": float(match["score"]),
                "coherence": float(match.get("coherence", 0.0)),
                "relevance": float(match.get("relevance", 0.0)),
                "lookback": float(match.get("lookback", 0.0)),
                "extension_wav": extension_name,   # ✅ new
                "stitched_wav": stitched_name,              # ✅ not produced by default
                "arranged_wav": None,
                "arranged_png": None,
            }

            # ✅ Only do heavy perform_music if requested
            if do_arrange:
                from src.models import perform_music as neural_band  # lazy import (torch only when arranging)
                # Feature extraction for perform_music temp leader
                final_pitch_arr, _, _ = librosa.pyin(
                    y_final_audio,
                    fmin=pyin_fmin,
                    fmax=pyin_fmax,
                    sr=sample_rate,
                    hop_length=pyin_hop,
                )
                final_pitch_arr = np.nan_to_num(final_pitch_arr)

                final_loudness_arr = librosa.feature.rms(
                    y=y_final_audio,
                    frame_length=rms_frame,
                    hop_length=rms_hop,
                    center=True,
                )[0]
                final_loudness_arr = np.nan_to_num(final_loudness_arr)

                N_frames = min(len(final_pitch_arr), len(final_loudness_arr))
                N_chunks = 1
                final_len_samples = len(y_final_audio)

                # Save temp features ONLY when arranging
                temp_name_rank = f"{temp_leader_name}_{rank}"
                np.save(
                    data_dir / f"pitch_{temp_name_rank}.npy",
                    final_pitch_arr[:N_frames].reshape(N_chunks, N_frames),
                )
                np.save(
                    data_dir / f"loudness_{temp_name_rank}.npy",
                    final_loudness_arr[:N_frames].reshape(N_chunks, N_frames),
                )
                np.save(
                    data_dir / f"audio_{temp_name_rank}.npy",
                    y_final_audio.reshape(N_chunks, final_len_samples),
                )

                neural_band.LEADER = temp_name_rank
                neural_band.GENERATE_SECONDS = final_audio_len_s
                neural_band.MANUAL_START_INDEX = 0

                print(f"\n   -> RANK {rank}: Index {best_idx} (Score: {match['score']:.4f})")
                print(f"      Running arrangement on {neural_band.LEADER}...")
                neural_band.run_complete_arrangement()  # ✅ ONCE

                # Rename the produced wav/png to Rank...wav/png
                default_out_name = f"Smart_Arrangement_{temp_name_rank}_Unison_0.wav"
                default_out_path = output_dir / default_out_name

                arranged_name = f"Rank{rank}_IDX_{best_idx}_LEN_{len_tag}_Score_{match['score']:.4f}.wav"
                arranged_path = output_dir / arranged_name

                if default_out_path.exists():
                    shutil.move(default_out_path, arranged_path)
                    item["arranged_wav"] = arranged_path.name

                    default_report_path = default_out_path.with_suffix(".png")
                    if default_report_path.exists():
                        arranged_png_path = arranged_path.with_suffix(".png")
                        shutil.move(default_report_path, arranged_png_path)
                        item["arranged_png"] = arranged_png_path.name

                cleanup_temp_files(temp_name_rank, data_dir=data_dir)

            results_manifest.append(item)

        print("\nAudio RAG Pipeline Execution Complete")

        return {
            "ok": True,
            "input": str(input_path),
            "target_duration": float(target_duration),
            "top_k": len(results_manifest),
            "output_dir": str(output_dir),
            "results": results_manifest,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        pass


if __name__ == "__main__":
    run_audio_rag()
