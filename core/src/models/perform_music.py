import torch
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
import os
import math
import random
import sys
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import scipy.signal

from src.models.train_instrument import AudioSynthTrainer
from src.models.decoder_conductor import NeuralArranger
from src.models.signal_processing import harmonic_synthesis, noise_synthesis
#from src.visualization.song_in_detail import save_report_card

from configs.load_config import load_yaml, resolve_project_root
import json

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # models -> src -> core -> REPO ROOT

cfg_file = REPO_ROOT / "configs" / "music_generator.yaml"
cfg = load_yaml(cfg_file)
project_root = (REPO_ROOT / cfg["paths"].get("project_root", ".")).resolve()

seed = int(cfg["generation"]["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device_cfg = cfg["generation"].get("device", "auto")
if device_cfg == "cpu":
    device = torch.device("cpu")
elif device_cfg == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

io_cfg = cfg.get("io", {})
SAVE_REFERENCE = bool(io_cfg.get("save_reference", True))
SAVE_STEMS = bool(io_cfg.get("save_stems", True))
SAVE_REPORT = bool(io_cfg.get("save_report_card", False))

# Mention the arranger checkpoint file location

ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
ARRANGER_CKPT = ckpt_dir / cfg["checkpoints"]["arranger"]
MODEL_REGISTRY = {k: str(ckpt_dir / v) for k, v in cfg["checkpoints"]["models"].items()}

# Configuration for the instruments

LEADER = cfg["instruments"]["leader"]

ARRANGER_FOLLOWERS = cfg["instruments"]["arranger_followers"]

# To separate Bass & Drums, since Drums possess no Pitch & require further processing

ARRANGER_MAP = cfg["instruments"]["arranger_map"]
NUM_ARRANGER_CHANNELS = int(cfg["instruments"]["num_arranger_channels"])

# DRUM CONFIG
ADD_DRUMS = cfg["instruments"]["add_drums"]
DRUM_VOLUME = float(cfg["instruments"]["drum_volume"])

# Main parameters

MANUAL_START_INDEX = cfg["generation"]["manual_start_index"]
GENERATE_SECONDS = float(cfg["generation"]["seconds"])
CONST_WINDOW = float(cfg["generation"]["const_window_sec"])
NOISE_SCALE_CFG = cfg["generation"]["noise_scale"]  # dict
DEFAULT_NOISE_SCALE = float(NOISE_SCALE_CFG.get("default", 0.05))
DS = int(cfg["generation"].get("rhythmic_match_ds", 8))

# Audio Parameters

SR = int(cfg["audio"]["sample_rate"])
FRAME_RATE = int(cfg["audio"]["frame_rate"])
HOP_SAMPLES = int(cfg["audio"]["hop_samples"])
CHROMA_HOP = int(cfg["audio"]["chroma_hop_length"])
CHROMA_SR = int(cfg["audio"]["chroma_sr"])


# ----------------------------------------------------------------

# Function definition

def load_audio_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing model: {ckpt_path}")
    model = AudioSynthTrainer.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def load_arranger_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing arranger: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = NeuralArranger(num_followers=NUM_ARRANGER_CHANNELS)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def rms_envelope(x, win=2048, hop=512):
    x2 = x.astype(np.float32, copy=False) ** 2
    w = np.ones(win, dtype=np.float32) / win
    env = np.sqrt(np.convolve(x2, w, mode="same"))
    return env

def rms(x, eps=1e-8):
    x = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(x * x) + eps))

def audio_to_chroma(audio_waveform):
    chromagram = librosa.feature.chroma_cqt(y=audio_waveform, sr=CHROMA_SR, hop_length=CHROMA_HOP)
    chroma_vector = np.mean(chromagram, axis=1)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector

def pitch_to_chroma(f0_curve):
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12)
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector

def find_best_match_smart(leader_audio, candidate_pitches, candidate_loudness):
    leader_chroma = audio_to_chroma(leader_audio).reshape(1, -1)
    best_idx, best_score, best_shift = 0, -1.0, 0
    total = len(candidate_pitches)
    indices = random.sample(range(total), min(2000, total))
    for idx in indices:
        if np.mean(candidate_loudness[idx]) < 0.1: continue
        cand_chroma = pitch_to_chroma(candidate_pitches[idx])
        for shift in range(-6, 7):
            shifted = np.roll(cand_chroma, shift).reshape(1, -1)
            score = cosine_similarity(leader_chroma, shifted)[0][0]
            if score > best_score:
                best_score, best_idx, best_shift = score, idx, shift
    return best_idx, best_shift

def find_rhythmic_match_downsampled(leader_loudness, drum_loudness_flat, ds=8):
    # Downsample envelopes: 250 Hz -> ~31 Hz when ds=8
    lead = leader_loudness.flatten().astype(np.float32, copy=False)[::ds]
    drum = drum_loudness_flat.flatten().astype(np.float32, copy=False)[::ds]

    lead_diff = np.diff(lead, prepend=lead[0])
    lead_env = lead_diff - lead_diff.mean()

    drum_diff = np.diff(drum, prepend=drum[0])
    drum_env = drum_diff  # keep your original behavior (you didn't always center drums)

    corr = scipy.signal.fftconvolve(drum_env, lead_env[::-1], mode="valid")
    best = int(np.argmax(corr))

    return best * ds  # convert back to original frame index (250 Hz grid)

def find_rhythmic_match(leader_loudness, drum_loudness_flat, device):
    # 1. Feature Engineering: Prioritize Rhythmic Onsets
    
    # Calculate the differentiated loudness curve
    # The derivative highlights where the energy *changes* rapidly
    lead_env_raw = leader_loudness.flatten()

    # Use padding to keep the size consistent, then calculate the difference
    # This acts like a high-pass filter on the energy envelope
    lead_env_diff = np.diff(lead_env_raw, prepend=lead_env_raw[0])

    # Remove DC offset and center the differential curve
    lead_env = lead_env_diff - np.mean(lead_env_diff)

    # We must also apply this to the drum input for a fair correlation
    drum_env_raw = drum_loudness_flat.flatten()
    drum_env_diff = np.diff(drum_env_raw, prepend=drum_env_raw[0])
    
    # Center the differential drum curve as well, we need to check this
    #drum_input_data = drum_env_diff - np.mean(drum_env_diff)
    
    drum_input_data = drum_env_diff

    # Convolution
    kernel = torch.from_numpy(lead_env).float().view(1, 1, -1).to(device)
    drum_input = torch.from_numpy(drum_input_data).float().view(1, 1, -1).to(device)

    with torch.no_grad():
        # F.conv1d now correlates the *rhythmic changes*
        correlation_map = F.conv1d(drum_input, kernel)

    best_idx = correlation_map.argmax().item()
    return best_idx

def boost_dynamics(curve, threshold=0.4, ratio=4.0):
    centered = curve - threshold
    expanded = centered * ratio
    result = expanded + threshold
    return torch.clamp(result, 0.0, 1.0)

def simplify_pitch(pitch_curve, factor=8):
    p_small = pitch_curve[:, ::factor, :]
    p_blocky = F.interpolate(p_small.permute(0, 2, 1), size=pitch_curve.shape[1], mode='nearest')
    return p_blocky.permute(0, 2, 1)

def force_octave(pitch_curve, target_hz=100.0):
    avg_pitch = torch.mean(pitch_curve[pitch_curve > 40])
    if torch.isnan(avg_pitch) or avg_pitch == 0: return pitch_curve
    num_octaves = torch.round(torch.log2(target_hz / avg_pitch))
    return pitch_curve * (2.0 ** num_octaves)

def run_complete_arrangement(root: Path | None = None, dev: torch.device | None = None):
    root = root or project_root
    dev = dev or device
    
    print(f"Starting Complete AI Arranger (Leader: {LEADER})...", flush=True)
    
    # 1. LOAD DATA
    candidates = [root / p for p in cfg["paths"]["features_candidates"]]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find features folder. Looked in:\n" + "\n".join(map(str, candidates)))

    datasets = {}
    load_list = [LEADER] + list(MODEL_REGISTRY.keys()) + (['Drums'] if ADD_DRUMS else [])

    req = cfg["validation"]["require_files"]

    for inst in load_list:
        templates = req["leader"] if inst == LEADER else req[inst]  # Bass/Drums keys match YAML
        datasets[inst] = {}

        for tmpl in templates:
            fname = tmpl.format(leader=LEADER)   # only affects leader templates
            fpath = data_path / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing {inst} feature file: {fpath}")

            key = Path(fname).stem.split("_", 1)[0]   # "audio_Guitar" -> "audio"
            datasets[inst][key] = np.load(fpath, mmap_mode="r", allow_pickle=False)

    # 2. PREPARE LEADER
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)
    leader_len = len(datasets[LEADER]['pitch'])

    if MANUAL_START_INDEX is not None:
        start_idx = MANUAL_START_INDEX
    else:
        start_idx = random.randint(0, leader_len - num_chunks - 1)

    out_base = root / cfg["paths"]["outputs_dir"] / cfg["io"]["run_subdir"]
    out_base.mkdir(parents=True, exist_ok=True)

    run_id = f"{LEADER}_idx{start_idx}_sec{int(GENERATE_SECONDS)}_seed{seed}"
    output_dir = out_base / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "leader": LEADER,
        "start_idx": start_idx,
        "seconds": GENERATE_SECONDS,
        "seed": seed,
        "device": str(dev),
        "data_path": str(data_path),
        "models": MODEL_REGISTRY,
        "arranger_ckpt": str(ARRANGER_CKPT),
        }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Normalizing Leader Audio...")
    a_lead = datasets[LEADER]["audio"][start_idx:start_idx + num_chunks].reshape(-1).astype(np.float32, copy=False)
    max_val = float(np.max(np.abs(a_lead)))
    if max_val > 0:
        a_lead = a_lead / max_val * 0.9

    LEADER_GAIN = float(cfg.get("mix", {}).get("leader_gain", 0.65))  # default 0.65
    a_lead *= LEADER_GAIN

    p_lead_np = datasets[LEADER]["pitch"][start_idx:start_idx + num_chunks].reshape(-1).astype(np.float32, copy=False)
    l_lead_np = datasets[LEADER]["loudness"][start_idx:start_idx + num_chunks].reshape(-1).astype(np.float32, copy=False)

    final_mix_len = len(a_lead)
    final_mix = np.zeros(final_mix_len, dtype=np.float32)
    final_mix += a_lead

    l_leader_t = torch.from_numpy(l_lead_np.copy()).unsqueeze(0).unsqueeze(-1).to(dev)

    # 3. ARRANGER
    print("Generating Dynamics...", flush=True)
    arranger = load_arranger_model(ARRANGER_CKPT).to(dev)
    with torch.no_grad():
        predicted_dynamics = arranger(l_leader_t)
        
    # Output the reference audio file        
    if SAVE_REFERENCE:
        sf.write(output_dir / f"Reference_{LEADER}.wav", a_lead, SR)

    # 4. DRUMS
    if ADD_DRUMS:
        print("Recording Drums...", flush=True)
        # drum_l_flat = datasets['Drums']['loudness'].reshape(-1)  #Pitch is not available for Drums
        # drum_a_flat = datasets['Drums']['audio'].reshape(-1)

        # 1. CNN Match
        # idx = find_rhythmic_match(l_lead_np, drum_l_flat)

        drum_l_flat = datasets["Drums"]["loudness"].reshape(-1)
        drum_a_flat = datasets["Drums"]["audio"].reshape(-1)

        # Match using downsampled FFT correlation (no giant conv tensor)
        idx = find_rhythmic_match_downsampled(l_lead_np, drum_l_flat, ds=DS)

        # 2. SLICE
        start_sample = idx * HOP_SAMPLES
        req_len = len(final_mix)

        if start_sample + req_len > len(drum_a_flat):
            part1 = drum_a_flat[start_sample:]
            remaining = req_len - len(part1)
            if remaining > len(drum_a_flat):
                part2 = np.tile(drum_a_flat, math.ceil(remaining / len(drum_a_flat)))[:remaining]
            else:
                part2 = drum_a_flat[:remaining]
            drum_track = np.concatenate([part1, part2])
        else:
            drum_track = drum_a_flat[start_sample: start_sample + req_len]

        # 3. APPLY ARRANGER
        if 'Drums' in ARRANGER_MAP:
            idx_map = ARRANGER_MAP['Drums']
            vol_curve = predicted_dynamics[0, :, idx_map].cpu().numpy()
            vol_audio = np.repeat(vol_curve, HOP_SAMPLES)

            if len(vol_audio) > len(drum_track):
                vol_audio = vol_audio[:len(drum_track)]
            elif len(vol_audio) < len(drum_track):
                vol_audio = np.pad(vol_audio, (0, len(drum_track) - len(vol_audio)), mode='edge')

            # Multiply and not mask
            vol_audio = np.clip(vol_audio * 1.5, 0.0, 1.0)

            # Force min volume if needed
            if vol_audio.mean() < 0.05: vol_audio = np.maximum(vol_audio, 0.2)

            drum_track = drum_track * vol_audio * DRUM_VOLUME

        final_mix += drum_track
        if SAVE_STEMS:
            sf.write(output_dir / "Drums_Stem.wav", drum_track, SR)

        # after saving drum stem
        del drum_l_flat, drum_a_flat

    # Other INSTRUMENTS
    frames_per_chunk = int(CONST_WINDOW * FRAME_RATE)
    samples_per_chunk = int(CONST_WINDOW * SR)

    # HELPER: FREQUENCY MAP
    # Maps Chroma Index (0=C, 1=C#, etc.) to Low Bass Frequencies
    # We aim for the octave between 40Hz and 80Hz
    BASS_FREQS = [
        32.70, 34.65, 36.71, 38.89, 41.20, 43.65,  # C  to F
        46.25, 49.00, 51.91, 55.00, 58.27, 61.74  # F# to B
    ]

    for inst_name in MODEL_REGISTRY.keys():
        print(f"Recording {inst_name}...", flush=True)

        model = load_audio_model(MODEL_REGISTRY[inst_name]).to(dev)
        inst_audio_list = []

        # Dynamics Setup
        if inst_name in ARRANGER_MAP:
            idx = ARRANGER_MAP[inst_name]
            l_generated = predicted_dynamics[:, :, idx].unsqueeze(-1)
        else:
            l_generated = l_leader_t * 0.8
        l_generated = boost_dynamics(l_generated, threshold=0.4, ratio=4.0)

        for c in range(num_chunks):
            # SLICING
            start_s = c * samples_per_chunk
            end_s = start_s + samples_per_chunk
            if start_s >= len(a_lead): break
            leader_audio_chunk = a_lead[start_s:end_s]

            start_f = c * frames_per_chunk
            end_f = start_f + frames_per_chunk
            l_chunk = l_generated[:, start_f:end_f, :]  # Target Volume

            
            if inst_name == 'Bass':
                # Shadow the Guitar, as found while listening, this is better
                p_raw = p_lead_np[start_f:end_f]

                # Smoothening
                p_smooth = scipy.signal.medfilt(p_raw, kernel_size=15)

                # Force Sub-Bass
                p_t = torch.from_numpy(p_smooth).float().reshape(1, -1, 1).to(dev)
                p_t = force_octave(p_t, target_hz=55.0)

                # We start with the Arranger's curve
                l_raw = l_chunk

                # Lock to Guitar Rhythm, for Bass this is better
                l_guitar_ref = l_lead_np[start_f:end_f]
                # Fixed Line 364
                l_guitar_t = torch.from_numpy(l_guitar_ref.copy()).float().reshape(1, -1, 1).to(dev)
                gate = (l_guitar_t > 0.05).float()
                l_raw = l_raw * gate

                # Normalize the chunk so the peak is exactly 1.0
                max_vol = torch.max(l_raw)
                if max_vol > 0:
                    l_raw = l_raw / max_vol

                # 0.6 is a safe "Shadow" level
                # Since we normalized to 1.0, we KNOW the output will peak at exactly 0.6
                l_chunk = l_raw * 0.6

                noise_scale = float(NOISE_SCALE_CFG.get(inst_name, DEFAULT_NOISE_SCALE))

            else:
                
                #Here we are matching the tune of other instruments with leader guitar using cosine similarity
                pitch_for_match = datasets[inst_name]["pitch"]
                loud_for_match  = datasets[inst_name]["loudness"]

                # Ensure 2D shape for the matcher: (num_chunks, frames_per_chunk)
                if pitch_for_match.ndim == 1:
                    pitch_for_match = pitch_for_match.reshape(-1, frames_per_chunk)
                if loud_for_match.ndim == 1:
                    loud_for_match = loud_for_match.reshape(-1, frames_per_chunk)

                best_idx, shift = find_best_match_smart(leader_audio_chunk, pitch_for_match, loud_for_match)

                pitch_arr = datasets[inst_name]["pitch"]
                req_frames = end_f - start_f

                if pitch_arr.ndim == 2:
                    # pitch stored as (num_chunks, frames_per_chunk) -> grab one row
                    p_chunk = pitch_arr[best_idx]

                    # Make sure length matches what the model expects
                    if p_chunk.shape[0] < req_frames:
                        # wrap if shorter than needed
                        need = req_frames - p_chunk.shape[0]
                        p_chunk = np.concatenate([p_chunk, pitch_arr[0][:need]])
                    else:
                        p_chunk = p_chunk[:req_frames]
                else:
                    # fallback: pitch stored as 1D stream
                    p_flat = pitch_arr.reshape(-1)
                    # if you know the real frames-per-chunk, use it instead of 1000
                    chunk_len_db = frames_per_chunk  # derived from CONST_WINDOW * 250
                    current_frame_idx = best_idx * chunk_len_db

                    if current_frame_idx + req_frames > len(p_flat):
                        tail = p_flat[current_frame_idx:]
                        need = req_frames - len(tail)
                        p_chunk = np.concatenate([tail, p_flat[:need]])
                    else:
                        p_chunk = p_flat[current_frame_idx: current_frame_idx + req_frames]

                # Apply chroma shift
                p_chunk = np.asarray(p_chunk, dtype=np.float32) * (2 ** (shift / 12.0))
                p_t = torch.from_numpy(p_chunk).reshape(1, -1, 1).to(dev)
                
                noise_scale = float(NOISE_SCALE_CFG.get(inst_name, DEFAULT_NOISE_SCALE))

            # SYNTHESIS
            with torch.no_grad():
                controls = model.model(p_t, l_chunk)
                controls['noise'] *= noise_scale
                target_samples = p_t.shape[1] * HOP_SAMPLES
                harm = harmonic_synthesis(p_t, controls['amplitudes'], controls['harmonics'],
                                          n_samples=target_samples)
                nz = noise_synthesis(controls['noise'], n_samples=target_samples)
                chunk_out = (harm + nz).squeeze().cpu().numpy()
                inst_audio_list.append(chunk_out)

        track_audio = np.concatenate(inst_audio_list)
        min_len = min(len(final_mix), len(track_audio))
        final_mix[:min_len] += track_audio[:min_len]
        track_audio = np.clip(track_audio, -1.0, 1.0).astype(np.float32, copy=False)

        if SAVE_STEMS:
            sf.write(output_dir / f"{inst_name}_Stem.wav", track_audio, SR)

    # 6. SAVE
    # Simple ducking: reduce leader when accompaniment is strong
    leader = a_lead.copy()
    acc = (final_mix - leader)  # everything except leader

    # --- A) Leader target RMS normalization ---
    leader_target_rms = float(cfg.get("mix", {}).get("leader_target_rms", 0.08))
    cur_rms = rms(leader)
    if cur_rms > 1e-8:
        g = leader_target_rms / cur_rms
        g = float(np.clip(g, 0.1, 10.0))
        leader *= g

    # recompute accompaniment relative to adjusted leader
    acc = final_mix - leader

    # --- B) Ducking (already in your code) ---
    env_acc = rms_envelope(acc, win=2048)
    env_lead = rms_envelope(leader, win=2048)

    DUCK = float(cfg.get("mix", {}).get("duck_strength", 0.5))
    ratio = env_acc / (env_lead + 1e-6)
    duck_gain = 1.0 / (1.0 + DUCK * ratio)

    leader_ducked = leader * duck_gain

    # --- C) Enforce leader-to-accompaniment loudness ratio ---
    leader_to_acc_ratio = float(cfg.get("mix", {}).get("leader_to_acc_ratio", 0.9))
    rms_lead = rms(leader_ducked)
    rms_acc = rms(acc)

    if rms_acc > 1e-8 and rms_lead > 1e-8:
        desired_rms_lead = leader_to_acc_ratio * rms_acc
        post = desired_rms_lead / rms_lead
        post = float(np.clip(post, 0.1, 10.0))
        leader_ducked *= post

    final_mix = leader_ducked + acc
    
    final_path = output_dir / "Complete_Arrangement.wav"
    final_mix = np.clip(final_mix, -1.0, 1.0).astype(np.float32, copy=False)
    sf.write(final_path, final_mix, SR)
    print(f"\nDONE! Output: {final_path}")
    if SAVE_REPORT:
        try:
            from src.visualization.song_in_detail import save_report_card
            save_report_card(str(final_path))
        except Exception as e:
            print(f"[warn] report card skipped: {e}", flush=True)
            
    return str(final_path)


if __name__ == "__main__":
    run_complete_arrangement(root=REPO_ROOT, dev=device)


