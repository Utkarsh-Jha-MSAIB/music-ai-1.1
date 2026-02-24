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

try:
    from src.train import AudioSynthTrainer
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis
    from src.visualization.song_in_detail import save_report_card
except ImportError:
    from src.models.train_instrument import AudioSynthTrainer
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis



# Model Checkpoints
MODEL_REGISTRY= {
    'Bass': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_0/checkpoints/epoch=19-step=1420.ckpt",
    'Strings': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_1/checkpoints/epoch=19-step=160.ckpt",
    'Piano': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_2/checkpoints/epoch=19-step=1520.ckpt",
}

# Band leader
LEADER_INSTRUMENT = 'Guitar'
ACCOMPANIMENT = ['Bass', 'Piano', 'Strings']

# Parameters
MANUAL_START_INDEX = None
GENERATE_SECONDS = 20
NOISE_SCALE = 0.05
CONST_WINDOW = 4.0


def load_model(instrument_name):
    if instrument_name not in MODEL_REGISTRY:
        print(f"Error: '{instrument_name}' not in registry.")
        sys.exit(1)
    ckpt_path = MODEL_REGISTRY[instrument_name]
    if not os.path.exists(ckpt_path):
        print(f"Error: Model not found: {ckpt_path}")
        sys.exit(1)
    model = AudioSynthTrainer.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def pitch_to_chroma(f0_curve):
    """Converts Pitch (Hz) to Key Signature (Chroma Vector)"""
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12)
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector


def find_best_match_smart(leader_pitch, candidate_pitches, candidate_loudness):
    """Finds the best match by virtually shifting candidates"""
    leader_chroma = pitch_to_chroma(leader_pitch).reshape(1, -1)
    best_idx = 0
    best_score = -1.0
    best_shift = 0

    total_candidates = len(candidate_pitches)
    sample_size = min(200, total_candidates)
    search_indices = random.sample(range(total_candidates), sample_size)

    for idx in search_indices:
        if np.mean(candidate_loudness[idx]) < 0.1: continue
        cand_chroma = pitch_to_chroma(candidate_pitches[idx])

        for shift in range(-2, 3):
            shifted_chroma = np.roll(cand_chroma, shift).reshape(1, -1)
            score = cosine_similarity(leader_chroma, shifted_chroma)[0][0]
            if score > best_score:
                best_score = score
                best_idx = idx
                best_shift = shift

    print(f"Match found! Score: {best_score:.2f} | Shift: {best_shift:+d} semitones")
    return best_idx, best_shift


def boost_dynamics(curve, threshold=0.5, ratio=2.0):
    centered = curve - threshold
    expanded = centered * ratio
    result = expanded + threshold
    return torch.clamp(result, 0.0, 1.0)


def run_smart_dj():
    print(f"Starting Neural DJ (Leader: {LEADER_INSTRUMENT})...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD ALL DATA
    project_root = Path(__file__).resolve().parents[1]
    if project_root.name == 'src': project_root = project_root.parent
    data_path = project_root / 'data' / 'processed' / 'features'

    datasets = {}
    for inst in [LEADER_INSTRUMENT] + ACCOMPANIMENT:
        print(f"Loading {inst} Database...", flush=True)
        try:
            datasets[inst] = {
                'pitch': np.load(data_path / f'pitch_{inst}.npy'),
                'loudness': np.load(data_path / f'loudness_{inst}.npy'),
                'audio': np.load(data_path / f'audio_{inst}.npy')
            }
        except FileNotFoundError:
            print(f"❌ Error: Missing data for {inst}")
            return

    # 2. PICK LEADER TRACK
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)
    leader_len = len(datasets[LEADER_INSTRUMENT]['pitch'])

    if MANUAL_START_INDEX is not None:
        start_idx = MANUAL_START_INDEX
        print(f"   🔒 Using Manual Start Index: {start_idx}")
    else:
        start_idx = random.randint(0, leader_len - num_chunks - 1)
        print(f"   🎲 Random Start Index: {start_idx}")

    p_lead = datasets[LEADER_INSTRUMENT]['pitch'][start_idx: start_idx + num_chunks].reshape(-1)
    a_lead = datasets[LEADER_INSTRUMENT]['audio'][start_idx: start_idx + num_chunks].reshape(-1)

    final_mix_len = len(p_lead) * 64
    final_mix = np.zeros(final_mix_len)

    # Add Leader Audio (Guitar) to the mix
    final_mix += a_lead

    # 3. FIND & GENERATE ACCOMPANIMENT
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    for inst_name in ACCOMPANIMENT:
        print(f"   🔎 Searching for a matching {inst_name}...", flush=True)

        best_idx, shift_amount = find_best_match_smart(
            p_lead,
            datasets[inst_name]['pitch'],
            datasets[inst_name]['loudness']
        )

        p_match = datasets[inst_name]['pitch'][best_idx: best_idx + num_chunks].reshape(-1)
        l_match = datasets[inst_name]['loudness'][best_idx: best_idx + num_chunks].reshape(-1)

        p_match = p_match * (2 ** (shift_amount / 12.0))

        # --- FIX: Pass Instrument Name (Key), NOT Path (Value) ---
        model = load_model(inst_name).to(device)

        p_t = torch.from_numpy(p_match).unsqueeze(0).unsqueeze(-1).to(device)
        l_t = torch.from_numpy(l_match).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            l_t = boost_dynamics(l_t, threshold=0.4, ratio=2.0)
            controls = model.model(p_t, l_t)
            controls['noise'] *= NOISE_SCALE
            target_samples = p_t.shape[1] * 64

            harm = harmonic_synthesis(p_t, controls['amplitudes'], controls['harmonics'], n_samples=target_samples)
            nz = noise_synthesis(controls['noise'], n_samples=target_samples)

            track_audio = (harm + nz).squeeze().cpu().numpy()

            min_len = min(len(final_mix), len(track_audio))
            final_mix[:min_len] += track_audio[:min_len]

            sf.write(output_dir / f"dj_stem_{inst_name}.wav", track_audio, 16000)

    # 4. SAVE MIX
    out_name = f"dj_mix_{LEADER_INSTRUMENT}_idx{start_idx}.wav"
    sf.write(output_dir / out_name, final_mix, 16000)
    sf.write(output_dir / f"dj_leader_{LEADER_INSTRUMENT}.wav", a_lead, 16000)

    print(f"\n🎉 DJ MIX COMPLETE!")
    print(f"   🎧 Mix: {output_dir / out_name}")
    try:
        save_report_card(str(output_dir / out_name))
    except:
        pass


if __name__ == "__main__":
    run_smart_dj()