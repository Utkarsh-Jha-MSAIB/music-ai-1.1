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

from src.models.train_instrument import AudioSynthTrainer
from src.models.decoder_conductor import NeuralArranger
from src.models.signal_processing import harmonic_synthesis, noise_synthesis
from src.visualization.song_in_detail import save_report_card

# ==============================================================================
# 🎼 THE COMPLETE AI ARRANGER (HYBRID)
# ==============================================================================

# 1. CHECKPOINTS
ARRANGER_CKPT = "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_9/checkpoints/Arranger-Guitar-epoch=31-train_loss=0.1043.ckpt"

MODEL_REGISTRY = {
    'Piano': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_2/checkpoints/epoch=19-step=1520.ckpt",
    'Bass': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_0/checkpoints/epoch=19-step=1420.ckpt",
    'Organ': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_6/checkpoints/epoch=19-step=400.ckpt",
}

# 2. BAND CONFIG
LEADER = 'Guitar'
ACCOMPANIMENT = ['Piano', 'Bass', 'Organ']  # Order matters! (Must match Arranger training order)

# 3. SETTINGS
MANUAL_START_INDEX = None
GENERATE_SECONDS = 20
NOISE_SCALE = 0.05
CONST_WINDOW = 4.0


# ==============================================================================

def load_audio_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Model not found: {ckpt_path}")
        sys.exit(1)
    model = AudioSynthTrainer.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def load_arranger_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Arranger not found: {ckpt_path}")
        sys.exit(1)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # Note: Arranger was trained on specific followers. Ensure len matches.
    model = NeuralArranger(num_followers=len(ACCOMPANIMENT))

    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pitch_to_chroma(f0_curve):
    active_f0 = f0_curve[f0_curve > 40]
    if len(active_f0) == 0: return np.zeros(12)
    midi_notes = 69 + 12 * np.log2(active_f0 / 440.0)
    chroma_vals = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_vector = np.bincount(chroma_vals, minlength=12)
    norm = np.linalg.norm(chroma_vector)
    return chroma_vector / norm if norm > 0 else chroma_vector


def find_best_match_smart(leader_pitch, candidate_pitches, candidate_loudness):
    """Cosine Similarity Search for Pitch."""
    leader_chroma = pitch_to_chroma(leader_pitch).reshape(1, -1)
    best_idx, best_score, best_shift = 0, -1.0, 0

    total_candidates = len(candidate_pitches)
    sample_size = min(200, total_candidates)
    search_indices = random.sample(range(total_candidates), sample_size)

    for idx in search_indices:
        if np.mean(candidate_loudness[idx]) < 0.1: continue
        cand_chroma = pitch_to_chroma(candidate_pitches[idx])

        for shift in range(-2, 3):
            shifted = np.roll(cand_chroma, shift).reshape(1, -1)
            score = cosine_similarity(leader_chroma, shifted)[0][0]
            if score > best_score:
                best_score, best_idx, best_shift = score, idx, shift

    print(f"   🤝 Found Key Match! Score: {best_score:.2f} | Shift: {best_shift:+d}")
    return best_idx, best_shift


def boost_dynamics(curve, threshold=0.4, ratio=2.5):
    centered = curve - threshold
    expanded = centered * ratio
    result = expanded + threshold
    return torch.clamp(result, 0.0, 1.0)


def apply_spotlight(dynamics, time_idx, inst_idx):
    """Forces instruments to take turns."""
    mask = torch.ones_like(dynamics)
    total_time = dynamics.shape[1]
    current_progress = time_idx / total_time

    # Intro: Bass Only (Mute Piano)
    if current_progress < 0.33:
        if inst_idx == 1:  # Piano
            mask *= 0.0
            # Outro: Piano Only (Mute Bass)
    elif current_progress > 0.66:
        if inst_idx == 0:  # Bass
            mask *= 0.0

    return dynamics * mask

def run_complete_arrangement():
    print(f"🎹 Starting Complete AI Arranger (Leader: {LEADER})...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. LOAD DATA ---
    project_root = Path(__file__).resolve().parents[1]
    if project_root.name == 'src': project_root = project_root.parent
    data_path = project_root / 'data' / 'processed' / 'features'

    datasets = {}
    for inst in [LEADER] + ACCOMPANIMENT:
        print(f"💿 Loading {inst}...", flush=True)
        try:
            datasets[inst] = {
                'pitch': np.load(data_path / f'pitch_{inst}.npy'),
                'loudness': np.load(data_path / f'loudness_{inst}.npy'),
                'audio': np.load(data_path / f'audio_{inst}.npy')
            }
        except FileNotFoundError:
            print(f"❌ Error: Missing data for {inst}")
            return

    # --- 2. PREPARE LEADER ---
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)
    leader_len = len(datasets[LEADER]['pitch'])

    if MANUAL_START_INDEX is not None:
        start_idx = MANUAL_START_INDEX
        print(f"   🔒 Manual Start: {start_idx}")
    else:
        start_idx = random.randint(0, leader_len - num_chunks - 1)
        print(f"   🎲 Random Start: {start_idx}")

    p_lead_np = datasets[LEADER]['pitch'][start_idx: start_idx + num_chunks].reshape(-1)
    l_lead_np = datasets[LEADER]['loudness'][start_idx: start_idx + num_chunks].reshape(-1)
    a_lead = datasets[LEADER]['audio'][start_idx: start_idx + num_chunks].reshape(-1)

    # Tensors for Arranger
    l_leader_t = torch.from_numpy(l_lead_np).unsqueeze(0).unsqueeze(-1).to(device)

    # Initialize Final Mix with Guitar Audio
    final_mix_len = len(p_lead_np) * 64
    final_mix = np.zeros(final_mix_len)
    final_mix += a_lead

    # --- 3. RUN ARRANGER (THE BRAIN) ---
    print("🧠 Generating Dynamics (Transformer)...", flush=True)
    arranger = load_arranger_model(ARRANGER_CKPT).to(device)

    with torch.no_grad():
        # Predicts volume for [Bass, Piano] simultaneously
        # Shape: (1, Time, 2)
        predicted_dynamics = arranger(l_leader_t)

    # --- 4. ASSEMBLE THE BAND ---
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    for i, inst_name in enumerate(ACCOMPANIMENT):
        print(f"   🎙️ Recording {inst_name}...", flush=True)

        # A. Pitch Source: RETRIEVAL (Cosine)
        # Find a chunk that matches the Guitar's Key
        best_idx, shift = find_best_match_smart(
            p_lead_np,
            datasets[inst_name]['pitch'],
            datasets[inst_name]['loudness']
        )

        # Get the notes
        p_retrieved = datasets[inst_name]['pitch'][best_idx: best_idx + num_chunks].reshape(-1)
        # Apply Key Shift
        p_retrieved = p_retrieved * (2 ** (shift / 12.0))

        # B. Loudness Source: GENERATION (Transformer)
        # Get the volume curve the AI invented for this instrument
        l_generated = predicted_dynamics[:, :, i].unsqueeze(-1)

        # Apply Boost to make it punchy
        l_generated = boost_dynamics(l_generated, threshold=0.4, ratio=2.5)

        # --- SPOTLIGHT LOGIC (Manual Fades) ---
        if inst_name == 'Piano':
            # Fade IN over the duration (Starts quiet, ends loud)
            fade_in = torch.linspace(0, 1, l_generated.shape[1]).unsqueeze(0).unsqueeze(-1).to(device)
            l_generated = l_generated * fade_in

        if inst_name == 'Bass':
            # Fade OUT over the duration (Starts loud, ends quiet)
            fade_out = torch.linspace(1, 0, l_generated.shape[1]).unsqueeze(0).unsqueeze(-1).to(device)
            l_generated = l_generated * fade_out

        # C. Synthesis: DDSP
        # Combine Retrieved Pitch + Generated Loudness
        model = load_audio_model(MODEL_REGISTRY[inst_name]).to(device)
        p_t = torch.from_numpy(p_retrieved).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            # Run the model with MIXED inputs
            controls = model.model(p_t, l_generated)
            controls['noise'] *= NOISE_SCALE

            target_samples = p_t.shape[1] * 64
            harm = harmonic_synthesis(p_t, controls['amplitudes'], controls['harmonics'], n_samples=target_samples)
            nz = noise_synthesis(controls['noise'], n_samples=target_samples)

            track_audio = (harm + nz).squeeze().cpu().numpy()

            # Mix
            min_len = min(len(final_mix), len(track_audio))
            final_mix[:min_len] += track_audio[:min_len]

            sf.write(output_dir / f"complete_stem_{inst_name}.wav", track_audio, 16000)

    # 5. SAVE FINAL
    out_name = f"Complete_Arrangement_{LEADER}_idx{start_idx}.wav"
    sf.write(output_dir / out_name, final_mix, 16000)

    print(f"\n🎉 DONE! Created via Hybrid Retrieval + Generation.")
    print(f"   🎧 Output: {output_dir / out_name}")
    try:
        save_report_card(str(output_dir / out_name))
    except:
        pass


if __name__ == "__main__":
    run_complete_arrangement()