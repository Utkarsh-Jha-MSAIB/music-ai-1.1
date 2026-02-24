import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import math
import random
import sys
import torch.nn.functional as F

try:
    from src.train import AudioSynthTrainer
    from src.models.arranger import NeuralArranger
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis
    from src.visualization.song_in_detail import save_report_card
except ImportError:
    from src.models.train_instrument import AudioSynthTrainer
    from src.models.decoder_conductor import NeuralArranger
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis

# ==============================================================================
# 🧠 NEURAL CONDUCTOR CONSOLE
# ==============================================================================

# 1. THE ARRANGER BRAIN (The one you JUST trained)
ARRANGER_CKPT = "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_5/checkpoints/epoch=29-step=1080.ckpt"

# 2. THE MUSICIANS (The Audio Generators)
MODEL_REGISTRY = {
    'Bass': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_0/checkpoints/epoch=19-step=1420.ckpt",
    'Piano': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_2/checkpoints/epoch=19-step=1520.ckpt",
}

LEADER = 'Guitar'
FOLLOWERS = ['Bass', 'Piano']

MANUAL_START_INDEX = 790  # Keep this to fix the song

HARMONY_OFFSETS = {
    'Bass': -12,
    'Piano': 7
}

LAZY_BASS = True
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
    model = NeuralArranger(num_followers=len(FOLLOWERS))
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_composition(lead_vol, band_vols, names, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(lead_vol, label=f'LEADER: {LEADER}', color='black', linewidth=2, linestyle='--')
    colors = ['#e24a90', '#4a90e2', '#f5a623']
    for i, name in enumerate(names):
        curve = band_vols[:, i]
        plt.plot(curve, label=f'{name}', color=colors[i % len(colors)], linewidth=2)
    plt.title(f"Neural Arrangement (Boosted): Dynamics over {GENERATE_SECONDS}s")
    plt.ylabel("Predicted Volume (0-1)")
    plt.xlabel("Time Frames")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def find_active_chunk(loudness_data, num_chunks):
    print("   🔎 Scanning for active audio candidates...")
    candidates = []
    total_len = len(loudness_data)
    step_size = 5
    for i in range(0, total_len - num_chunks, step_size):
        chunk = loudness_data[i: i + num_chunks]
        energy = np.sum(chunk)
        candidates.append((energy, i))
    candidates.sort(key=lambda x: x[0], reverse=True)
    pool_size = min(20, len(candidates))
    top_candidates = candidates[:pool_size]
    if pool_size == 0: return 0
    winner = random.choice(top_candidates)
    print(f"   🎲 Selected Candidate #{top_candidates.index(winner)} (Energy: {winner[0]:.2f})")
    print(f"   📍 INDEX: {winner[1]}")
    return winner[1]


def simplify_pitch(pitch_curve, factor=8):
    p_small = pitch_curve[:, ::factor, :]
    p_blocky = F.interpolate(p_small.permute(0, 2, 1), size=pitch_curve.shape[1], mode='nearest')
    return p_blocky.permute(0, 2, 1)


# --- THE FIXES: DYNAMICS BOOSTERS ---

def boost_dynamics(curve, threshold=0.5, ratio=3.0):
    """
    Expands the dynamic range.
    Anything below threshold gets pushed down.
    Anything above threshold gets pushed up.
    """
    # 1. Center the data around 0
    centered = curve - threshold
    # 2. Expand it (Multiply)
    expanded = centered * ratio
    # 3. Shift back
    result = expanded + threshold
    # 4. Clip
    return torch.clamp(result, 0.0, 1.0)


def hard_gate(volume_curve, threshold=0.4):  # Increased threshold slightly
    """Silences any sound that is too quiet."""
    mask = (volume_curve > threshold).float()
    return volume_curve * mask


def run_neural_arrangement():
    print(f"🎹 Starting Neural Arrangement (Conductor: {LEADER})...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD DATA
    project_root = Path(__file__).resolve().parents[1]
    if project_root.name == 'src': project_root = project_root.parent
    data_path = project_root / 'data' / 'processed' / 'features'

    try:
        p_path = data_path / f'pitch_{LEADER}.npy'
        l_path = data_path / f'loudness_{LEADER}.npy'
        a_path = data_path / f'audio_{LEADER}.npy'
        pitch_data = np.load(p_path)
        loudness_data = np.load(l_path)
        audio_data = np.load(a_path)
    except FileNotFoundError:
        return

    # 2. PREPARE TENSORS
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)

    if MANUAL_START_INDEX is not None:
        start_idx = MANUAL_START_INDEX
    else:
        start_idx = find_active_chunk(loudness_data, num_chunks)

    p_long = pitch_data[start_idx: start_idx + num_chunks].reshape(-1)
    l_long = loudness_data[start_idx: start_idx + num_chunks].reshape(-1)
    a_long = audio_data[start_idx: start_idx + num_chunks].reshape(-1)

    p_leader = torch.from_numpy(p_long).unsqueeze(0).unsqueeze(-1).to(device)
    l_leader = torch.from_numpy(l_long).unsqueeze(0).unsqueeze(-1).to(device)

    # 3. ARRANGER
    print("🧠 Arranger is composing dynamics...", flush=True)
    arranger = load_arranger_model(ARRANGER_CKPT).to(device)

    with torch.no_grad():
        # Initial Prediction (The "Safe" Curve)
        predicted_dynamics = arranger(l_leader)

        # --- APPLY FIX: BOOST DYNAMICS ---
        print("   💥 Applying Dynamic Expansion...", flush=True)

        # Bass (Channel 0): Make it thump.
        # Threshold 0.5 means anything below 0.5 volume gets pushed to silence.
        predicted_dynamics[:, :, 0] = boost_dynamics(predicted_dynamics[:, :, 0], threshold=0.55, ratio=2.5)

        # Piano (Channel 1): Make it sparkle.
        # Higher ratio makes it very staccato.
        predicted_dynamics[:, :, 1] = boost_dynamics(predicted_dynamics[:, :, 1], threshold=0.45, ratio=3.0)

    # Save Plot (Check this to see if lines are spiky now!)
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    idx_tag = f"_idx{start_idx}"
    plot_composition(l_leader.squeeze().cpu().numpy(), predicted_dynamics.squeeze().cpu().numpy(), FOLLOWERS,
                     output_dir / f"score_{LEADER}{idx_tag}.png")

    # Patch Silence
    silence_mask = (p_leader < 10)
    valid_pitch = p_leader[~silence_mask]
    avg_pitch_val = torch.mean(valid_pitch) if len(valid_pitch) > 0 else 100.0
    p_filled = p_leader.clone()
    p_filled[silence_mask] = avg_pitch_val

    # 4. RECORD BAND
    final_mix_len = p_leader.shape[1] * 64
    final_mix = np.zeros(final_mix_len)

    for i, inst_name in enumerate(FOLLOWERS):
        print(f"   🎙️ Recording {inst_name}...", flush=True)

        # A. Get Boosted Volume
        l_pred = predicted_dynamics[:, :, i].unsqueeze(-1)

        # B. Pitch Logic
        semitones = HARMONY_OFFSETS.get(inst_name, 0)
        p_shifted = p_filled * (2 ** (semitones / 12.0))

        if inst_name == 'Bass' and LAZY_BASS:
            p_shifted = simplify_pitch(p_shifted, factor=50)

            # C. Generate
        musician = load_audio_model(MODEL_REGISTRY[inst_name]).to(device)

        with torch.no_grad():
            controls = musician.model(p_shifted, l_pred)
            controls['noise'] *= NOISE_SCALE

            target_samples = p_leader.shape[1] * 64
            harmonics = harmonic_synthesis(p_shifted, controls['amplitudes'], controls['harmonics'],
                                           n_samples=target_samples)
            noise = noise_synthesis(controls['noise'], n_samples=target_samples)

            track_audio = (harmonics + noise).squeeze().cpu().numpy()
            sf.write(output_dir / f"stem_{inst_name}.wav", track_audio, 16000)
            final_mix += track_audio

    # 5. SAVE MIX
    out_name = f"neural_arrange_{LEADER}{idx_tag}.wav"
    sf.write(output_dir / out_name, final_mix, 16000)
    sf.write(output_dir / f"reference_{LEADER}_source{idx_tag}.wav", a_long, 16000)

    print(f"\n🎉 DONE!")
    print(f"   🎧 Mix: {output_dir / out_name}")
    try:
        save_report_card(str(output_dir / out_name))
    except:
        pass


if __name__ == "__main__":
    run_neural_arrangement()