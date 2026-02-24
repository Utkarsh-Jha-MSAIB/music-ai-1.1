import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import os
import math
import random
import sys

try:
    from src.train import AudioSynthTrainer
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis
except ImportError:
    from src.models.train_instrument import AudioSynthTrainer
    from src.models.signal_processing import harmonic_synthesis, noise_synthesis

# ==============================================================================
# 🎛️  USER CONTROL PANEL
# ==============================================================================

MODEL_REGISTRY = {
    'Bass': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_0/checkpoints/epoch=19-step=1420.ckpt",
    'Strings': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_1/checkpoints/epoch=19-step=160.ckpt",
    'Piano': "/u/erdos/csga/uj3/projects/project_111/src/models/lightning_logs/version_2/checkpoints/epoch=19-step=1520.ckpt",
}

MODE = 'mix'

# --- MIX SETTINGS ---
# To hear pure piano, we set Bass/Strings to 0
HYBRID_MIX = {
    'Bass': 0.3,
    'Strings': 2.0,
    'Piano': 0.7
}

# --- GENERATION SETTINGS ---
INPUT_MELODY = 'Bass'
GENERATE_SECONDS = 20

# --- PITCH SHIFT ---
# 0 = Normal, 1 = +1 Octave, 2 = +2 Octaves
PITCH_SHIFT_OCTAVES = 1

# --- NOISE CONTROL ---
NOISE_SCALE = 0.05

# ==============================================================================

CONST_WINDOW = 4.0


def load_model(instrument_name):
    if instrument_name not in MODEL_REGISTRY:
        print(f"❌ Error: '{instrument_name}' not in registry.")
        sys.exit(1)
    ckpt_path = MODEL_REGISTRY[instrument_name]

    if "PASTE" in ckpt_path or not os.path.exists(ckpt_path):
        root = Path(__file__).resolve().parents[1]
        found = list(root.rglob(f"*{instrument_name}*.ckpt"))
        if not found:
            print(f"❌ Error: Invalid path for {instrument_name}.")
            sys.exit(1)
        ckpt_path = str(found[0])

    print(f"⚡ Loading {instrument_name}...", flush=True)
    model = AudioSynthTrainer.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def blend_weighted(controls_map):
    hybrid = {}
    first_key = list(controls_map.keys())[0]
    sample_ctrl = controls_map[first_key]
    total_weight = sum(HYBRID_MIX.values()) + 1e-7

    for param in sample_ctrl.keys():
        weighted_sum = 0
        for inst_name, ctrl in controls_map.items():
            weight = HYBRID_MIX[inst_name]
            weighted_sum += ctrl[param] * weight
        hybrid[param] = weighted_sum / total_weight
    return hybrid


def run_generator():
    print(f"🎹 Starting Generator...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    # 1. Load Models
    for name in HYBRID_MIX.keys():
        if HYBRID_MIX[name] > 0:
            m = load_model(name)
            m.to(device)
            models[name] = m

    target_name = "Mix_" + "_".join([f"{k}{v}" for k, v in HYBRID_MIX.items() if v > 0])

    # 2. Load Data
    project_root = Path(__file__).resolve().parents[1]
    if project_root.name == 'src': project_root = project_root.parent
    data_path = project_root / 'data' / 'processed' / 'features'

    print(f"💿 Loading Melody from {INPUT_MELODY}...", flush=True)
    try:
        p_path = data_path / f'pitch_{INPUT_MELODY}.npy'
        l_path = data_path / f'loudness_{INPUT_MELODY}.npy'
        a_path = data_path / f'audio_{INPUT_MELODY}.npy'
        pitch_data = np.load(p_path)
        loudness_data = np.load(l_path)
        audio_data = np.load(a_path)
    except FileNotFoundError:
        print(f"❌ Error: Data for {INPUT_MELODY} not found.")
        return

    # 3. Prepare Tensor
    num_chunks = math.ceil(GENERATE_SECONDS / CONST_WINDOW)
    start_idx = random.randint(0, len(pitch_data) - num_chunks - 1)

    p_long = pitch_data[start_idx: start_idx + num_chunks].reshape(-1)
    l_long = loudness_data[start_idx: start_idx + num_chunks].reshape(-1)
    a_long = audio_data[start_idx: start_idx + num_chunks].reshape(-1)

    # --- APPLY PITCH SHIFT ---
    if PITCH_SHIFT_OCTAVES != 0:
        print(f"   ⬆️ Shifting Pitch up by {PITCH_SHIFT_OCTAVES} Octaves...")
        p_long = p_long * (2 ** PITCH_SHIFT_OCTAVES)

    p = torch.from_numpy(p_long).unsqueeze(0).unsqueeze(-1).to(device)
    l = torch.from_numpy(l_long).unsqueeze(0).unsqueeze(-1).to(device)

    # 4. Generate
    print("   Synthesizing...", flush=True)
    with torch.no_grad():
        all_controls = {}
        for name, m in models.items():
            all_controls[name] = m.model(p, l)

        controls = blend_weighted(all_controls)
        controls['noise'] = controls['noise'] * NOISE_SCALE

        target_samples = p.shape[1] * 64
        harmonics = harmonic_synthesis(p, controls['amplitudes'], controls['harmonics'], n_samples=target_samples)
        noise = noise_synthesis(controls['noise'], n_samples=target_samples)
        fake_audio = harmonics + noise

    # 5. Save
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    fake_flat = fake_audio.squeeze().cpu().numpy()

    # Save Generated (AI)
    out_path_gen = output_dir / f"gen_{target_name}_shift{PITCH_SHIFT_OCTAVES}.wav"
    sf.write(out_path_gen, fake_flat, 16000)

    # Save Real Source (Original Bass)
    out_path_real = output_dir / f"real_source_{INPUT_MELODY}_{GENERATE_SECONDS}s.wav"
    sf.write(out_path_real, a_long, 16000)

    print(f"🎉 Saved AI:   {out_path_gen}")
    print(f"✅ Saved Real: {out_path_real}")


if __name__ == "__main__":
    run_generator()