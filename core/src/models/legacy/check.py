import numpy as np
import librosa
import torch
import sys
import math
from pathlib import Path
from scipy.stats import entropy
import random
import os
import shutil
import warnings
from tqdm import tqdm
from collections import defaultdict

# Suppress all Librosa warnings before execution
warnings.filterwarnings("ignore", module="librosa")

# Import engine
try:
    import src.models.perform_music as neural_band
except ImportError as e:
    sys.exit(f"❌ FATAL ERROR: Cannot import neural_band. Check dependencies. Error: {e}")

try:
    import torchcrepe

    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
INPUT_FILENAME = "data/raw/user_uploads/ITR_Instrumental.wav"
TARGET_DURATION = 30.0
BASE_CHUNK_DURATION = 2.0
TEMP_LEADER_NAME = 'RAG_DEBUG_FINAL'

# --- NEW: GRU MODEL PLACEHOLDER ---
GRU_MODEL_CHECKPOINT = "checkpoints/Guitar.ckpt"

# --- INTERNAL CONFIG ---
SAMPLE_RATE = 16000
FRAME_RATE = 250
CROSSFADE_TIME_SAMPLES = 0
FIXED_BEST_INDEX = 41021

# --- DYNAMIC ENERGY FILTER CONSTANTS ---
RMS_TOLERANCE_PCT = 0.20
LOUDNESS_TRIM_THRESHOLD = 0.15
FINAL_LOUDNESS_THRESHOLD = 1e-6
# CRITICAL FINAL FIX: Use a high threshold based on visual plot evidence (1500 Hz spike)
PITCH_OUTLIER_HZ = 1400.0

# --- PATH FINDING (Omitted for brevity) ---
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != 'project_111' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name != 'project_111':
    sys.exit("❌ FATAL ERROR: Could not locate 'project_111' root directory.")
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE_PATH = PROJECT_ROOT / INPUT_FILENAME
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'features'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'


# ==============================================================================
# Helper functions
# ==============================================================================
def crossfade_audio(audio_out, audio_in, length_samples):
    if len(audio_out) < length_samples: return audio_out
    fade_out = np.linspace(1.0, 0.0, length_samples)
    fade_in = np.linspace(0.0, 1.0, length_samples)
    audio_out[-length_samples:] *= fade_out
    audio_in[:length_samples] *= fade_in
    return audio_out


def cleanup_temp_files(temp_name):
    temp_files = [
        DATA_DIR / f'pitch_{temp_name}.npy',
        DATA_DIR / f'loudness_{temp_name}.npy',
        DATA_DIR / f'audio_{temp_name}.npy'
    ]
    cleaned = 0
    for f in temp_files:
        if f.exists():
            os.remove(f)
            cleaned += 1
    if cleaned > 0:
        pass


def post_process_trim(audio_array):
    """Removes silence based on fixed absolute loudness threshold (1e-6)."""
    abs_audio = np.abs(audio_array)
    trim_threshold = FINAL_LOUDNESS_THRESHOLD
    nonzero_indices = np.where(abs_audio >= trim_threshold)[0]

    if nonzero_indices.size == 0:
        return np.array([], dtype=audio_array.dtype)

    start_index = nonzero_indices[0]
    end_index = nonzero_indices[-1] + 1

    return audio_array[start_index:end_index]


def trim_pitch_outliers(audio_array, pitch_array, sr, frame_rate, outlier_hz=PITCH_OUTLIER_HZ):
    """Removes the audio segment corresponding to pitch artifacts (> PITCH_OUTLIER_HZ)."""

    # 1. Identify outlier frames
    outlier_frames = np.where(pitch_array > outlier_hz)[0]

    if outlier_frames.size == 0:
        return audio_array

    # 2. Find the boundary of the outlier event
    start_outlier_frame = outlier_frames[0]
    end_outlier_frame = outlier_frames[-1]

    # Calculate the sample indices corresponding to the frames
    hop_length = sr // frame_rate

    # The start sample of the problematic audio section:
    start_sample = start_outlier_frame * hop_length

    # The end sample of the problematic audio section:
    end_sample = (end_outlier_frame + 1) * hop_length

    # Safety check: ensure end_sample does not exceed array bounds
    end_sample = min(end_sample, len(audio_array))

    # 3. Reconstruct the audio, removing the outlier segment
    # Audio before the artifact + Audio after the artifact
    audio_before = audio_array[:start_sample]
    audio_after = audio_array[end_sample:]

    # Concatenate, effectively creating a hard-cut removal of the artifact region
    return np.concatenate([audio_before, audio_after])


def run_single_step_rag_debug():
    print(f"🚀 Starting DEBUG MODE Pipeline (Fixed Index: {FIXED_BEST_INDEX}, Target: {TARGET_DURATION}s)...")

    if not INPUT_FILE_PATH.exists():
        sys.exit(f"❌ ERROR: Input audio file not found at: {INPUT_FILE_PATH}")

    try:
        # 1. LOAD USER INPUT AND DETERMINE SEARCH DURATION
        y_user_input, sr = librosa.load(str(INPUT_FILE_PATH), sr=SAMPLE_RATE)
        input_duration_original = len(y_user_input) / SAMPLE_RATE

        # --- CRITICAL FIX 1: FEATURE-BASED TRIM TRAILING SILENCE (ON INPUT) ---
        input_rms_frames = librosa.feature.rms(y=y_user_input, frame_length=512, hop_length=128, center=True)[0]
        max_rms = np.max(input_rms_frames)
        trim_threshold = max_rms * LOUDNESS_TRIM_THRESHOLD

        active_frames = np.where(input_rms_frames > trim_threshold)[0]

        if active_frames.size > 0:
            last_active_frame = active_frames[-1]
            last_active_sample = (last_active_frame + 1) * 128

            if last_active_sample < len(y_user_input):
                silence_removed_s = (len(y_user_input) - last_active_sample) / SAMPLE_RATE
                print(
                    f"   [Input Trim]: Removed {silence_removed_s:.2f}s of trailing silence below {trim_threshold:.4f} RMS.")
                y_user_input = y_user_input[:last_active_sample]

                # --- End Trim Fix ---

        input_len_samples = len(y_user_input)
        input_duration = input_len_samples / SAMPLE_RATE

        if TARGET_DURATION <= input_duration:
            sys.exit(f"⚠️ Warning: Target duration must be greater than trimmed input duration.")

        # --- Calculate required extension length ---
        TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)
        required_extension_len = TARGET_SAMPLES - input_len_samples + CROSSFADE_TIME_SAMPLES

        SEARCH_DURATION = required_extension_len / SAMPLE_RATE
        print(
            f"   Input duration (MUSICAL): {input_duration:.1f}s. Using fixed {SEARCH_DURATION:.1f}s extension segment.")

        # --- Database Loading ---
        db_audio = np.load(DATA_DIR / 'audio_Guitar.npy').reshape(-1)

        # 2. CALCULATE NECESSARY SIZES
        base_chunk_samples = int(BASE_CHUNK_DURATION * SAMPLE_RATE)
        search_len_samples = required_extension_len
        start_s_db = FIXED_BEST_INDEX * base_chunk_samples

        if start_s_db + search_len_samples > len(db_audio):
            sys.exit("❌ CRITICAL ERROR: Fixed index and search duration exceed database bounds.")

        # 3. CONCATENATION AND INJECTION (Bypassing search)
        retrieved_audio = db_audio[start_s_db: start_s_db + search_len_samples]

        # --- B. Stitch the Input and Retrieval (HARD CUT) ---
        y_final_audio_raw = np.concatenate([y_user_input, retrieved_audio])

        # 4. PRE-PROCESSING FOR ARTIFACT REMOVAL (Pitch and Loudness)

        # First, calculate pitch for the entire raw audio array
        # NOTE: We use the Frame Rate defined in CONFIG to match hop length
        pitch_hop_length = SAMPLE_RATE // FRAME_RATE
        raw_pitch_arr, _, _ = librosa.pyin(y_final_audio_raw, fmin=40, fmax=2000, sr=SAMPLE_RATE,
                                           hop_length=pitch_hop_length)
        raw_pitch_arr = np.nan_to_num(raw_pitch_arr)

        # --- CRITICAL FINAL FIX 2: REMOVE PITCH OUTLIER ARTIFACT ---
        print(f"   [Post-Process]: Checking for pitch artifacts > {PITCH_OUTLIER_HZ} Hz...")
        y_artifact_fixed = trim_pitch_outliers(y_final_audio_raw, raw_pitch_arr, sr=SAMPLE_RATE, frame_rate=FRAME_RATE)

        if len(y_artifact_fixed) < len(y_final_audio_raw):
            removed_samples = len(y_final_audio_raw) - len(y_artifact_fixed)
            removed_s = removed_samples / SAMPLE_RATE
            print(f"   ✅ ARTIFACT REMOVAL SUCCESS: Removed {removed_s:.3f}s pitch anomaly region.")
        else:
            print("   ⚠️ ARTIFACT CHECK: No extreme pitch outliers found.")
            y_artifact_fixed = y_final_audio_raw

        # --- FINAL TRIM (Amplitude-based trim for any remaining zero-samples at ends) ---
        y_final_audio = post_process_trim(y_artifact_fixed)

        # --- Check final duration ---
        final_audio_len_s = len(y_final_audio) / SAMPLE_RATE

        # --- FEATURE EXTRACTION FOR ARRANGEMENT ENGINE ---
        print(f"   (Re)Extracting Pitch/Loudness for final track ({final_audio_len_s:.2f}s) for arrangement engine...")

        final_pitch_arr, _, _ = librosa.pyin(y_final_audio, fmin=40, fmax=2000, sr=SAMPLE_RATE, hop_length=64)
        final_pitch_arr = np.nan_to_num(final_pitch_arr)

        final_loudness_arr = librosa.feature.rms(y=y_final_audio, frame_length=2048, hop_length=64, center=True)[0]
        final_loudness_arr = np.nan_to_num(final_loudness_arr)

        N_frames = min(len(final_pitch_arr), len(final_loudness_arr))
        N_chunks = 1
        final_len_samples_output = len(y_final_audio)

        temp_name_rank = TEMP_LEADER_NAME
        np.save(DATA_DIR / f'pitch_{temp_name_rank}.npy', final_pitch_arr[:N_frames].reshape(N_chunks, N_frames))
        np.save(DATA_DIR / f'loudness_{temp_name_rank}.npy', final_loudness_arr[:N_frames].reshape(N_chunks, N_frames))
        np.save(DATA_DIR / f'audio_{temp_name_rank}.npy', y_final_audio.reshape(N_chunks, final_len_samples_output))

        # 5. EXECUTE GENERATION
        neural_band.LEADER = temp_name_rank
        neural_band.GENERATE_SECONDS = final_audio_len_s
        neural_band.MANUAL_START_INDEX = 0

        print(f"\n   ✅ Stitching Complete. Running arrangement on fixed index {FIXED_BEST_INDEX}...")
        neural_band.run_complete_arrangement()

        # 6. CHECK OUTPUT FILE
        default_out_name = f"Smart_Arrangement_{temp_name_rank}_Unison_0.wav"
        default_out_path = OUTPUT_DIR / default_out_name

        if default_out_path.exists():
            print(
                f"\n✅ Output Saved! Final Duration: {final_audio_len_s:.2f}s. Please check for file: {default_out_path}")
        else:
            print(f"\n❌ Output File Not Found. Check for temporary files in {OUTPUT_DIR}.")


    except Exception as e:
        print(f"❌ Critical Error during RAG execution: {e}")

    finally:
        cleanup_temp_files(TEMP_LEADER_NAME)


if __name__ == "__main__":
    run_single_step_rag_debug()