import numpy as np
import librosa
from pathlib import Path
import math

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "configs").exists():
            return p
    return start.parents[2]  # fallback

def main():
    repo_root = find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "core" / "data" / "processed" / "features"

    instrument = "Guitar"
    audio_path = data_dir / f"audio_{instrument}.npy"

    # MUST match your YAML
    sr = 16000
    hop = 512  # chroma_hop_length

    # Chunk size: tune this (30–120s is usually safe)
    chunk_seconds = 60
    chunk_samples = int(chunk_seconds * sr)

    # A small overlap reduces boundary artifacts between chunks
    overlap_seconds = 2
    overlap_samples = int(overlap_seconds * sr)

    # Output (frames)
    out_npy = data_dir / f"chroma_cqt_frames_{instrument}.npy"
    out_memmap = data_dir / f"chroma_cqt_frames_{instrument}.dat"

    print("Loading DB audio (memmap)...")
    db_audio = np.load(audio_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    n_samples = int(db_audio.shape[0])

    # Total frames for hop
    n_frames = 1 + (n_samples // hop)
    print(f"DB samples: {n_samples:,}  -> approx chroma frames: {n_frames:,}")

    # Create an output memmap (12, n_frames)
    print(f"Allocating output memmap: {out_memmap}")
    chroma_mm = np.memmap(out_memmap, dtype="float32", mode="w+", shape=(12, n_frames))

    # Process in chunks
    num_chunks = math.ceil(n_samples / chunk_samples)
    print(f"Processing in {num_chunks} chunks of ~{chunk_seconds}s (overlap {overlap_seconds}s)")

    for ci in range(num_chunks):
        start = ci * chunk_samples
        end = min(n_samples, (ci + 1) * chunk_samples)

        # extend with overlap on both sides (clamped)
        start_ext = max(0, start - overlap_samples)
        end_ext = min(n_samples, end + overlap_samples)

        y = np.asarray(db_audio[start_ext:end_ext], dtype=np.float32)

        # Compute chroma on this chunk.
        # Critical: tuning=0 avoids expensive estimate_tuning call.
        C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, tuning=0.0)  # shape (12, Tchunk)

        # Figure out where these frames land in the global timeline.
        # Frame index corresponding to start_ext sample:
        frame0 = start_ext // hop

        # Now we want to write only the non-overlap interior corresponding to [start, end)
        # Compute interior frame range relative to C
        interior_start_frames = (start - start_ext) // hop
        interior_end_frames = interior_start_frames + ((end - start) // hop)

        C_int = C[:, interior_start_frames:interior_end_frames]

        out_start = start // hop
        out_end = out_start + C_int.shape[1]

        chroma_mm[:, out_start:out_end] = C_int.astype(np.float32, copy=False)

        if (ci + 1) % 5 == 0 or ci == 0 or ci == num_chunks - 1:
            print(f"  chunk {ci+1}/{num_chunks}: samples [{start:,}:{end:,}] -> frames [{out_start:,}:{out_end:,}]")

    # Flush memmap to disk
    chroma_mm.flush()

    # Save as .npy header pointing to data (so np.load works nicely)
    # We load the memmap and then np.save it; this will create a normal .npy.
    # NOTE: This step will read+write the whole array; if that's too slow/disk-heavy,
    # you can skip it and instead load via np.memmap in your retrieval code.
    print("Writing .npy wrapper (may take time, but doesn't require huge RAM)...")
    chroma_final = np.memmap(out_memmap, dtype="float32", mode="r", shape=(12, n_frames))
    np.save(out_npy, chroma_final)

    mb = out_npy.stat().st_size / (1024 * 1024)
    print(f"Done. Saved: {out_npy} ({mb:.1f} MB) shape={(12, n_frames)} dtype=float32")

if __name__ == "__main__":
    main()
