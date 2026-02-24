from pathlib import Path
import numpy as np

p = Path(r"C:\Music_AI_1.1\core\data\processed\features\audio_Guitar.npy")

# Read header (no full load)
arr = np.load(p, mmap_mode="r")
shape, dtype = arr.shape, arr.dtype

expected_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
actual_bytes = p.stat().st_size

print("shape:", shape)
print("dtype:", dtype)
print("expected GB:", expected_bytes / (1024**3))
print("actual   GB:", actual_bytes / (1024**3))
