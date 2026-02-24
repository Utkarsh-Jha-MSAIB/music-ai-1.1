from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import json

from src.models.perform_music_service import run_arrangement_for_api


@dataclass
class GenerateRequest:
    seconds: float = 20.0
    start_index: int = 1855


class MusicGenerator:
    """
    DB-only wrapper.
    Returns a "run package": final + stems + meta + (optional) report cards.
    """
    def __init__(self):
        pass

    def generate(self, req: GenerateRequest) -> dict:
        t0 = time.time()

        final_wav = Path(
            run_arrangement_for_api(seconds=float(req.seconds), start_index=int(req.start_index))
        )

        out_dir = final_wav.parent
        runtime_sec = time.time() - t0

        # meta.json written by perform_music
        meta_path = out_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        # Collect wavs (final + stems + reference)
        wavs = sorted(out_dir.glob("*.wav"))

        # Collect report cards / images if you start saving them (png)
        pngs = sorted(out_dir.glob("*.png"))

        return {
            "run_id": out_dir.name,
            "output_dir": str(out_dir),
            "runtime_sec": runtime_sec,
            "final_wav": str(final_wav),
            "wavs": [str(p) for p in wavs],
            "pngs": [str(p) for p in pngs],
            "meta": meta,
        }
