# core/inference_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

from src.models.audio_rag_service import run_audio_rag_for_api


@dataclass
class RagRequest:
    """
    Request for the Audio RAG pipeline.

    These map to overrides in configs/audio_rag.yaml.
    """
    input_filename: str
    target_duration: float = 30.0
    top_k: int = 10

    # Optional knobs (only add if you want to override YAML; else leave None)
    db_instrument: Optional[str] = None
    stride: Optional[int] = None
    attempts: Optional[int] = None


class AudioRAGGenerator:
    """
    API-friendly wrapper for Audio RAG, consistent with your DB-only MusicGenerator.
    Returns a structured package {runtime_sec, results, config_snapshot}.
    """
    def generate(self, req: RagRequest) -> dict[str, Any]:
        t0 = time.time()

        payload = run_audio_rag_for_api(
            input_filename=req.input_filename,
            target_duration=req.target_duration,
            top_k=req.top_k,
            db_instrument=req.db_instrument,
            stride=req.stride,
            attempts=req.attempts,
        )

        runtime_sec = time.time() - t0
        payload.setdefault("runtime_sec", runtime_sec)
        return payload
