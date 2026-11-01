"""The speech engine behind a one-method interface, so the clip builder and the tests never touch torch.

`PocketEngine` is Kyutai Pocket TTS (pip `pocket-tts`, MIT code, CC BY 4.0 weights): 100M parameters, 24 kHz mono, about 9x real time on one CPU
thread. The model and the voice state are loaded once per run. Built-in voices need no Hugging Face login (the library falls back to the ungated
weights repo); only voice *cloning* is gated, and we do not clone."""

from __future__ import annotations

import importlib.util
import logging
from importlib import metadata
from typing import Protocol

import numpy as np

log = logging.getLogger("atlas.audio.engine")


class TtsEngine(Protocol):
    name: str
    version: str
    model: str
    voice: str
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """One sentence -> float32 mono samples in [-1, 1] at `sample_rate` (leading/trailing silence allowed; the builder trims it)."""
        ...

    def close(self) -> None: ...


def pocket_available() -> bool:
    """pocket-tts and torch are installed (checked without importing torch: that takes seconds and half a gigabyte)."""
    try:
        metadata.version("pocket-tts")
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def whisper_available() -> bool:
    try:
        metadata.version("faster-whisper")
        return True
    except Exception:
        return False


def credit_for(voice: str) -> str:
    return f"Voice: {voice.capitalize()} · Pocket TTS by Kyutai (CC BY 4.0)"


class PocketEngine:
    name = "pocket-tts"

    def __init__(self, voice: str = "alba", language: str = "english", model: str = "kyutai/pocket-tts", threads: int = 0):
        import torch
        from pocket_tts import TTSModel   # importing it pins torch to one thread

        self.version = metadata.version("pocket-tts")
        self.model, self.voice = model, voice
        if threads > 0:
            torch.set_num_threads(threads)
        log.info("Loading Pocket TTS %s (%s, voice %s) on CPU ...", self.version, language, voice)
        self._tts = TTSModel.load_model(language=language)
        self._state = self._tts.get_state_for_audio_prompt(voice)
        self.sample_rate = int(self._tts.sample_rate)
        self._torch = torch

    def synthesize(self, text: str) -> np.ndarray:
        audio = self._tts.generate_audio(self._state, text)   # no torch.inference_mode() here: the library's decoder thread runs outside it
        return audio.detach().cpu().float().reshape(-1).numpy().astype(np.float32)

    def close(self) -> None:
        self._tts = None
        self._state = None


def load_engine(voice: str, language: str, model: str, threads: int = 0) -> TtsEngine:
    """The one place the real engine is created (tests monkeypatch it)."""
    return PocketEngine(voice=voice, language=language, model=model, threads=threads)
