"""The daily audio brief: a Kyutai Pocket TTS narration clip per paper, with word timings, published next to the feed.
Contract: scienceos.atlas.audio-day/1 (schemas/audio.schema.json). Heavy dependencies (torch, pocket-tts, faster-whisper) are imported lazily,
only by the real engine and the real aligner, so everything else here is importable and testable without them."""

SCHEMA_ID = "scienceos.atlas.audio-day/1"
