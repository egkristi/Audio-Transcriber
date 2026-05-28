# Audio-Transcriber Issues

This file tracks known issues, bugs, and feature gaps identified during the project audit.

## Critical

### #1: Transcription config parameters not passed to WhisperX
- **File:** `src/transcribe.py`
- **Status:** Resolved (2026-05-28)
- **Description:** The `Transcriber` class accepts `beam_size`, `word_timestamps`, `condition_on_previous_text`, and `initial_prompt` in its constructor and `transcribe()` method, but the actual `whisperx` call in `transcribe()` does not use these parameters. Only `language` and `batch_size` are passed.
- **Impact:** Users cannot control decoding behavior, word-level timestamps, or inject vocabulary via initial prompts.
- **Fix:** Pass `beam_size`, `word_timestamps`, `vad_filter`, `condition_on_previous_text`, and `initial_prompt` into the `whisperx` transcription call.

### #2: Dependency mismatch — `analyze.py` uses `whisper` not declared in `pyproject.toml`
- **File:** `src/analyze.py`, `pyproject.toml`
- **Status:** Resolved (2026-05-28) — improved in v0.1.2
- **Description:** `detect_language()` in `analyze.py` imports `whisper` (OpenAI's reference implementation), but `pyproject.toml` only lists `whisperx` as a dependency. This can cause a runtime `ModuleNotFoundError` on fresh installs.
- **Impact:** Language detection fails on clean environments.
- **Fix:** Reimplemented language detection using `faster_whisper` directly (cached across calls), processing only the first 30 seconds of audio for speed. No standalone `whisper` dependency needed.

### #3: Diarization requires Hugging Face auth but has no helper
- **File:** `src/diarize.py`
- **Status:** Resolved (2026-05-28)
- **Description:** `Pipeline.from_pretrained(..., use_auth_token=True)` is hardcoded. If the user has not run `huggingface-cli login`, the pipeline crashes with an authentication error. There is no graceful fallback or helper to guide the user.
- **Impact:** First-time users hit an opaque auth error.
- **Fix:** Added `check_hf_auth()` helper that verifies the token is available and prints a helpful message if not.

## High

### #4: `segmentation_model` from `config.yaml` is ignored
- **File:** `src/diarize.py`, `config.yaml`
- **Status:** Open
- **Description:** `config.yaml` defines `diarization.segmentation_model: "pyannote/segmentation-3.0"`, but `Diarizer._load_model()` only reads `diarization.model` and never uses the segmentation model setting.
- **Impact:** Users cannot override the segmentation model via config.
- **Fix:** Pass the segmentation model identifier to the pyannote pipeline if the API supports it, or document that it is not configurable.

### #5: Stereo audio collapsed to mono without channel separation option
- **File:** `src/preprocess.py`
- **Status:** Open
- **Description:** `convert_to_mono()` always averages both channels. For true stereo recordings with one speaker per channel, this mixes speakers and reduces diarization accuracy.
- **Impact:** Lower transcription quality for stereo call recordings.
- **Fix:** When `metadata.has_stereo_separation` is `True`, split channels into separate mono files and process them independently, or at least preserve channel identity for downstream diarization.

### #6: `database.py`, `spell_check.py`, `vocabulary.py` not wired into pipeline
- **File:** `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-28)
- **Description:** These modules exist but are never imported or called by the orchestration script.
- **Impact:** Features like job tracking, spell checking, and custom vocabulary are unavailable.
- **Fix:** Added optional integration points in `run_pipeline.py` with `--use-database`, `--spell-check`, and `--vocabulary-file` CLI flags.

## Medium

### #7: No test coverage
- **Status:** Resolved (2026-05-28) — partial
- **Description:** There is no `tests/` directory and no CI pipeline.
- **Impact:** Regressions are not caught; refactoring is risky.
- **Fix:** Added `tests/` directory with 31 unit tests for `analyze.py`, `preprocess.py`, `compare.py`, and `diarize.py`. CI pipeline (GitHub Actions) still needed.

### #11: `ThreadPoolExecutor` does not parallelize CPU-bound work
- **File:** `scripts/run_pipeline.py`
- **Status:** Open
- **Description:** Batch processing uses `ThreadPoolExecutor`, but transcription and diarization are CPU-bound tasks. Python's GIL prevents true parallelism with threads, so multiple workers do not speed up processing and may even cause memory contention.
- **Impact:** Batch processing is not faster than single-file; may cause OOM with multiple large models in memory.
- **Fix:** Switch to `ProcessPoolExecutor` for CPU-bound stages, or document that `--workers` should be kept at 1 for CPU-only inference.

### #12: `analyze.py` loads full whisperx model just for language detection
- **File:** `src/analyze.py`
- **Status:** Resolved (2026-05-28) — v0.1.2
- **Description:** `detect_language()` loaded `whisperx.load_model("tiny")` (~39 MB) for every audio file. This was wasteful for a metadata extraction step.
- **Fix:** Reimplemented using `faster_whisper.WhisperModel("tiny")` with module-level cache (`_language_model`) so the model is loaded once and reused. Processes only the first 30 seconds of audio for speed. No standalone `whisper` dependency needed.

### #13: Device auto-detection (replaces #10)
- **File:** `src/transcribe.py`, `src/diarize.py`
- **Status:** Resolved (2026-05-28)
- **Description:** Both modules previously hardcoded `device="cpu"` and `compute_type="int8"`. No auto-detection of `mps` (Apple Silicon), `cuda` (NVIDIA), or config override.
- **Korreksjon:** CTranslate2 (motoren under faster-whisper/WhisperX) **støtter ikke Apple Metal/MPS** — `device="mps"` gir `ValueError: unsupported device mps`. På Mac er `cpu` eneste alternativ for transkripsjon.
- **Impact:** Users with GPU or Apple Silicon get no hardware acceleration for transcription. Diarization (PyTorch) can use MPS.
- **Fix:**
  - `transcribe.py`: auto-detects `cuda` only (CTranslate2 does not support MPS); falls back to `cpu`
  - `diarize.py`: auto-detects `cuda` and `mps` (PyTorch); falls back to `cpu`
  - For betydelig hastighetsøkning på Mac: vurder whisper.cpp+CoreML eller MLX — men dette er en egen motor, ikke et device-flagg.

### #14: Confidence-flagging for review prioritization
- **File:** `src/confidence.py` (new), `src/transcribe.py`, `src/compare.py`
- **Status:** Open — design complete, stub created
- **Description:** Pipeline outputs transcripts but provides no signal about which segments are most likely to contain errors. Manual review is therefore uniform rather than prioritized.
- **Signals available:**
  1. **WhisperX alignment score** (`word["score"]`) — acoustic "text vs audio" confidence from wav2vec2 forced alignment
  2. **faster-whisper decoder signals:** `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, `word.probability`
  3. **Cross-model disagreement** — already computed in `compare.py`
  4. **Acoustic features** from `analyze.py`: SNR, VAD overlap (simultaneous speech)
- **Approach:**
  - Phase A (immediate): Extract all signals, normalize to [0,1], compute unweighted priority score. Rank segments by priority for review.
  - Phase B (future): Use ground-truth fasit to fit a logistic regression model mapping signals → P(error). This calibrates priority into a true probability.
- **Honest limitation:** Confidence-flagging catches "model knew it was uncertain" errors but misses "confidently wrong" errors — especially plausible substitutions of names and numbers. These get high decoder confidence because they are linguistically plausible. Therefore: confidence is a supplement, not a replacement. Proper nouns and numbers should be reviewed regardless of score.
- **Fix:** Create `src/confidence.py` that extracts signals from transcription output, computes priority scores, and exports a prioritized review list. Wire into `run_pipeline.py` as optional step.

### #8: `editor.py` is a placeholder
- **File:** `src/editor.py`
- **Status:** Open
- **Description:** The module only prints manual editing instructions. There is no actual web-based or integrated editor.
- **Impact:** Users must use external tools for manual review.
- **Fix:** Build a minimal web UI (e.g., FastAPI + wavesurfer.js) or integrate with an existing subtitle editor.

### #9: `compare.py` alignment is simplistic
- **File:** `src/compare.py`
- **Status:** Open
- **Description:** Segment alignment is based purely on time overlap (>50%). There is no word-level WER diff or robust alignment algorithm.
- **Impact:** False positives/negatives in disagreement detection.
- **Fix:** Implement word-level alignment (e.g., using `jiwer` or a custom DTW-based approach) and compute actual WER between segments.

### #10: No Apple Silicon / GPU acceleration
- **Status:** Resolved (2026-05-28) — merged into #13
- **Description:** Overlapped with #13 (hardcoded device). CTranslate2 does not support MPS; see #13 for canonical status.

## Resolved

_None yet._
