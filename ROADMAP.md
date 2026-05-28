# Audio-Transcriber Roadmap

## Current implementation status

This roadmap reflects the existing implementation, identified gaps from the audit, and the next deliverables.

### Phase 1: Project Infrastructure
- [x] `pyproject.toml` with `uv` configuration
- [x] `config.yaml` with pipeline settings
- [x] Repository structure created (`src/`, `scripts/`)
- [x] Hugging Face authentication helper for pyannote and token management

### Phase 2: Core Modules
- [x] `src/analyze.py` — audio analysis and metadata extraction
- [x] `src/preprocess.py` — adaptive audio preprocessing
- [x] `src/diarize.py` — speaker diarization wrapper
- [x] `src/transcribe.py` — WhisperX transcription and exporter
- [x] `src/compare.py` — model comparison and prioritized review output
- [~] `src/editor.py` — export SRT / manual instructions (placeholder for web editor)

### Phase 3: Orchestration & CLI
- [x] `scripts/run_pipeline.py` — pipeline orchestration
- [x] CLI argument parsing (`--input`, `--output-dir`, `--step`, `--workers`, etc.)
- [x] Batch folder processing with worker pool
- [x] Single-file processing support
- [~] Step-level execution is present, but some step interdependencies and metadata reuse need hardening

### Phase 4: Configuration & Utilities
- [x] `src/utils.py` — logging, file helpers, JSON utilities
- [x] `src/config.py` — YAML configuration loader
- [x] Logging setup with console/file handlers
- [x] `src/database.py` integrated into pipeline via `--use-database`
- [x] `src/spell_check.py` integrated into pipeline via `--spell-check`
- [x] `src/vocabulary.py` integrated into pipeline via `--vocabulary-file`
- [ ] VAD configuration and actual model selection should be hardened

### Phase 5: Feature gap closing
- [ ] Word-level confidence-based review filtering
- [x] Norwegian spell-checking integration in pipeline (basic integration via `--spell-check`)
- [x] Automatic `initial_prompt` / vocabulary injection for Whisper (via `--vocabulary-file`)
- [ ] Proper stereo handling for one-speaker-per-channel audio
- [ ] More robust alignment and diffing beyond simple overlap / SequenceMatcher
- [x] Full audit logging / job tracking using SQLite or JSON logs (basic integration via `--use-database`)
- [ ] True editor UI with waveform and speaker-aware review

### Phase 6: Optimization
- [ ] Apple Silicon acceleration with CoreML / MLX support
- [ ] CUDA/GPU support for Linux/Windows
- [ ] Model caching and memory optimization for batch jobs
- [ ] Performance profiling and resource usage monitoring

### Phase 7: Quality & documentation
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Add CI pipeline
- [ ] Add troubleshooting guide
- [ ] Add example workflows to README
- [ ] Add API documentation / developer reference

## Key audit findings to address next

### Resolved (2026-05-28)
- ✅ `transcribe.py` now passes `beam_size`, `vad_filter`, `condition_on_previous_text`, and `initial_prompt` into WhisperX.
- ✅ `analyze.py` language detection now uses `whisperx` instead of standalone `whisper`.
- ✅ `diarize.py` has `check_hf_auth()` with graceful error messages.
- ✅ `database.py`, `spell_check.py`, and `vocabulary.py` are wired into pipeline via CLI flags.
- ✅ Unit tests added for `analyze.py`, `preprocess.py`, `compare.py`, and `diarize.py` (31 tests passing).

### Remaining
- `config.yaml` `segmentation_model` is ignored by `diarize.py`.
- `preprocess.py` collapses stereo audio into mono; real-channel separation is not handled optimally.
- `ThreadPoolExecutor` in batch mode does not provide true parallelism for CPU-bound tasks due to Python GIL.
- `analyze.py` loads a full `whisperx` tiny model (~39 MB) just for language detection.
- `device="cpu"` and `compute_type="int8"` are hardcoded in `transcribe.py` and `diarize.py`.
- `compare.py` alignment is simplistic (time overlap + SequenceMatcher), not word-level WER.
- `editor.py` remains a placeholder (SRT export only, no web UI).
- No CI pipeline or integration tests.

## Near-term priorities

1. Add device auto-detection (`mps` / `cuda`) in `transcribe.py` and `diarize.py`
2. Implement proper stereo channel separation in `preprocess.py`
3. Replace `ThreadPoolExecutor` with `ProcessPoolExecutor` or document single-worker recommendation
4. Add lightweight language detection in `analyze.py` (avoid loading full whisperx model)
5. Improve `compare.py` alignment with word-level WER using `jiwer`
6. Add GitHub Actions CI workflow for automated testing

## Future ideas

- Fine-tune a Norwegian ASR model on domain-specific vocabulary
- Add REST API / local server wrapper around pipeline
- Provide a web-based review editor with waveform and speaker labels
- Containerize with Docker for reproducible deployments
- Add support for Swedish / Danish / Finnish in addition to Norwegian
