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
- [~] `src/database.py` implemented but not integrated into pipeline
- [~] `src/spell_check.py` prototype exists but not integrated
- [~] `src/vocabulary.py` prototype exists but not integrated
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

- `transcribe.py` ignores several transcription config parameters in current WhisperX invocation.
- `analyze.py` language detection depends on `whisper` while `pyproject.toml` lists only `whisperx`.
- `diarize.py` requires Hugging Face auth token and does not use `segmentation_model` from config.
- `preprocess.py` collapses stereo audio into mono; real-channel separation is not handled optimally.
- `database.py`, `spell_check.py`, and `vocabulary.py` are present as modules but are not wired into the pipeline.
- There is no test coverage or CI yet.

## Near-term priorities

1. Fix transcription config usage in `src/transcribe.py`
2. Make `src/analyze.py` language detection resilient without requiring `whisper` as a separate dependency
3. Add Hugging Face token handling and validate pyannote auth flow
4. Integrate `database.py` for job/log tracking and `spell_check.py` for optional review support
5. Add unit tests for `analyze.py`, `preprocess.py`, `diarize.py`, and `compare.py`

## Future ideas

- Fine-tune a Norwegian ASR model on domain-specific vocabulary
- Add REST API / local server wrapper around pipeline
- Provide a web-based review editor with waveform and speaker labels
- Containerize with Docker for reproducible deployments
- Add support for Swedish / Danish / Finnish in addition to Norwegian
