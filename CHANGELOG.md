# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with pyproject.toml and uv configuration
- ROADMAP.md for feature tracking
- CHANGELOG.md for release notes
- **Step 1 - Audio Analysis** (`src/analyze.py`):
  - FFprobe integration for audio metadata extraction
  - Bandwidth detection (narrowband/wideband/fullband)
  - Stereo separation detection for real stereo audio
  - Language detection using Whisper
  - Voice Activity Detection (VAD) using Silero model
  - Loudness and dynamic range calculation (ITU-R BS.1770-4)
  - Metadata serialization to JSON format
- **Step 2 - Preprocessing** (`src/preprocess.py`):
  - Adaptive resampling to 16 kHz mono
  - High-pass filter (80 Hz) for rumble removal
  - Conditional low-pass filter (8 kHz) for narrowband audio only
  - ITU-R BS.1770-4 loudness normalization
  - Optional noise reduction (denoising) with configurable strength
  - WAV output for next stages
- **Step 3 - Diarization** (`src/diarize.py`):
  - PyAnnote speaker-diarization-3.1 integration
  - Flexible speaker count configuration (auto-detect, min/max, exact)
  - Speaker timeline generation
  - JSON output with speaker segments
- **Steps 3 & 4 - Transcription** (`src/transcribe.py`):
  - WhisperX integration with Norwegian-optimized models
  - NB-Whisper (verbatim and main variants)
  - OpenAI Whisper multilingual support
  - Word-level timestamp generation
  - Speaker diarization alignment
  - Multiple output formats: SRT, WebVTT, JSON
- **Step 5 - Model Comparison** (`src/compare.py`):
  - Segment-level alignment between transcriptions
  - Similarity scoring with text normalization
  - Low-confidence detection
  - Disagreement flagging
  - Priority-based sorting (high/medium/low)
  - Human-readable comparison reports
- **Step 6 - Manual Editor** (`src/editor.py`):
  - SRT export for external editors (Subtitle Edit for Mac)
  - Editing workflow instructions
  - Placeholder for future web-based UI
- **Main Orchestration** (`scripts/run_pipeline.py`):
  - Single-file and batch processing modes
  - Parallel worker pool (configurable)
  - Step-by-step or full pipeline execution
  - Comprehensive CLI interface
  - JSON logging with file output
  - Pipeline summary reporting
- **Database Module** (`src/database.py`):
  - SQLite-based transcription job tracking
  - Performance metrics recording (per-stage timing)
  - Correction history and analysis
  - Job statistics and summaries
  - JSON export for results
- **Spell-Checking Module** (`src/spell_check.py`):
  - Norwegian spell-checking with SymSpell integration
  - Number transcription error detection
  - Proper noun identification
  - Transformer-based model support (placeholder)
  - Configurable spell-checking pipeline
- **Vocabulary Module** (`src/vocabulary.py`):
  - Custom vocabulary loading from JSON
  - Initial prompt generation for Whisper vocabulary injection
  - Domain-specific vocabulary (medical, legal, technical, finance)
  - Common Norwegian proper nouns database
  - Similarity-based correction suggestions
- Utility modules (`src/utils.py`, `src/config.py`):
  - JSON and text logging formatters
  - Configuration management with YAML support
  - Audio processing helper functions
  - File utilities (save/load JSON, ensure directories)

### Changed

### Fixed

### Removed

## [0.1.4] - 2026-05-29

### Added
- **Confidence-flagging wired into pipeline** (`scripts/run_pipeline.py`)
  - Automatic confidence extraction after every transcription
  - Exports `*_review_list.txt` with top 20 flagged segments for manual review
  - Integrates decoder signals, acoustic features, and metadata
- **Hard-rules for high-risk content** (`src/confidence.py`)
  - Numbers: always flagged regardless of decoder confidence (WhisperX alignment is weak for numeric tokens)
  - Proper nouns: capitalized words (not sentence-start) always flagged — catches "confidently wrong" name substitutions
- **Corrupted file filtering** (`scripts/run_pipeline.py`)
  - `_find_audio_files()` skips files smaller than 1KB (likely corrupted)
  - Logs count of skipped files
- **Language detection confidence threshold** (`src/analyze.py`)
  - Falls back to "no" (Norwegian) when `language_probability < 0.5`
  - Prevents false "et" (Estonian) detections on Norwegian speech
- **Decoder signals passed through pipeline** (`src/transcribe.py`, `scripts/run_pipeline.py`)
  - `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature` now flow from transcription to confidence scoring

### Changed
- **Loudness target** (`config.yaml`): `-16 LUFS` → `-20 LUFS` for high-dynamic-range recordings
- **Beam size** (`config.yaml`): `5` → `10` for improved verbatim accuracy (quality/speed tradeoff)
- **Default workers** (`scripts/run_pipeline.py`): `4` → `1` for CPU-only inference (avoids GIL contention and OOM)
- **SRT speaker format** (`src/transcribe.py`): Speaker label now inline (`SPEAKER_00: text`) instead of separate line

### Fixed
- **Loudness clipping** (`src/preprocess.py`): Gain is now capped so peak never exceeds 1.0 (pre-clipping instead of post-clipping)
- **Language detection false positives** (`src/analyze.py`): Tiny model no longer trusted for low-confidence detections
- **Confidence priority all-zero** (`src/transcribe.py`, `scripts/run_pipeline.py`): Decoder signals were not passed from transcription to confidence extractor

## [0.1.3] - 2026-05-28

### Added
- `src/confidence.py` — confidence-flagging module for review prioritization
  - Extracts WhisperX alignment scores (acoustic confidence)
  - Extracts faster-whisper decoder signals: `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, `word.probability`
  - Integrates cross-model disagreement from `compare.py`
  - Integrates acoustic features from `analyze.py` (SNR, VAD overlap)
  - Computes unweighted priority scores for segment ranking
  - Exports prioritized review list for manual correction
- `tests/test_confidence.py` — 7 unit tests for confidence module

## [0.1.2] - 2026-05-28

### Added
- `scripts/evaluate.py` — WER/CER evaluation harness using `jiwer` for ground-truth comparison
- `jiwer` dependency in `pyproject.toml`
- Device auto-detection in `src/diarize.py` (`cuda` / `mps` / `cpu`) via `_auto_detect_device()`
- Module-level cache for language detection model in `src/analyze.py` to avoid reloading

### Fixed
- `src/analyze.py` language detection now uses `faster_whisper` directly (not `whisperx.load_model`), with 30-second audio clip for speed
- `src/transcribe.py` device auto-detection: only `cuda` is supported for GPU; `cpu` fallback documented (CTranslate2 does not support MPS)
- `src/diarize.py` device auto-detection: supports `cuda`, `mps` (PyTorch), and `cpu`

## [0.1.1] - 2026-05-28

### Added
- `ISSUES.md` tracking known bugs and feature gaps from audit
- Unit tests for core modules (`tests/test_analyze.py`, `tests/test_preprocess.py`, `tests/test_compare.py`, `tests/test_diarize.py`)
- `pytest` configuration in `pyproject.toml` with `pythonpath = ["."]`
- Hugging Face auth helper (`check_hf_auth`) in `src/diarize.py` with graceful error messages
- CLI flags `--use-database`, `--vocabulary-file`, `--spell-check` in `scripts/run_pipeline.py`
- Integration of `database.py`, `spell_check.py`, and `vocabulary.py` into pipeline orchestration

### Fixed
- `src/transcribe.py` now passes `beam_size`, `vad_filter`, `condition_on_previous_text`, and `initial_prompt` into the WhisperX transcription call
- `src/analyze.py` language detection now uses `whisperx` instead of standalone `whisper`, removing dependency mismatch
- `src/compare.py` bug: `TranscriptionComparer.__init__` referenced `config` instead of `self.config`
- `pyproject.toml` invalid `[tool.uv] python-version` field removed; `ffmpeg-python` version constraint relaxed to `>=0.2.0`

### Changed
- `README.md` rewritten to reflect actual proof-of-concept status and audit findings
- `ROADMAP.md` updated with accurate implementation status and near-term priorities

## [0.1.0] - 2026-05-28

### Added
- Project scaffolding and infrastructure
- README with detailed pipeline documentation
- MIT License
