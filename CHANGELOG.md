# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-05-29

### Added
- **Explicit dependencies** (`pyproject.toml`) — pinned `symspellpy>=6.7.0` and `soundfile>=0.12.0` that were previously implicit transitive dependencies (ISSUES.md #24 / AUDIT.md M1).
- **HF token validation** (`src/diarize.py`) — `check_hf_auth()` now calls `huggingface_hub.whoami()` to verify token validity, not just existence. Logs clear error if token is invalid (ISSUES.md #26 / AUDIT.md M4).
- **Audio data caching** (`src/analyze.py`, `src/preprocess.py`) — `analyze_audio()` now loads audio once with `mono=False` and stores it in `AudioMetadata.audio_data` (ephemeral, excluded from JSON). `preprocess_audio()` reuses it instead of reloading from disk, eliminating the double-load (ISSUES.md #22 / AUDIT.md H2).
- **Accurate vocabulary token counting** (`src/vocabulary.py`) — `generate_initial_prompt()` now uses `transformers.AutoTokenizer` from `openai/whisper-tiny` for actual token counting instead of a naive "2 tokens per word" estimate. Default `max_tokens` raised to 150 (still well under Whisper's 224-token hard limit). Warning logged if limit exceeded (ISSUES.md #23 / AUDIT.md H4).

### Changed
- **README.md** — updated "Løst" section with all resolved items through v0.1.6; fixed batch example to use `--workers 1` instead of `--workers 4`.
- **AGENTS.md** — updated §6 Current Reality with new open issues (#22–#26) and resolved issues (#4, #5, #9, #21).

### Fixed
- **Documentation drift** (AGENTS.md §7 / AUDIT.md drift list):
  - `README.md` "Gjenstående" no longer lists resolved issues #11, #14
  - `ROADMAP.md` "Remaining" no longer lists resolved issues
  - `ROADMAP.md` "Resolved" now includes all resolved issues through #21
  - Batch example uses `--workers 1` matching default
- **ISSUES.md** — added 5 new tracked issues from AUDIT.md findings: #22 (H2), #23 (H4), #24 (M1), #25 (M2), #26 (M4).

## [0.1.6] - 2026-05-29

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

## [0.1.6] - 2026-05-29

### Added
- **Aggressive Norwegian hard-rules** (`src/confidence.py`) — 20 rules for maximum error detection without ground truth:
  - Repetition: 3+ repeated words or 2+ repeated phrases = hallucination flag
  - English words: 50+ common English words detected in Norwegian text
  - Duration: segments <2s or >60s flagged as segmentation errors
  - Word count: <3 or >50 words flagged
  - Norwegian char patterns: "aa"→"å", "ae"→"æ", "oe"→"ø" substitution flags
  - Formatting: missing spaces after punctuation
  - Unusual characters: symbols, emojis, mixed scripts
  - Excessive fillers: "hæ" ≥3 times, " ja " ≥4 times
  - Incomplete endings: trailing hyphen or ellipsis
  - Lowercase starts: segments not starting with capital letter
- **Enhanced review list export** (`src/confidence.py`)
  - Exports ALL segments (not just top 20) when `review_list_top_n: null`
  - Full signal data exported as JSON with histogram and flag distribution
  - Human-readable report with priority histogram and flag breakdown
- **Norwegian text normalization** (`src/normalize.py`)
  - Fixes missing spaces after punctuation
  - Flags character substitutions (aa/ae/oe)
  - Flags English words with Norwegian suggestions
  - Flags excessive repetition and short segments
  - Exports normalization report per file
  - Regenerates SRT with normalized text
- **Default Norwegian vocabulary** (`data/norwegian_vocabulary.json`)
  - 100+ Norwegian places, names, institutions, companies, political parties
  - Auto-loaded into Whisper `initial_prompt` for better recognition
  - `vocabulary.py` loads default vocab automatically when no custom file provided

### Changed
- **Aggressive config thresholds** (`config.yaml`)
  - `low_confidence_threshold`: 0.85 → 0.70 (flags more segments)
  - `logprob_threshold`: -0.5 → -0.3 (flags more low-confidence)
  - `no_speech_threshold`: 0.5 → 0.3 (flags more hallucination risk)
  - `compression_threshold`: 2.4 → 2.0 (flags more repetition)
  - `review_list_top_n`: 20 → null (exports ALL segments)
  - `review_list_export_json`: true (full signal data)
- **Pipeline integration** (`scripts/run_pipeline.py`)
  - Normalization step runs automatically after transcription
  - SRT regenerated with normalized text
  - Default Norwegian vocabulary auto-loaded for `initial_prompt`

## [0.1.5] - 2026-05-29

### Added
- **Stereo channel splitting** (`src/preprocess.py`)
  - `split_stereo_channels()` splits stereo audio into separate mono files per channel (`{stem}_ch0.wav`, `{stem}_ch1.wav`)
  - `preprocess_audio()` detects `metadata.has_stereo_separation` and saves channel files when `output_dir` is provided
  - `convert_to_mono()` now logs a warning when `has_stereo_separation=True` to alert that averaging mixes speakers
- **Word-level WER similarity** (`src/compare.py`)
  - `calculate_similarity()` now uses `jiwer.wer()` for word-level similarity when available
  - Falls back to `SequenceMatcher` if jiwer fails
  - More linguistically meaningful than character-level matching for transcription comparison

### Changed
- **Config cleanup** (`config.yaml`)
  - Removed misleading `segmentation_model` field from `diarization` section
  - Added comment explaining that pyannote/speaker-diarization-3.1 bundles its own segmentation model
- **Spell-check honesty** (`src/spell_check.py`)
  - `_init_symspell()` now explicitly disables spell-checking when no Norwegian dictionary is loaded
  - Logs clear warning instead of silently doing nothing
  - Added inline comment documenting why no dictionary is bundled (licensing restrictions)

### Fixed
- **Issue #4** (`src/diarize.py`, `config.yaml`): `segmentation_model` config field was dead code — pyannote 3.1 does not expose segmentation model configuration
- **Issue #5** (`src/preprocess.py`): Stereo audio with one speaker per channel was averaged to mono without warning or channel splitting option
- **Issue #9** (`src/compare.py`): Text similarity used character-level `SequenceMatcher` instead of word-level WER
- **Issue #21** (`src/spell_check.py`): Spell-checking was non-functional because no Norwegian dictionary was loaded; now explicitly disabled with clear logging

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
