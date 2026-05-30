# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.17] - 2026-05-30

### Added
- **Hard-rule tests for `confidence.py`** — 11 new tests covering digit detection, capitalized OOV detection, repetition, English words, short segments, all-caps tokens, incomplete endings, and lowercase start detection.
- **Unit tests for `spell_check.py`** — 14 tests covering NorwegianSpellChecker initialization, word checking (empty, short, acronyms), text checking, correction, number detection, proper noun detection, and check_transcription convenience function.
- **K5 fix: `--spell-check` CLI flag now overrides config** — previously the CLI flag set `spell_check=True` in the pipeline, but `check_transcription()` read `enabled: false` from `config.yaml` and returned immediately. Now the CLI flag forces `spell_config["enabled"] = True` so the flag actually works.

### Changed
- **#5 stereo verification** — analyzed first 20 testdata files for stereo separation. All 20 are mono (1 channel). No stereo recordings found in the working set. Closing #5 as verified — no code changes needed.

## [0.1.16] - 2026-05-30

### Added
- **`--num-speakers` CLI flag** — convenience flag that sets both `min_speakers` and `max_speakers` to the same value. Useful for telephone calls with 2 speakers. Resolved in `main()` before constructing pipeline kwargs.
- **Unit tests for `normalize.py`** — 7 test classes covering stuttering removal, punctuation restoration, capitalization, dialect flagging, English word flagging, short segment detection, repetition detection, missing space fixing, empty input, batch segment processing, report export, proper noun loading, and constant verification.
- **Unit tests for `vocabulary.py`** — 5 test classes covering VocabularyManager (add, load, save, prompt generation, token limits, domain filtering, corrections), CommonNorwegianVocabulary (domain/dialect vocabulary, create_manager), load_vocabulary (default, with dialect, with domain, custom file, empty fallback), and count_tokens.

## [0.1.15] - 2026-05-30

### Fixed
- **#31 — Editor Step 6 SRT filename mismatch** — Step 6 now reuses the actual output path from `transcribe_audio()` (`primary_output`) instead of reconstructing the filename, which was missing the `_preprocessed` infix.
- **#32 — compare.py config key mismatch** — `TranscriptionComparer` now reads `min_agreement_score` from the `comparison` config block and `low_confidence_threshold` from the `transcription` block, matching `config.yaml` structure.
- **#34 — pyproject.toml version drift** — bumped from `0.1.7` to `0.1.14` to match `CHANGELOG.md`.
- **#37 — `--diarize` flag cleanup** — made diarization opt-in (`default=False`). Removed redundant `--no-diarize` flag. `--diarize` now explicitly enables it.

### Changed
- **#35 — Normalization opt-in** — Norwegian text normalization is now opt-in via `--normalize` CLI flag (default off). Raw verbatim output is preserved as `*_raw.srt` when normalization is enabled. This protects the verbatim model output from heuristic punctuation/capitalization errors.
- **#33 — Model caching across files** — added module-level model cache in `transcribe.py` (`_model_cache` for WhisperX, `_align_model_cache` for wav2vec2). Models are loaded once per run and reused across files in batch mode, eliminating the dominant batch runtime cost.
- **#36 — Privacy: real names removed from source** — `NORWEGIAN_PROPER_NOUNS` in `normalize.py` now contains only place names and public entities. Added `load_proper_nouns()` function that loads personal names from a local gitignored data file (`data/proper_nouns.json`). Added `data/` to `.gitignore`.

## [0.1.14] - 2026-05-30

### Added
- **Empirical confidence baseline from 10-file stratified test run** — ran full pipeline on stratified sample (0.0MB–63.6MB, total 107MB) from 410 testdata files. Mean confidence: 0.447. 100% segment flag rate. All alignment scores null (wav2vec2 returns 0 word-level scores). See ROADMAP.md for full results.
- **Milestone-based roadmap toward 2% WER** — structured M0–M5 milestones with confidence targets, WER targets, effort estimates, and key dependencies. M0 (baseline) marked complete. M1 (fasit creation + calibration) is the immediate next step.

### Changed
- **ROADMAP.md** — replaced single-file test findings with comprehensive 10-file empirical data. Added "Milestones toward 98% confidence / 2% WER" section with calibrated targets based on real measurements. Updated near-term priorities to reference milestone structure.

## [0.1.13] - 2026-05-30

### Added
- **Word-level forced alignment fallback** (`src/transcribe.py`) — when `FasterWhisperPipeline` lacks an `align()` method (nb-whisper-large-verbatim model), the code now falls back to `whisperx.load_align_model()` + `whisperx.align()` with the Norwegian wav2vec2 model (`NbAiLab/nb-wav2vec2-1b-bokmaal-v2`). Alignment scores are merged into original segments, preserving decoder signals alongside acoustic confidence scores. Resolves ISSUES.md #30.

### Changed
- **ROADMAP.md** — elevated dialect recognition to priority feature (Phase 8). Restructured with clear tiers: implemented items, immediate next steps (dialect confidence scoring, preserve-dialect flag, expanded vocabulary, auto-detection), medium-term (multi-dialect support, dialect-specific LM), and long-term/research (corpus collection, fine-tuning). Updated near-term priorities to list dialect recognition as #1.

### Added
- **Word-level forced alignment fallback** (`src/transcribe.py`) — when `FasterWhisperPipeline` lacks an `align()` method (nb-whisper-large-verbatim model), the code now falls back to `whisperx.load_align_model()` + `whisperx.align()` with the Norwegian wav2vec2 model (`NbAiLab/nb-wav2vec2-1b-bokmaal-v2`). Alignment scores are merged into original segments, preserving decoder signals alongside acoustic confidence scores. Resolves ISSUES.md #30.

## [0.1.12] - 2026-05-29

### Changed
- **Confidence hard-rules strengthened** (`src/confidence.py`) — digit tokens and capitalized OOV tokens now get a much stronger priority boost proportional to their count (up to 0.9 for numbers, up to 0.85 for proper nouns). This ensures "confidently wrong" errors (where Whisper is certain but wrong) always appear at the top of the review list, regardless of acoustic confidence scores.
- **New hard-rules added** (`src/confidence.py`) — all-caps tokens (acronyms/abbreviations like NRK, TV2) flagged with up to 0.7 boost; single-letter word artifacts flagged with 0.4 boost.

## [0.1.11] - 2026-05-29

### Added
- **Dialect-adaptive vocabulary** (`src/vocabulary.py`) — new `DIALECT_VOCABULARY` constant with 30+ Northern Norwegian dialect words across 6 categories (pronouns, negation, question words, adverbs, verbs, expressions). `load_vocabulary()` accepts `dialect="northern_norwegian"` parameter to inject dialect words into Whisper's `initial_prompt`. `CommonNorwegianVocabulary.get_dialect_vocabulary()` and `create_manager(dialect=...)` added.
- **`--dialect` CLI flag** (`scripts/run_pipeline.py`) — new `--dialect northern_norwegian` argument passes dialect region through to vocabulary loading. Wired into `process_single_file()` and `pipeline_kwargs`.

### Changed
- **ROADMAP.md** — Phase 8 "Dialect-adaptive vocabulary" marked as implemented (`[x]`).

## [0.1.10] - 2026-05-29

### Added
- **Dialect recognition roadmap** (`ROADMAP.md`) — new Phase 8 with 6 dialect-related items: dialect-adaptive vocabulary, dialect-specific language model, multi-dialect support, dialect confidence scoring, and dialect-preserving output. Current dialect flagging in `normalize.py` marked as partial implementation.

## [0.1.9] - 2026-05-29

### Added
- **Northern Norwegian dialect awareness** (`src/normalize.py`) — new `NORWEGIAN_DIALECT_MAP` with 30+ dialect words (æ, ikkje, ka, kor, mæ, dæ, dokker, etc.). Dialect words are flagged with `[dialect_word]` type for informational purposes but NOT auto-corrected — dialect is valid Norwegian.
- **Northern Norwegian question word detection** (`src/normalize.py`) — dialect question words ("ka", "kæ", "kor", "korsn", "koffer") now trigger `?` at segment end alongside standard question words.
- **Northern Norwegian place names** (`src/normalize.py`) — expanded `NORWEGIAN_PROPER_NOUNS` with 50+ place names from Nordland, Troms, and Finnmark.

### Changed
- **Module docstring** (`src/normalize.py`) — updated to document Northern Norwegian dialect features and the module's approach to dialect handling (flag but don't correct).
- **README.md** — updated project description to mention Northern Norwegian dialect focus.
- **AGENTS.md** — added dialect information to project identity section.

## [0.1.8] - 2026-05-29

### Added
- **Punctuation restoration** (`src/normalize.py`) — rule-based insertion of periods, commas, and question marks in Norwegian conversational speech. Uses filler words ("ja", "nei", "da") and clause markers ("så", "men", "for") to determine sentence boundaries. Question words ("hæ", "hva", "hvorfor", etc.) trigger `?` instead of `.` at segment end.
- **Sentence capitalization** (`src/normalize.py`) — first word of each segment and words following `.`, `!`, or `?` are now capitalized.
- **Stuttering removal** (`src/normalize.py`) — consecutive duplicate words (e.g., "jeg jeg vil" → "jeg vil") are automatically removed as a normalization step.
- **`--min-speakers` / `--max-speakers` CLI flags** (`scripts/run_pipeline.py`) — new optional arguments to constrain speaker count during diarization. Passed through to `diarize_audio()` which already supported these parameters.

### Changed
- **`normalize_norwegian_text()`** (`src/normalize.py`) — completely rewritten with three new sub-functions (`_fix_stuttering`, `_restore_punctuation`, `_capitalize_sentence`) that run before the existing flagging logic. The module now auto-corrects punctuation, capitalization, and stuttering rather than just flagging them.
- **`process_single_file()`** (`scripts/run_pipeline.py`) — added `min_speakers` and `max_speakers` parameters, forwarded to `diarize_audio()`.

## [0.1.7] - 2026-05-29

### Added
- **Explicit dependencies** (`pyproject.toml`) — pinned `symspellpy>=6.7.0` and `soundfile>=0.12.0` that were previously implicit transitive dependencies (ISSUES.md #24 / AUDIT.md M1).
- **HF token validation** (`src/diarize.py`) — `check_hf_auth()` now calls `huggingface_hub.whoami()` to verify token validity, not just existence. Logs clear error if token is invalid (ISSUES.md #26 / AUDIT.md M4).
- **Audio data caching** (`src/analyze.py`, `src/preprocess.py`) — `analyze_audio()` now loads audio once with `mono=False` and stores it in `AudioMetadata.audio_data` (ephemeral, excluded from JSON). `preprocess_audio()` reuses it instead of reloading from disk, eliminating the double-load (ISSUES.md #22 / AUDIT.md H2).
- **Accurate vocabulary token counting** (`src/vocabulary.py`) — `generate_initial_prompt()` now uses `transformers.AutoTokenizer` from `openai/whisper-tiny` for actual token counting instead of a naive "2 tokens per word" estimate. Default `max_tokens` raised to 150 (still well under Whisper's 224-token hard limit). Warning logged if limit exceeded (ISSUES.md #23 / AUDIT.md H4).
- **Removed `pydub` dependency** (`pyproject.toml`, `src/preprocess.py`) — `pydub` was unused (no code called `load_audio_pydub()`). Removing it eliminates the `audioop` deprecation warning and unlocks Python 3.13 compatibility. `requires-python` upper bound `<3.13` removed (ISSUES.md #25 / AUDIT.md M2).

### Changed
- **README.md** — updated "Løst" section with all resolved items through v0.1.6; fixed batch example to use `--workers 1` instead of `--workers 4`.
- **AGENTS.md** — updated §6 Current Reality with new open issues (#22–#26) and resolved issues (#4, #5, #9, #21). Later updated to mark #22 and #23 as resolved.
- **ROADMAP.md** — "Resolved" section now includes all resolved issues through #26.

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
