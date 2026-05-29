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
- **Status:** Resolved (2026-05-29)
- **Description:** `config.yaml` previously defined `diarization.segmentation_model: "pyannote/segmentation-3.0"`, but `Diarizer._load_model()` only read `diarization.model` and never used the segmentation model setting.
- **Impact:** Users could not override the segmentation model via config.
- **Fix:** Removed `segmentation_model` from `config.yaml` and added a comment explaining that pyannote/speaker-diarization-3.1 bundles its own segmentation model internally. The field was misleading — pyannote 3.1 does not expose segmentation model configuration. Added inline comment in `diarize.py` referencing ISSUES.md #4.
- **Rationale:** pyannote 3.1 uses an internal segmentation model that is not user-configurable. The config field was dead code.

### #5: Stereo audio collapsed to mono without channel separation option
- **File:** `src/preprocess.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `convert_to_mono()` always averaged both channels. For true stereo recordings with one speaker per channel, this mixed speakers and reduced diarization accuracy.
- **Impact:** Lower transcription quality for stereo call recordings.
- **Fix:** 
  1. `convert_to_mono()` now logs a warning when `has_stereo_separation=True`, alerting the caller that averaging will mix speakers.
  2. Added `split_stereo_channels()` function that splits stereo audio into separate mono files per channel (`{stem}_ch0.wav`, `{stem}_ch1.wav`).
  3. `preprocess_audio()` now detects `metadata.has_stereo_separation` and calls `split_stereo_channels()` when `output_dir` is provided, saving channel files for separate transcription.
  4. Added inline comments referencing ISSUES.md #5.
- **Limitation:** The pipeline does not yet automatically transcribe each channel separately and merge results. Channel files are saved but downstream orchestration (`run_pipeline.py`) still processes the averaged mono. Full channel-aware pipeline is future work.
- **Verification:** All test files are mono; stereo separation path is code-reviewed but not exercised on real data.

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
- **Status:** Resolved (2026-05-29)
- **Description:** Batch processing used `ThreadPoolExecutor` with default 4 workers, but transcription and diarization are CPU-bound. Python's GIL prevents true parallelism with threads.
- **Impact:** Batch processing was not faster than single-file; risk of OOM with multiple large models in memory.
- **Fix:** Changed default `--workers` from 4 to 1. Documented that >1 only makes sense on CUDA with sufficient VRAM.

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
- **File:** `src/confidence.py`, `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-29)
- **Description:** Pipeline outputs transcripts but provides no signal about which segments are most likely to contain errors. Manual review is therefore uniform rather than prioritized.
- **Signals available:**
  1. **WhisperX alignment score** (`word["score"]`) — acoustic "text vs audio" confidence from wav2vec2 forced alignment
  2. **faster-whisper decoder signals:** `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, `word.probability`
  3. **Cross-model disagreement** — already computed in `compare.py`
  4. **Acoustic features** from `analyze.py`: SNR, VAD overlap (simultaneous speech)
- **Fix:** `src/confidence.py` extracts signals from transcription output, computes priority scores, and exports a prioritized review list. Wired into `run_pipeline.py` as automatic step after transcription. Exports `*_review_list.txt` with top 20 flagged segments.
- **Honest limitation:** Confidence-flagging catches "model knew it was uncertain" errors but misses "confidently wrong" errors — especially plausible substitutions of names and numbers. These get high decoder confidence because they are linguistically plausible. Therefore: confidence is a supplement, not a replacement. Proper nouns and numbers should be reviewed regardless of score.

### #8: `editor.py` is a placeholder
- **File:** `src/editor.py`
- **Status:** Open
- **Description:** The module only prints manual editing instructions. There is no actual web-based or integrated editor.
- **Impact:** Users must use external tools for manual review.
- **Fix:** Build a minimal web UI (e.g., FastAPI + wavesurfer.js) or integrate with an existing subtitle editor.

### #9: `compare.py` alignment is simplistic
- **File:** `src/compare.py`
- **Status:** Resolved (2026-05-29)
- **Description:** Segment alignment was based purely on time overlap (>50%). Text similarity used `difflib.SequenceMatcher` on raw characters, which is less accurate than word-level WER for transcription comparison.
- **Impact:** False positives/negatives in disagreement detection.
- **Fix:** `calculate_similarity()` now uses `jiwer.wer()` for word-level similarity when available, falling back to `SequenceMatcher` if jiwer fails. WER gives a more linguistically meaningful similarity score than character-level matching. Time-alignment remains overlap-based (appropriate for segment-level pairing).
- **Note:** Full DTW-based alignment is still future work but low ROI for this use case.

### #15: Language detection returns wrong language for Norwegian
- **File:** `src/analyze.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `detect_language()` using faster-whisper tiny model returned "et" (Estonian) for Norwegian speech with confidence 0.29.
- **Impact:** Metadata reports wrong language. Future multi-language support would be broken.
- **Fix:** Added confidence threshold (0.5). If `info.language_probability < 0.5`, falls back to "no" (Norwegian). Logs warning when fallback triggers.

### #16: Loudness normalization causes clipping
- **File:** `src/preprocess.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `normalize_loudness()` with target -16 LUFS caused peak 0.66 → 1.91 on high-dynamic-range recordings, triggering clipping reduction.
- **Impact:** Digital distortion from clipping can reduce transcription accuracy.
- **Fix:** Two changes: (1) Capped gain so peak never exceeds 1.0 (pre-clipping instead of post-clipping). (2) Changed default `loudness_target_lufs` from -16 to -20 in `config.yaml`.

### #17: Corrupted test files cause batch failures
- **File:** `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-29)
- **Description:** ~400 of 410 test files were corrupted ("moov atom not found", 0 bytes). Batch processing failed on all of them.
- **Impact:** Cannot run batch processing without manual filtering.
- **Fix:** `_find_audio_files()` now skips files smaller than 1KB. Logs count of skipped files.

### #18: SRT speaker labels on separate line break compatibility
- **File:** `src/transcribe.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `_segments_to_srt()` placed speaker label (`SPEAKER_00`) on its own line between timestamp and text. Most SRT parsers treat this as subtitle text.
- **Impact:** External editors (Subtitle Edit, VLC) display speaker labels as visible text.
- **Fix:** Speaker label now inline: `SPEAKER_00: text` on the same line as subtitle content.

### #19: Beam size too low for best accuracy
- **File:** `config.yaml`
- **Status:** Resolved (2026-05-29)
- **Description:** Default `beam_size: 5` is conservative. For verbatim transcription (nb-whisper-large-verbatim), higher beam search improves accuracy at the cost of ~2× slower decoding.
- **Impact:** Suboptimal transcription accuracy, especially for rare words and names.
- **Fix:** Increased `beam_size` from 5 to 10. Best-of remains 5. This is a quality/speed tradeoff appropriate for a verbatim model.

### #20: Confidence priority scores all zero — decoder signals not passed to extractor
- **File:** `src/transcribe.py`, `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `TranscriptionSegment` dataclass did not include `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature` fields. Pipeline passed segments to confidence extractor without these decoder signals, resulting in all priority scores being 0.000.
- **Impact:** Review list was useless — no segments were prioritized; all had equal priority.
- **Fix:** Added decoder signal fields to `TranscriptionSegment` dataclass. Updated pipeline to pass these signals through to `extract_confidence_signals()`. Added hard-rules for numbers and proper nouns as fallback when decoder signals are weak.

### #21: `spell_check.py` has no Norwegian dictionary — feature is non-functional
- **File:** `src/spell_check.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `NorwegianSpellChecker._init_symspell()` created a `SymSpell` object but loaded **no dictionary**. `check_word()` would always return `True, None` because there was no vocabulary to compare against. The `--spell-check` CLI flag gave users false confidence that spelling was being checked.
- **Impact:** `--spell-check` did nothing. Users believed spelling was verified when it was not.
- **Fix:** Added explicit check in `_init_symspell()`: if no dictionary is loaded, set `symspell_available = False` and log a clear warning explaining that spell-checking is disabled without a Norwegian word list. Added inline comment documenting why no dictionary is bundled (licensing restrictions for Norwegian word lists) and how to enable it (download NST/UiB word list and call `load_dictionary()`).
- **Rationale:** Norwegian dictionaries have licensing restrictions. Bundling one is non-trivial. Better to be honest that the feature is disabled than to silently do nothing.

### #22: `preprocess.py` loads audio twice — unnecessary I/O and memory use
- **File:** `src/preprocess.py`, `src/analyze.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `analyze_audio()` ran `librosa.load(..., mono=True)`. Then `preprocess_audio()` ran `librosa.load(..., mono=False)`. The same file was loaded twice.
- **Impact:** Unnecessary I/O and memory use. For batch of long recordings this was noticeable.
- **Fix:** 
  1. `analyze_audio()` now loads audio once with `mono=False` and stores it in `AudioMetadata.audio_data` (an ephemeral field excluded from JSON serialization).
  2. `preprocess_audio()` reuses `metadata.audio_data` when available, falling back to `librosa.load()` only when absent.
  3. `save_metadata()` explicitly strips `audio_data` before JSON serialization to avoid massive metadata files.
- **Reference:** AUDIT.md H2

### #23: `vocabulary.py` token estimate is BPE-naive — risks silent prompt truncation
- **File:** `src/vocabulary.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `generate_initial_prompt()` estimated "2 tokens per word". Whisper uses BPE — common words can be 1 token, rare compound words 3–5 tokens. `max_tokens=100` was hardcoded. The 224-token `initial_prompt` limit could be silently exceeded.
- **Impact:** Vocabulary injection may be truncated without warning, reducing effectiveness.
- **Fix:** 
  1. Added `_get_tokenizer()` helper that lazily loads `transformers.AutoTokenizer` from `openai/whisper-tiny` for accurate token counting.
  2. `count_tokens()` uses the tokenizer when available; falls back to a conservative 1.5 tokens/word estimate if unavailable.
  3. `generate_initial_prompt()` default changed from 100 to 150 tokens (still well under 224), and now checks each item's actual token count before adding it.
  4. Final prompt is checked against the 224-token hard limit; warning logged if exceeded.
- **Reference:** AUDIT.md H4

### #24: Implicit dependencies (`symspellpy`, `soundfile`) not pinned in `pyproject.toml`
- **File:** `pyproject.toml`
- **Status:** Resolved (2026-05-29)
- **Description:** `symspellpy` and `soundfile` are imported in the code but not listed in `dependencies`. They are currently transitive dependencies, but should be explicit.
- **Impact:** Fresh installs may break if transitive dependency versions change.
- **Fix:** Added `symspellpy>=6.7.0` and `soundfile>=0.12.0` to `dependencies` in `pyproject.toml`.
- **Reference:** AUDIT.md M1

### #25: `pydub` uses deprecated `audioop` — Python 3.13 time bomb
- **File:** `pyproject.toml`, `src/preprocess.py`
- **Status:** Open
- **Description:** `pydub` imports `audioop` which is "deprecated and slated for removal in Python 3.13". `requires-python = ">=3.11,<3.13"` protects now, but is a time bomb.
- **Impact:** Project is locked to Python 3.11–3.12. Cannot upgrade to 3.13.
- **Fix:** Monitor pydub updates; consider migrating I/O to `soundfile`+`librosa`.
- **Reference:** AUDIT.md M2

### #26: `check_hf_auth()` does not verify token validity — only checks existence
- **File:** `src/diarize.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `check_hf_auth()` checks that a token *exists* (file or env var), but does not verify that the token is valid. User gets "auth OK" but pyannote call may still fail with 403/401.
- **Impact:** Misleading auth status; opaque failures downstream.
- **Fix:** `check_hf_auth()` now calls `huggingface_hub.whoami(token=token)` to verify token validity before returning True. Logs clear error if token is invalid.
- **Reference:** AUDIT.md M4

## Resolved

- #1, #2, #3, #4, #5, #6, #7, #9, #11, #12, #13, #14, #15, #16, #17, #18, #19, #20, #21
- See individual issue entries above for details.
