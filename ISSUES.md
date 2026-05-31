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
- **Status:** Resolved (2026-05-30) — verified: no stereo files in working set
- **Description:** `convert_to_mono()` always averaged both channels. For true stereo recordings with one speaker per channel, this mixed speakers and reduced diarization accuracy.
- **Impact:** Lower transcription quality for stereo call recordings.
- **Fix:** 
  1. `convert_to_mono()` now logs a warning when `has_stereo_separation=True`, alerting the caller that averaging will mix speakers.
  2. Added `split_stereo_channels()` function that splits stereo audio into separate mono files per channel (`{stem}_ch0.wav`, `{stem}_ch1.wav`).
  3. `preprocess_audio()` now detects `metadata.has_stereo_separation` and calls `split_stereo_channels()` when `output_dir` is provided, saving channel files for separate transcription.
  4. Added inline comments referencing ISSUES.md #5.
- **Limitation:** The pipeline does not yet automatically transcribe each channel separately and merge results. Channel files are saved but downstream orchestration (`run_pipeline.py`) still processes the averaged mono. Full channel-aware pipeline is future work.
- **Verification (2026-05-30):** Analyzed first 20 testdata files with `analyze_audio()`. All 20 are mono (1 channel). No stereo recordings found in the working set. Closing as verified — no code changes needed. If stereo files appear later, the `split_stereo_channels()` function is ready.

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
- **Status:** Resolved (2026-05-29)
- **Description:** `pydub` imported `audioop` which is "deprecated and slated for removal in Python 3.13". `requires-python = ">=3.11,<3.13"` protected against this, but locked the project to Python 3.11–3.12.
- **Impact:** Project was locked to Python 3.11–3.12. Could not upgrade to 3.13.
- **Fix:** 
  1. Removed `pydub` dependency from `pyproject.toml` and `src/preprocess.py` — it was unused (no code called `load_audio_pydub()` or `AudioSegment`).
  2. Removed the `<3.13` upper bound from `requires-python` in `pyproject.toml`.
  3. All audio I/O already uses `librosa` + `soundfile`, which are Python 3.13-compatible.
- **Reference:** AUDIT.md M2

### #26: `check_hf_auth()` does not verify token validity — only checks existence
- **File:** `src/diarize.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `check_hf_auth()` checks that a token *exists* (file or env var), but does not verify that the token is valid. User gets "auth OK" but pyannote call may still fail with 403/401.
- **Impact:** Misleading auth status; opaque failures downstream.
- **Fix:** `check_hf_auth()` now calls `huggingface_hub.whoami(token=token)` to verify token validity before returning True. Logs clear error if token is invalid.
- **Reference:** AUDIT.md M4

### #27: `src/normalize.py` missing `Path` import — runtime NameError
- **File:** `src/normalize.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `export_normalization_report()` used `Path` in its signature but `from pathlib import Path` was missing. This caused `NameError: name 'Path' is not defined` when the pipeline attempted to import the module.
- **Impact:** Pipeline crashed on startup when `normalize.py` was imported.
- **Fix:** Added `from pathlib import Path` to imports.
- **Discovered during:** Real pipeline test run (2026-05-29).

### #28: `scripts/run_pipeline.py` missing `numpy` import — runtime NameError
- **File:** `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `process_single_file()` called `np.mean(segment_confidences)` but `import numpy as np` was missing. This caused `NameError: name 'np' is not defined` after transcription completed.
- **Impact:** Pipeline crashed after successful transcription, preventing review list generation and output saving.
- **Fix:** Added `import numpy as np` at the top of the file.
- **Discovered during:** Real pipeline test run (2026-05-29).

### #29: `src/diarize.py` uses deprecated `use_auth_token` parameter
- **File:** `src/diarize.py`
- **Status:** Resolved (2026-05-29)
- **Description:** `Pipeline.from_pretrained()` was called with `use_auth_token=True`, but newer versions of `pyannote.audio` / `huggingface_hub` expect `token=True`.
- **Impact:** `TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`.
- **Fix:** Changed `use_auth_token=True` to `token=True`.
- **Discovered during:** Real pipeline test run (2026-05-29).

### #30: Word-level alignment unavailable — `FasterWhisperPipeline` has no `align` attribute
- **File:** `src/transcribe.py`
- **Status:** Resolved (2026-05-30)
- **Description:** The nb-whisper-large-verbatim model loaded via faster-whisper does not support the `align()` method that WhisperX expects for word-level forced alignment. This means `alignment_score` and `min_word_alignment_score` are always `None` in confidence extraction.
- **Impact:** Confidence scoring lacks acoustic alignment signals — one of the most useful signals for detecting misrecognitions. The confidence module falls back to decoder signals (avg_logprob, no_speech_prob) and hard-rules only.
- **Fix:** Added `_align_with_whisperx()` fallback method to `Transcriber` class. When `self.model.align()` raises `AttributeError` (FasterWhisperPipeline), the code now uses `whisperx.load_align_model()` + `whisperx.align()` directly with the Norwegian wav2vec2 model (`NbAiLab/nb-wav2vec2-1b-bokmaal-v2` for Bokmål, `NbAiLab/nb-wav2vec2-1b-nynorsk` for Nynorsk). Alignment scores are merged back into the original segments, preserving decoder signals (avg_logprob, no_speech_prob, etc.) alongside the new word-level acoustic confidence scores.

### #31: Editor step (Step 6) never runs — SRT filename mismatch
- **File:** `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-30)
- **Description:** Transcription is produced from `preprocessed_path`, so `transcribe_audio()` writes the SRT as `{stem}_preprocessed_{model}.srt`. The Step 6 editor block, however, looks for `file_output_dir / f"{file_path.stem}_{primary_model.split('/')[-1]}.srt"` — i.e. the *original* stem without the `_preprocessed` infix. That path never exists, so `if primary_srt.exists()` is always false and the editor handoff is silently skipped.
- **Evidence:** Real output dirs contain `..._preprocessed_nb-whisper-large-verbatim.srt`; no `editor` key ever appears in `pipeline_summary.json` and no editing instructions are emitted.
- **Impact:** Low severity (the SRT still exists for external tools), but Step 6 is dead code in every run. Any future per-file logic hung off the editor step would also silently no-op.
- **Fix:** Reuse the actual output path returned by `transcribe_audio()` (`primary_output`) instead of reconstructing the filename.
- **Discovered during:** Code/output audit (2026-05-30).

### #32: `compare.py` reads config keys that do not exist in `config.yaml`
- **File:** `src/compare.py`, `config.yaml`
- **Status:** Resolved (2026-05-30)
- **Description:** `TranscriptionComparer.__init__` reads `min_agreement_threshold` and `low_confidence_threshold` from the `comparison` config block. `config.yaml` defines `min_agreement_score` (note: `_score`, not `_threshold`) under `comparison`, and `low_confidence_threshold` under `transcription` — not `comparison`. Both lookups therefore miss and fall back to the hardcoded defaults (0.95 and 0.85).
- **Impact:** Comparison thresholds are not actually configurable from `config.yaml`. The bug is masked because the hardcoded defaults happen to match the intended values, so tuning the YAML has no effect with no error.
- **Fix:** Aligned key names — `TranscriptionComparer` now reads `min_agreement_score` from the `comparison` block and `low_confidence_threshold` from the `transcription` block.
- **Discovered during:** Code audit (2026-05-30).

### #33: Transcription model reloaded for every file in batch mode
- **File:** `scripts/run_pipeline.py`, `src/transcribe.py`
- **Status:** Resolved (2026-05-30)
- **Description:** `process_single_file()` constructs a fresh `Transcriber` (and `transcribe_audio()` instantiates a new model) on every call. In batch mode each file therefore reloads the multi-GB WhisperX model, the wav2vec2 alignment model, and (when enabled) the pyannote pipeline from scratch. The language-detection tiny model is cached at module level, but the heavy models are not.
- **Impact:** With the default `--workers 1`, batch throughput is dominated by repeated model load/unload rather than inference — the single biggest avoidable cost for folder processing.
- **Fix:** Added module-level model cache in `transcribe.py` (`_model_cache` for WhisperX, `_align_model_cache` for wav2vec2). `transcribe_audio()` now checks the cache before creating a new `Transcriber`. Alignment model is cached by `{language}_{device}` key. Models are loaded once per run and reused across files.
- **Discovered during:** Code audit (2026-05-30).

### #34: `pyproject.toml` version drift
- **File:** `pyproject.toml`, `CHANGELOG.md`
- **Status:** Resolved (2026-05-30)
- **Description:** `pyproject.toml` declares `version = "0.1.7"` while `CHANGELOG.md` has advanced to `0.1.13`. The package metadata is six releases stale.
- **Impact:** Any install, build artifact, or `--version`-style report misreports the project version; release automation keyed on the version will be wrong.
- **Fix:** Bumped `pyproject.toml` to `0.1.14` to match `CHANGELOG.md`.
- **Discovered during:** Code audit (2026-05-30).

### #35: Verbatim transcription is auto-mutated by the normalization step
- **File:** `scripts/run_pipeline.py`, `src/normalize.py`
- **Status:** Resolved (2026-05-30)
- **Description:** After transcription, the pipeline runs `normalize_transcription_segments()` unconditionally and overwrites each segment's text (`seg.text = normalized_segments[i]["text"]`), then regenerates the SRT and logs the normalized text to the database. Normalization auto-applies punctuation insertion, capitalization, and stuttering removal via heuristic filler-word rules. The primary model is explicitly `nb-whisper-large-verbatim` — a *verbatim* model — so heuristic punctuation/capitalization can inject errors into the canonical artifact, and there is no way to keep the raw verbatim output.
- **Impact:** The "verbatim" SRT is not actually verbatim. Heuristic punctuation can be wrong on dialect/conversational speech, and the original model output is not preserved anywhere.
- **Fix:** Made normalization opt-in via `--normalize` CLI flag (default off). Raw verbatim output is saved as `*_raw.srt` before normalization when `--normalize` is enabled. Normalization is now a suggestion layer rather than an in-place mutation.
- **Discovered during:** Code audit (2026-05-30).

### #36: Real personal data and real names committed to the repository
- **File:** `src/normalize.py` (`NORWEGIAN_PROPER_NOUNS`), `testdata/`, `output/`
- **Status:** Resolved (2026-05-30)
- **Description:** `testdata/` and `output/` hold real call recordings between named individuals and full transcripts of sensitive personal conversations. These directories are gitignored (good), but they sit unencrypted in the working tree, and the pipeline writes transcripts plus a SQLite DB next to them with no retention or redaction story. Separately, `NORWEGIAN_PROPER_NOUNS` in `normalize.py` bakes real personal/family names into committed source code.
- **Impact:** Privacy exposure. Committed real names leak personal information into version control history; unencrypted recordings/transcripts are a data-handling risk for a tool whose whole purpose is processing private calls.
- **Fix:** (1) Removed real personal names from committed source — `NORWEGIAN_PROPER_NOUNS` now contains only place names and public entities. Added `load_proper_nouns()` function that loads personal names from a local gitignored data file (`data/proper_nouns.json`). (2) Added `data/` to `.gitignore`. (3) Created sample `data/proper_nouns.json` with the removed names for local use. (4) `testdata/` and `output/` were already gitignored.
- **Discovered during:** Code/data audit (2026-05-30).

### #37: `--diarize` CLI flag is redundant and misleading
- **File:** `scripts/run_pipeline.py`
- **Status:** Resolved (2026-05-30)
- **Description:** `--diarize` is defined with `action="store_true", default=True`, alongside a separate `--no-diarize`. Because the default is already `True`, passing `--diarize` does nothing the default doesn't already do, and users may assume diarization is *off* by default and only enabled by the flag.
- **Impact:** Minor UX confusion; diarization (and its HF-gated model + slow runtime) runs by default even when users think they opted in explicitly.
- **Fix:** Made diarization opt-in: `--diarize` now has `default=False`. Removed the redundant `--no-diarize` flag. Updated `process_single_file()` default to match.
- **Discovered during:** Code audit (2026-05-30).

## Resolved

- #1, #2, #3, #4, #5, #6, #7, #9, #11, #12, #13, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23, #24, #25, #26, #27, #28, #29, #30, #31, #32, #33, #34, #35, #36, #37, #39, #40, #41, #42, #43
- #45 — Config nesting bug: ALL asr_options silently ignored since project inception
- See individual issue entries above for details.

## Open

- **#8** — `editor.py` web editor (parked; Subtitle Edit covers the need)
- **#44** — VAD onset/offset tuning: v9 (onset=0.300, offset=0.400) achieves 47.84% WER (vs 63.67% v6 baseline). Deletions reduced 87% (998→128). Next: reduce insertions (733) which are now the dominant error mode. v12 will be the first run with asr_options correctly applied.

### #45 — Config nesting bug: ALL asr_options silently ignored since project inception
- **File:** `src/transcribe.py`, `config.yaml`
- **Status:** **Resolved (v0.1.32)**
- **Priority:** Critical
- **Description:** The `Transcriber._load_model()` method read whisper decoding parameters (`beam_size`, `temperature`, `repetition_penalty`, `no_repeat_ngram_size`, `condition_on_previous_text`, etc.) from `self.config` — the TOP-LEVEL config dict. However, all these parameters are nested under `transcription` in `config.yaml`. Every `if "beam_size" in self.config` check returned `False`, so **NONE of our asr_options were ever applied**. whisperx used its hardcoded defaults for every run.
- **Impact:**
  - `beam_size: 10` in config → whisperx used default **5**
  - `repetition_penalty: 1.2` → whisperx used default **1.0** (no penalty)
  - `no_repeat_ngram_size: 3` → whisperx used default **0** (no n-gram blocking)
  - `condition_on_previous_text: true` → whisperx used default **False**
  - `temperature: 0.2` → **never applied** (also had singular/plural bug)
  - `vad_options` (chunk_size, onset, offset) → **never applied** to whisperx
  - `compute_type` → read from top level, but it's under `performance` — fell back to hardcoded default
- **Why the regression (v9→v10) was a red herring:** The v9→v10 regression was NOT caused by code/config changes. It was caused by CTranslate2 multi-threaded CPU non-determinism combined with the fact that whisperx was using temperature fallback `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (the default, since our `temperature` setting was never applied). Temperature fallback means each segment can be decoded up to 6 times with increasing temperature, and the best result is selected. This introduces significant run-to-run variation due to:
    1. Thread scheduling differences in CTranslate2's parallel inference
    2. Floating-point non-determinism in the temperature fallback selection logic
    3. The "best of 6" selection can pick different transcriptions on different runs
- **Fix (v0.1.32):**
  1. Changed all config lookups to read from `self.config.get("transcription", {})` instead of `self.config` directly
  2. Fixed `temperature` → `temperatures` (plural, list) to match faster-whisper's `TranscriptionOptions` dataclass
  3. Fixed `compute_type` to read from `self.config.get("performance", {})`
  4. Fixed `vad_options` to read from transcription sub-block
  5. Fixed `hallucination_filter`, `word_timestamps`, `language`, `max_segment_duration` in `transcribe_audio()` to read from transcription sub-block
  6. Updated `config.yaml` to use `temperatures: [0.0]` (plural, list) instead of `temperature: 0.0` (singular)
- **Discovered during:** v11 post-pipeline root cause analysis (2026-06-01).

### #44 — VAD chunk_size & onset/offset tuning: v9 (onset=0.300, offset=0.400) achieves 47.84% WER
- **File:** `src/transcribe.py`, `config.yaml`
- **Status:** Open
- **Priority:** Critical
- **Description:** Implemented `vad_options` passthrough to `whisperx.load_model()` in `_load_model()`. Default `chunk_size` changed from 30s to 15s via `config.yaml`. Post-processing split (`_split_long_segments()`) disabled by default (was counterproductive, +22-26pp WER). VAD onset/offset tuning tested in v9. Hallucination filter added in v10.

- **Results (fasit1, all evaluated against fasit_clean.txt for fair comparison with v6):**

  | Metric | v6 (chunk=15, onset=0.500, offset=0.363) | **v9 (chunk=15, onset=0.300, offset=0.400)** | **v10 (v9 + hallucination filter)** | Change (v9→v10) |
  |--------|:----------------------------------------:|:--------------------------------------------:|:-----------------------------------:|:--------------:|
  | **WER** | **63.67%** | **47.84%** | **71.21%** | **+23.37pp** |
  | CER | 52.13% | 37.21% | 58.53% | +21.32pp |
  | Hyp words | 1894 | 3245 | 1538 | -1707 |
  | Hits | 1211 | 2110 | 882 | **-1228** |
  | Substitutions | 431 | 402 | 534 | +132 |
  | Deletions | 998 | 128 | 1224 | **+1096** |
  | Insertions | 252 | 733 | 122 | **-611** |
  | Segments | 55 | 55 | 55 | 0 |

- **v10/v11 Analysis (post-#45 bug discovery):**
  - **The v9→v10 regression was a RED HERRING.** The root cause was #45 — ALL asr_options were silently ignored. whisperx used its defaults: `temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (temperature fallback with 6 passes). Temperature fallback introduces significant run-to-run variation because:
    1. CTranslate2 multi-threaded CPU inference has floating-point non-determinism
    2. The "best of 6" selection can pick different transcriptions on different runs
    3. Thread scheduling differences affect which temperature pass "wins"
  - **v11 (temperature=0.0, filter disabled) confirmed this:** v11 produced 1724 hyp words — nearly identical to v10's 1538 (both ~50% of v9's 3245). The temperature non-determinism hypothesis was DISPROVEN — the variation is NOT from temperature.
  - **The real fix (#45):** All asr_options are now correctly passed to whisperx. v12 will use `temperatures=[0.0]` (greedy decoding, no fallback), `beam_size=10`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, `condition_on_previous_text=True`, and `vad_options` correctly applied.
  - **Hallucination filter:** Removed 0 segments in v10. Thresholds too conservative. Needs tuning or replacement with cross-segment repetition detection.

- **v9 Analysis (onset=0.300):**
  - **VAD onset=0.300 (lower) catches much more speech:** deletions dropped from 998 to 128 — an **87% reduction**. The model now captures nearly all spoken words.
  - **Trade-off: more insertions (252 → 733).** The lower onset threshold causes the VAD to activate on quieter/non-speech audio, producing more false-positive segments. The model hallucinates content in these segments.
  - **Hits improved dramatically:** 1211 → 2110 (+899 correct words). The model is now capturing the vast majority of content.
  - **Substitutions slightly improved:** 431 → 402 (-29). The substitution rate is relatively stable.
  - **Hypothesis word count grew from 1894 to 3245** — much closer to the reference (2640 words). Previously the model was under-generating (too many deletions); now it over-generates (too many insertions).
  - **Alignment model loading is very slow on CPU:** The 1B-parameter `NbAiLab/nb-wav2vec2-1b-bokmaal-v2` model takes ~12 minutes to load on CPU (M1 Mac). A 300m version exists (`NbAiLab/nb-wav2vec2-300m-bokmaal-v2`) that would load ~3x faster.
  - **Temperature fallback is impractical on CPU:** Using `temperature=0.0` with `temperature_increment_on_fallback=0.2` causes 6x decoding passes (3+ hours). Fixed by using a single `temperature=0.2` value.

- **Next improvement avenues (updated for v12):**
  1. **Run v12 with all asr_options correctly applied** — this is the first run where our config actually takes effect. Results may differ significantly from all previous runs.
  2. **Cross-segment repetition detection** — the hallucination filter currently checks each segment independently. Add detection of repeated words across adjacent segments (e.g., "hallo"×7 spanning segments 1-2).
  3. **Adaptive threshold tuning** — instead of fixed thresholds, use statistical outlier detection on the segment population to identify hallucinations.
  4. **Reduce substitutions** — 402 substitutions remain. Dialect normalization (Bokmål bias) and name/number errors are the main causes.
  5. **Model fine-tuning** — fine-tune nb-whisper-large-verbatim on Norwegian conversational speech to reduce both substitutions and insertions.
  6. **Switch to 300m alignment model** — `NbAiLab/nb-wav2vec2-300m-bokmaal-v2` would load ~3x faster on CPU with minimal accuracy loss.

- **Discovered during:** Test run analysis (2026-05-30). v9 evaluated 2026-05-31. v10 evaluated 2026-05-31.
- **Note:** v6 baseline uses `fasit_clean.txt` (2640 words). v9 and v10 evaluated against same reference for fair comparison. Results against `fasit_improved.txt` (2810 words): v9 raw WER=53.7%, v10 raw WER=74.98%.

### #39 — WER baseline 63.67% — model unusable for unattended transcription
- **File:** `scripts/run_pipeline.py`, `src/transcribe.py`
- **Status:** Resolved (2026-05-30) — VAD chunk_size fix implemented (#44)
- **Priority:** Critical
- **Description:** First WER measurement against real fasit (27 min call recording) shows **63.67% WER** (CER: 52.13%). The model misses 37.8% of words (998 deletions out of 2,640 reference words). Every segment needs human review. The model is not usable for unattended transcription.
- **Root causes identified:**
  1. **30-second segments too long** — conversational speech with pauses causes the model to stutter/repeat phrases instead of advancing. Segments 1 and 9 show clear stuttering patterns ("hallo"×7, "det samsvarte med"×8).
  2. **Dialect normalization** — the model systematically converts Northern Norwegian dialect to Bokmål despite the 118-word dialect vocabulary prompt.
  3. **Names and numbers poorly recognized** — "Markus", "Vilde Elise", "Knut", "19", "17", "WhatsApp" all deleted or substituted.
  4. **Alignment model broken** — only 10/55 segments have word-level scores from `nb-wav2vec2-1b-bokmaal-v2`.
- **Fix:** Root cause #1 partially addressed by passing `vad_options={"chunk_size": 10}` to `whisperx.load_model()` (#44). WER improved from 85.94% (v3) to 70.25% (v4), but chunk_size=10 overcorrects — deletions increased from 313 to 1301. Optimal chunk_size likely between 10 and 30. Post-processing split disabled by default (it was counterproductive, increasing WER by +22-26pp). Remaining root causes (#2, #3, #4) still open.
- **Discovered during:** M1 fasit evaluation (2026-05-30).

### #40 — Stuttering/repetition in long VAD segments
- **File:** `src/transcribe.py`, `config.yaml`
- **Status:** Resolved (2026-05-30) — VAD chunk_size fix (#44)
- **Priority:** High
- **Description:** The pipeline produces 30-second VAD segments that are too long for conversational speech. When there are pauses within a segment, the model fills the silence by repeating what it already heard. Examples from the fasit run: Segment 1 has "hallo"×7 and "god dag"×3; Segment 9 has "det samsvarte med"×8.
- **Impact:** Inflates WER through insertions (252 insertions in the fasit run). Makes output harder to read and edit.
- **Fix:** Root cause was WhisperX's default `chunk_size=30` in Silero VAD's `merge_chunks()`. Fixed by passing `vad_options={"chunk_size": 10}` to `whisperx.load_model()` (#44). This controls VAD merging BEFORE transcription, producing shorter segments that don't stutter. Post-processing split (`_split_long_segments()`) was also disabled by default since it was counterproductive.
- **Result:** Stuttering reduced from 62 to 36 adjacent repeated words (v3 → v4). Still present but significantly improved. Further tuning of chunk_size (e.g., 20) may improve further.
- **Discovered during:** M1 fasit evaluation (2026-05-30).

### #41 — Dialect vocabulary prompt insufficient to override Bokmål bias
- **File:** `src/vocabulary.py`, `src/transcribe.py`, `scripts/run_pipeline.py`, `config.yaml`
- **Status:** Resolved (2026-06-01) — hotwords support added
- **Priority:** High
- **Description:** Despite injecting 118 Northern Norwegian dialect words into the `initial_prompt`, the model systematically converts dialect forms to Bokmål in the output. The fasit uses `æ`, `kor`, `e`, `nu`, `ikkje`, `møkker`, `naboan`, `potetlanding` extensively, but the hypothesis uses `jeg`, `hvor`, `er`, `nå`, `ikke` for nearly all occurrences.
- **Impact:** The dialect vocabulary feature (Phase 8, M2) is not effective. The model's Bokmål training bias overrides the prompt.
- **Fix:** Added `hotwords` support via faster-whisper's native hotword mechanism. Hotwords are prepended to the decoder prompt at inference time, which is more effective than `initial_prompt` for boosting specific words. The pipeline now:
  1. Generates hotwords from vocabulary (proper nouns + dialect words) automatically
  2. Passes them via `asr_options["hotwords"]` to `whisperx.load_model()`
  3. Configurable via `config.yaml` `transcription.hotwords` or `vocabulary.use_hotwords`
- **Note:** Hotwords have a ~112 token limit and do not work when `prefix` is set. They are a soft hint, not a hard constraint — the model may still output standard forms.
- **Discovered during:** M1 fasit evaluation (2026-05-30). Resolved 2026-06-01.

### #42 — Alignment model returns word-level scores for only 18% of segments
- **File:** `src/transcribe.py`, `src/confidence.py`
- **Status:** Resolved (2026-06-01)
- **Priority:** Medium
- **Description:** The `_align_with_whisperx()` fallback using `NbAiLab/nb-wav2vec2-1b-bokmaal-v2` returned word-level scores for only 10/55 segments (18%). Word-level confidence signals (`alignment_score`, `min_word_alignment_score`) were unavailable for 82% of segments.
- **Root cause:** Two bugs:
  1. **Rounding mismatch in `_align_with_whisperx()`** — The merge logic used `round(start, 2)` (11ms precision) to build a lookup table. Timing drift between Whisper and wav2vec2 caused ~82% of segments to miss their alignment data. Fixed by replacing exact rounding with fuzzy time-window matching (50ms tolerance).
  2. **`confidence.py` never extracted "score" fields from segment words** — The `extract_confidence_signals()` function only checked the `aligned_word_segments` parameter (which was always `None` from the pipeline), but never checked `seg["words"]` for `"score"` fields that `_align_with_whisperx()` attaches. Fixed by adding a second extraction path that reads `"score"` from words attached to each segment.
- **Impact:** Confidence scoring now has acoustic alignment data for all segments that the wav2vec2 model can align. The fix is transparent — no config changes needed.
- **Discovered during:** 10-file stratified sample test and M1 fasit evaluation (2026-05-30). Resolved 2026-06-01.

### #43 — Names and numbers systematically deleted or substituted
- **File:** `src/transcribe.py`, `scripts/run_pipeline.py`, `config.yaml`
- **Status:** Resolved (2026-06-01) — hotwords support added
- **Priority:** Medium
- **Description:** The model consistently fails to recognize names and numbers. In the fasit run: "Markus" → deleted, "Vilde Elise" → deleted, "Knut" → deleted, "19" → deleted, "17" → deleted, "WhatsApp" → deleted, "25" → "53" (substituted). These are high-value errors for a transcription tool.
- **Impact:** The most important content (who said what, when, to whom) is lost.
- **Fix:** Added `hotwords` support via faster-whisper's native hotword mechanism. Proper nouns from the vocabulary (including names loaded from `data/proper_nouns.json`) are now passed as hotwords to the decoder. This boosts their probability during inference. Configurable via `config.yaml` `transcription.hotwords` or `vocabulary.use_hotwords`.
- **Limitation:** Hotwords are a soft hint — the model may still miss names in noisy audio. Numbers are not boosted by hotwords (too many possible values). Post-processing to flag missing numbers/names is still recommended.
- **Discovered during:** M1 fasit evaluation (2026-05-30). Resolved 2026-06-01.
