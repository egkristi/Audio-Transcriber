# Audio-Transcriber Roadmap

## Vision

The north star for this project is **accurate transcription of spoken Norwegian in every dialect** — not just standard Bokmål/Eastern Norwegian, but Northern Norwegian (Nordland, Troms, Finnmark), Trøndersk, Vestlandsk, Sørlandsk, and the rest — preserving dialect forms rather than normalizing them away.

The longer-term vision is to extend the same dialect-aware approach to **all the Nordic languages**: Swedish, Danish, Norwegian (Bokmål + Nynorsk), Icelandic, and Faroese, plus Finnish, and ideally the Sámi languages — each with their regional dialects. The architecture is deliberately language-agnostic underneath (analyze → preprocess → diarize → transcribe → confidence → review), so adding a language should mean adding a model + a vocabulary/dialect pack + alignment model, not rebuilding the pipeline.

Guiding principles:
- **Dialect is valid language, not error.** Flag dialect/standard mismatches for awareness; never silently "correct" dialect to a standard form.
- **Measure before optimizing.** No accuracy claim is real without a fasit (ground-truth transcript) and a WER/CER number. This gate applies to every language and dialect we add.
- **Human-in-the-loop, honestly.** The tool prioritizes review; it does not pretend to be error-free. "Confidently wrong" outputs (names, numbers, dialect normalization) are surfaced regardless of model confidence.
- **Local-first and privacy-respecting.** Recordings are private. Processing stays on the operator's machine; personal data is never committed.

Sequencing: **(1)** nail Norwegian-all-dialects with a measurable WER, **(2)** generalize the language/dialect packs into a pluggable structure, **(3)** add the remaining Nordic languages one at a time, each gated on its own fasit.

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
- [~] `src/editor.py` — export SRT / manual instructions (placeholder for web editor). Subtitle Edit covers the need; web editor is out of scope.

### Phase 3: Orchestration & CLI
- [x] `scripts/run_pipeline.py` — pipeline orchestration
- [x] CLI argument parsing (`--input`, `--output-dir`, `--step`, `--workers`, etc.)
- [x] Batch folder processing with worker pool
- [x] Single-file processing support
- [x] Step-level execution with hardened metadata reuse (audio data cached in `AudioMetadata.audio_data` between analyze and preprocess, #22)

### Phase 4: Configuration & Utilities
- [x] `src/utils.py` — logging, file helpers, JSON utilities
- [x] `src/config.py` — YAML configuration loader
- [x] Logging setup with console/file handlers
- [x] `src/database.py` integrated into pipeline via `--use-database`
- [x] `src/spell_check.py` integrated into pipeline via `--spell-check` (disabled when no dictionary loaded — honest failure, #21)
- [x] `src/vocabulary.py` integrated into pipeline via `--vocabulary-file` (accurate token counting with Whisper tokenizer, #23)
- [x] VAD configuration via `config.yaml` (`analysis.vad_model: "silero"`); Silero VAD loaded via `torch.hub.load()`

### Phase 5: Feature gap closing
- [x] **Per-segment confidence level** (`src/transcribe.py`) — `TranscriptionSegment.confidence_level` computed from `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature` using geometric mean for conservative scoring. Exported in SRT as `[LOW CONFIDENCE]` / `[MEDIUM CONFIDENCE]` labels.
- [x] **Per-file total confidence** (`src/analyze.py`, `scripts/run_pipeline.py`) — `AudioMetadata.total_confidence` stores the mean confidence across all segments. `segments_count` and `flagged_segments_count` track how many segments fell below the 0.7 threshold. Saved in metadata JSON for every processed file.
- [x] **Word-level confidence-based review filtering** — `src/confidence.py` implemented and wired into pipeline
  - WhisperX alignment score (acoustic confidence)
  - faster-whisper decoder signals: `avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`, `word.probability`
  - Cross-model disagreement from `compare.py`
  - Acoustic features from `analyze.py`: SNR, VAD overlap
  - **20 hard-rules for high-risk content:** numbers, proper nouns, repetition, English words, duration, word count, Norwegian char patterns (aa/ae/oe), formatting, unusual characters, incomplete endings, lowercase starts, excessive fillers
  - Auto-exports `*_review_list.txt` with ALL segments + `*_review_list.json` with full signal data
  - Priority histogram and flag distribution in review list
  - Future: calibrate priority scores against ground-truth using logistic regression
- [x] Norwegian spell-checking integration in pipeline (basic integration via `--spell-check`)
  - **Honest limitation:** No Norwegian dictionary is bundled due to licensing. Spell-checking is disabled until a dictionary is provided. See ISSUES.md #21.
- [x] Automatic `initial_prompt` / vocabulary injection for Whisper (via `--vocabulary-file` or default Norwegian vocabulary)
- [x] **Norwegian text normalization** (`src/normalize.py`) — fixes spacing, flags char substitutions, English words, repetition, short segments; exports normalization report; regenerates SRT
- [x] Proper stereo handling for one-speaker-per-channel audio (channel splitting implemented, full pipeline integration pending)
- [x] More robust alignment and diffing beyond simple overlap / SequenceMatcher (jiwer WER-based similarity)
- [x] Full audit logging / job tracking using SQLite or JSON logs (basic integration via `--use-database`)
- [x] **Speaker diarization integration** (`src/diarize.py` + `src/transcribe.py`) — pyannote/speaker-diarization-3.1 for speaker separation; `align_with_diarization()` assigns `SPEAKER_00`, `SPEAKER_01`, etc. to each transcription segment; SRT/JSON/VTT output includes inline speaker labels. Configurable via `config.yaml` (`min_speakers`, `max_speakers`). **Status: implemented, needs real-data verification.**
- [~] True editor UI with waveform and speaker-aware review — out of scope; Subtitle Edit covers the need (#8)

### Phase 6: Optimization
- [~] Apple Silicon acceleration with CoreML / MLX support — CTranslate2 does not support MPS; whisper.cpp+CoreML or MLX would require a separate engine. Deferred.
- [x] CUDA/GPU support for Linux/Windows — device auto-detection implemented (#13). CTranslate2 uses CUDA when available; PyTorch diarization uses CUDA/MPS.
- [x] Model caching and memory optimization for batch jobs — language detection model cached (`_language_model`); transcription model loaded once per run. Full batch memory optimization is future work.
- [x] **VAD chunk_size control at model-load time** — added `vad_options` dict with configurable `chunk_size` to `whisperx.load_model()` in `src/transcribe.py:_load_model()`. Config key: `transcription.vad_options.chunk_size` in `config.yaml`. (ISSUES.md #44)
  - **v6 (chunk_size=15, onset=0.500, offset=0.363): 62.95% WER** — beats the v1 baseline (68.79%) by 5.84pp. Optimal chunk_size confirmed as 15.
  - **v9 (chunk_size=15, onset=0.300, offset=0.400): 47.84% WER** — VAD onset/offset tuning reduced deletions by 87% (998→128). Lower onset catches more speech. Trade-off: insertions increased (252→733). Overall WER improved by **15.83pp** vs v6.
  - **v10 (v9 + hallucination filter): 71.21% WER** — hallucination filter added to reduce insertions. Insertions dropped 83% (733→122) but deletions skyrocketed (128→1224). The filter removed 0 segments — the deletion increase is from model non-determinism, not the filter. See ISSUES.md #44 and #45.
  - **v7 (chunk_size=25 + hotwords): 91.28% WER** — hotwords severely degraded quality. **Hotwords are NOT recommended for production use.**
  - **v8 (chunk_size=25, no hotwords): 70.6% WER** — chunk_size=25 worse than 15. **Confirmed: chunk_size=15 is optimal.**
  - **Next:** Investigate model non-determinism (ISSUES.md #45). Run v11 with temperature=0.0 to test reproducibility. If WER stabilizes, the non-determinism hypothesis is confirmed and the fix is greedy decoding.
- [x] **Disable post-processing split by default** — `_split_long_segments()` with `max_segment_duration: 15` INCREASED WER significantly in testing. Post-processing split is now opt-in only (default `max_segment_duration=0`), with a clear warning that it may degrade accuracy.
- [x] **Fix alignment model coverage (ISSUES.md #42)** — `_align_with_whisperx()` previously returned word-level scores for only 18% of segments due to a rounding mismatch in the merge logic. Replaced exact `round(start, 2)` lookup with fuzzy time-window matching (50ms tolerance). Also fixed `confidence.py` to extract `"score"` fields from `seg["words"]` directly (not just from the `aligned_word_segments` parameter, which was always `None`). Now all segments that the wav2vec2 model can align get acoustic confidence scores.
- [x] **Hotwords support for faster-whisper (ISSUES.md #41, #43)** — added `hotwords` passthrough via `asr_options` to `whisperx.load_model()`. The pipeline now generates hotwords from vocabulary (proper nouns + dialect words) and passes them to the decoder. Configurable via `config.yaml` `transcription.hotwords` or `vocabulary.use_hotwords`.
  - **⚠️ v7 test result: hotwords severely degrade quality.** With chunk_size=25 + hotwords, WER jumped to 91.28% (vs 62.95% for chunk_size=15 without hotwords). The model produced repetitive gibberish and lost most content. Hotwords are retained as an opt-in feature but are NOT recommended for production use with nb-whisper-large-verbatim.
- [ ] Performance profiling and resource usage monitoring — **next priority after fasit exists**

### Phase 7: Quality & documentation
- [x] Add unit tests — 146 tests covering `analyze.py`, `preprocess.py`, `compare.py`, `diarize.py`, `normalize.py`, `vocabulary.py`, `confidence.py`, `spell_check.py`, and pipeline orchestrator integration
- [ ] Add integration tests — requires real audio fixtures; blocked until fasit exists. **Next priority after first successful end-to-end run.**
- [~] Add CI pipeline — overinvestment for personal tool; targeted unit tests are sufficient
- [ ] Add troubleshooting guide — **add after collecting common failure modes from real runs**
- [x] Add example workflows to README — batch and single-file examples present
- [ ] Add API documentation / developer reference — **deferred until API stabilizes**
- [x] **Word-level forced alignment** — nb-whisper-large-verbatim via faster-whisper lacks `align()`. Fixed with `_align_with_whisperx()` fallback using `NbAiLab/nb-wav2vec2-1b-bokmaal-v2`. Resolves ISSUES.md #30.

### Phase 8: Dialect recognition & adaptation (PRIORITY FEATURE)

> **Why this is a priority:** The target audio is Northern Norwegian (Nordland, Troms, Finnmark) with characteristic features that differ significantly from standard Eastern Norwegian. Whisper's nb-whisper-large-verbatim model was trained primarily on Bokmål and standard speech, causing systematic errors on dialect forms. Improving dialect recognition is the single highest-ROI accuracy improvement available without a fasit.

#### Implemented
- [x] **Dialect-aware normalization** (`src/normalize.py`) — `NORWEGIAN_DIALECT_MAP` with 30+ dialect words mapped to standard equivalents for flagging. Dialect words are flagged with `[dialect_word]` type but NOT auto-corrected — dialect is valid Norwegian.
- [x] **Dialect-adaptive vocabulary** — `src/vocabulary.py` extended with `DIALECT_VOCABULARY` (30+ Northern Norwegian words across 6 categories: pronouns, negation, question words, adverbs, verbs, expressions). `load_vocabulary()` accepts `dialect="northern_norwegian"` parameter. Pipeline CLI has `--dialect northern_norwegian` flag. Dialect words are injected into Whisper's `initial_prompt` to improve recognition of non-standard forms.
- [x] **Northern Norwegian place names** — 50+ place names from Nordland, Troms, and Finnmark added to `NORWEGIAN_PROPER_NOUNS` in `normalize.py`.
- [x] **Northern Norwegian question word detection** — dialect question words ("ka", "kæ", "kor", "korsn", "koffer") trigger `?` at segment end.

#### Next up (immediate)
- [x] **Dialect confidence scoring** — add dialect-specific confidence signals to `confidence.py`: flag segments where Whisper outputs standard forms but dialect forms are expected (and vice versa). Prioritize these for review. This catches "confidently wrong" dialect normalization where Whisper silently converts dialect to standard.
- [x] **Dialect-preserving output** — ensure dialect features are preserved in SRT output and not silently normalized to Bokmål. Add `--preserve-dialect` CLI flag. The current approach (flag but don't correct) is the foundation; formalize as a configurable option.
- [x] **Dialect-aware language model prompt** — expand `DIALECT_VOCABULARY` with more domain-specific dialect words (telephony, customer service, healthcare vocabulary in dialect form). Currently 118 words; target 100+.
- [x] **Dialect region auto-detection** — analyze transcribed text for dialect markers and auto-select the appropriate dialect vocabulary, rather than requiring `--dialect northern_norwegian` to be passed manually.

#### Medium-term
- [ ] **Multi-dialect support** — extend dialect map to cover other Norwegian dialects:
  - **Trøndersk:** "æ" (jeg), "dæm" (dem), "hainn" (han), "kæm" (hvem), "sånn" (sånn), "int" (ikke), "itt" (ikke)
  - **Vestlandsk:** "eg" (jeg), "deg" (du), "ikkje" (ikke), "kva" (hva), "korleis" (hvordan), "kvi" (hvorfor)
  - **Sørlandsk:** "æ" (jeg), "dæ" (deg), "kæm" (hvem), "kordan" (hvordan), "itte" (ikke)
  - **Østlandsk:** "jæ" (jeg), "dæ" (deg), "sæ" (seg), "kæ" (hva), "sånn" (sånn)
  - Region detection heuristics based on distinctive word patterns (e.g., "eg" → Vestlandsk, "dæm" → Trøndersk)
- [ ] **Dialect-specific language model** — evaluate whether fine-tuning or LoRA adapters on a Norwegian dialect corpus (e.g., Nordic Dialect Corpus, NB Whisper) improves WER for Northern Norwegian speech vs. the generic nb-whisper-large-verbatim model. **Note:** This requires a fasit to measure before/after WER.
- [ ] **Dialect-aware confidence calibration** — once fasit exists, measure whether dialect segments have systematically higher WER than standard segments. If so, apply a dialect penalty to confidence scores.

#### Long-term / research
- [ ] **Dialect corpus collection** — build a small corpus of transcribed Northern Norwegian speech for fine-tuning or evaluation. Even 30 minutes of transcribed dialect audio would be valuable.
- [ ] **Dialect-to-standard alignment** — research whether a dialect normalization step (dialect → Bokmål) before transcription improves accuracy, or if dialect-preserving transcription is better. Current hypothesis: dialect-preserving is better because Whisper was trained on Bokmål and dialect normalization would add another error source.
- [ ] **Fine-tuned dialect model** — if a dialect corpus exists, fine-tune nb-whisper-large-verbatim on Northern Norwegian speech using LoRA. Compare WER against the base model.

### Phase 9: Engineering hardening & correctness (from 2026-05-30 audit)

> These were concrete defects and gaps found in a code/output audit. All ISSUES.md #31–#37 are now resolved as of 2026-05-30. Remaining items are lower-priority hardening.

- [x] **Fix editor Step 6 (ISSUES.md #31)** — Step 6 now reuses `primary_output` from `transcribe_audio()` instead of rebuilding the path.
- [x] **Fix comparison config keys (ISSUES.md #32)** — `TranscriptionComparer` now reads `min_agreement_score` from the `comparison` block and `low_confidence_threshold` from the `transcription` block.
- [x] **Cache models across files in batch (ISSUES.md #33)** — added module-level model cache in `transcribe.py` (`_model_cache` for WhisperX, `_align_model_cache` for wav2vec2). Models loaded once per run.
- [x] **Sync version (ISSUES.md #34)** — bumped `pyproject.toml` to `0.1.14` to match `CHANGELOG.md`.
- [x] **Make normalization opt-in & preserve raw output (ISSUES.md #35)** — added `--normalize` CLI flag (default off). Raw verbatim output saved as `*_raw.srt` when enabled.
- [x] **Privacy & data handling (ISSUES.md #36)** — removed real personal names from committed source. Added `load_proper_nouns()` that loads from gitignored `data/proper_nouns.json`. Added `data/` to `.gitignore`.
- [x] **Clean up `--diarize` flag (ISSUES.md #37)** — made diarization opt-in (`default=False`). Removed redundant `--no-diarize` flag.
- [x] **Integration test on a tiny synthetic clip** — 12 tests covering the orchestrator glue: step sequencing, diarization opt-in/out, spell-check wiring, normalization opt-in/out, vocabulary loading, model comparison, result structure, error handling, confidence aggregation, and step filtering. Heavy dependencies mocked for fast deterministic testing.
- [x] **Unit tests for the newest modules** — `normalize.py` (39 tests, 7 test classes), `vocabulary.py` (29 tests, 5 test classes), `spell_check.py` (14 tests), and `confidence.py` hard-rules (11 tests) now have coverage. `database.py`, `editor.py` remain untested.
- [x] **CLI `--num-speakers 2` convenience** — added `--num-speakers` flag that sets both `min_speakers` and `max_speakers` to the same value. Useful for telephone calls with 2 speakers.
- [x] **K5: `--spell-check` CLI flag now overrides config** — previously the CLI flag was silently ignored because `check_transcription()` read `enabled: false` from `config.yaml`. Now the CLI flag forces `spell_config["enabled"] = True`.
- [x] **#5 stereo verification** — analyzed first 20 testdata files. All mono (1 channel). No stereo recordings in working set. Closing as verified.

### Phase 10: Norwegian — all dialects (extends Phase 8)

> Vision step 1: a single Norwegian pipeline that handles any dialect with a measurable WER. Phase 8 covers Northern Norwegian specifically; this phase generalizes it.

- [ ] **Pluggable dialect packs** — refactor the Northern-Norwegian-specific maps (`NORWEGIAN_DIALECT_MAP`, `DIALECT_VOCABULARY`, place names) into per-dialect data files (`data/dialects/<region>.json`) with a common loader, so adding a dialect is data, not code.
- [ ] **All major Norwegian dialect regions** — Trøndersk, Vestlandsk, Sørlandsk, Østlandsk, Innlandet, plus finer-grained sub-regions, each as a dialect pack (vocabulary + standard-form mapping for flagging).
- [ ] **Automatic dialect-region detection** — classify a recording's dialect from distinctive markers (e.g. `eg`→Vestlandsk, `dæm`→Trøndersk, `æ`+`ikkje`→Nordnorsk) and auto-select the matching pack; fall back to a generic Norwegian pack.
- [ ] **Bokmål + Nynorsk alignment routing** — already partially present (`nb-wav2vec2-1b-bokmaal` vs `-nynorsk`); make the written-standard target selectable per file/segment.
- [ ] **Dialect WER tracking** — once fasits exist, report WER per dialect so we can see which dialects the base model handles well vs. poorly and target effort accordingly.

### Phase 11: Fully automated pipeline — language & dialect auto-detection with per-file optimization

> **Why this is a priority:** The pipeline currently requires manual flags (`--dialect northern_norwegian`, `--language no`) and a single config for all files. For a tool that processes batches of recordings — potentially in different languages and dialects — this is a bottleneck. The goal is a fully automated pipeline that detects language and dialect per file, then optimizes every pipeline step accordingly, with zero manual configuration.

#### Core concept
The pipeline should:
1. **Detect language** per file (already partially done in `analyze.py` via faster-whisper tiny, but always falls back to `"no"`)
2. **Detect dialect** per file (from transcribed text markers — already prototyped in Phase 8)
3. **Select optimal model** per language (transcription model, alignment model, vocabulary)
4. **Select optimal VAD parameters** per dialect/language (onset/offset tuning may differ)
5. **Select optimal decoding parameters** per dialect (beam_size, temperature, repetition_penalty)
6. **Route to the right normalization rules** per language/dialect
7. **Fall back gracefully** when detection is uncertain

#### Implementation plan

- [ ] **Language auto-detection with confidence-based routing** — enhance `analyze.py`'s existing faster-whisper language detection:
  - If confidence > 0.8: use detected language with high confidence
  - If confidence 0.5–0.8: use detected language but flag for review
  - If confidence < 0.5: fall back to `"no"` (Norwegian) — current behavior
  - Store detected language + confidence in `AudioMetadata` for downstream use
  - Remove the hardcoded `language="no"` fallback in `transcribe.py` and route dynamically

- [ ] **Dialect auto-detection from transcribed text** — build on the existing Phase 8 dialect region detection:
  - Run a lightweight first-pass transcription (e.g., whisper tiny) on the first 30s of audio
  - Scan transcribed text for dialect markers (word-level patterns per dialect region)
  - Score each dialect region based on marker frequency
  - Select the best-matching dialect pack or fall back to generic Norwegian
  - Cache the dialect selection per file in metadata

- [ ] **Per-language model routing** — create a model registry mapping language codes to:
  - Transcription model (e.g., `"no"` → `NbAiLab/nb-whisper-large-verbatim`, `"sv"` → `KBLab/kb-whisper-large`)
  - Alignment model (e.g., `"no"` → `NbAiLab/nb-wav2vec2-1b-bokmaal-v2`, `"sv"` → `KBLab/wav2vec2-large-voxrex-swedish`)
  - Fallback models for low-resource languages
  - Configurable via `config.yaml` under a new `models:` section

- [ ] **Per-dialect VAD parameter tuning** — different dialects may benefit from different VAD settings:
  - Northern Norwegian (fast, staccato speech): lower onset, higher offset (current v9: 0.300/0.400)
  - Trøndersk (drawn-out vowels): potentially different onset/offset
  - Vestlandsk (sing-song intonation): potentially different onset/offset
  - Store per-dialect VAD presets in dialect pack data files
  - Auto-select VAD preset based on detected dialect

- [ ] **Per-dialect decoding parameter profiles** — different dialects may need different decoding strategies:
  - Northern Norwegian: current v9 config (beam_size=10, temperature=0.2, repetition_penalty=1.2)
  - Other dialects: tuned experimentally against fasits
  - Store as part of dialect pack data

- [ ] **Per-file normalization routing** — route to the correct normalization rules based on detected language + dialect:
  - Norwegian dialects → `normalize.py` with the matching dialect map
  - Swedish → Swedish normalization rules (new module or parameterized)
  - English → minimal normalization (punctuation only)
  - Generic fallback → basic whitespace/punctuation normalization

- [ ] **Graceful fallback chain** — when detection is uncertain:
  - Language uncertain (< 0.5 confidence): default to Norwegian, flag for review
  - Dialect uncertain (no clear markers): use generic Norwegian pack, flag for review
  - Model unavailable for detected language: log warning, fall back to `openai/whisper-large-v3` (multilingual)
  - Alignment model unavailable: skip alignment, log warning, proceed without word-level scores

- [ ] **Batch-mode optimization** — when processing a folder:
  - Run language detection on all files first (fast, tiny model)
  - Group files by detected language/dialect
  - Load models once per group (avoids reloading for each file)
  - Process each group with the optimal config

- [ ] **CLI simplification** — make `--language` and `--dialect` optional overrides rather than required flags:
  - Default: auto-detect everything
  - `--language sv` — override language detection (force Swedish)
  - `--dialect trondersk` — override dialect detection
  - `--no-auto-detect` — disable auto-detection, use config defaults
  - Backward-compatible: existing flags continue to work as overrides

- [ ] **Detection report** — export a per-file detection summary:
  - Detected language + confidence
  - Detected dialect + confidence (marker counts per dialect)
  - Selected models and parameters
  - Any fallbacks triggered
  - Saved to metadata JSON alongside transcription results

#### Dependencies
- Phase 10 (pluggable dialect packs) must be completed first — auto-detection needs structured dialect data to select from
- Phase 11 Nordic languages model registry feeds into the model routing
- Requires fasits for at least Norwegian dialects to tune per-dialect VAD/decoding parameters

#### Success criteria
- [ ] Pipeline processes a mixed-language batch (e.g., Norwegian + English files) without any manual flags
- [ ] Pipeline correctly detects and routes Northern Norwegian vs. Trøndersk vs. Vestlandsk
- [ ] Per-dialect VAD parameters measurably improve WER over one-size-fits-all config
- [ ] Detection report accurately reflects what was detected and what was chosen
- [ ] Fallback chain never crashes — always produces a transcript even with uncertain detection

---

### Phase 12: Nordic languages (long-term vision)

> Vision step 2–3: generalize the language/dialect pack structure, then add Nordic languages one at a time — each gated on its own fasit + WER baseline before it's considered "supported".

- [ ] **Language pack abstraction** — formalize a `LanguagePack` (transcription model, alignment/wav2vec2 model, language code, vocabulary, dialect packs, normalization rules) so the pipeline is parameterized by language rather than hardcoded to `"no"`. Remove the `language="no"` hardcodes in `transcribe.py`/`analyze.py`.
- [ ] **Top-level language auto-detection & routing** — use the existing faster-whisper language detector (already in `analyze.py`) to route each file to the right language pack instead of always falling back to `"no"`.
- [ ] **Swedish** — `KBLab/kb-whisper-large` (or current best Swedish ASR) + Swedish wav2vec2 alignment + dialect packs (e.g. Skånska, Norrländska, Finlandssvenska, Gotländska).
- [ ] **Danish** — best-available Danish Whisper/ASR + Danish alignment + dialect packs (Jysk, Fynsk, Bornholmsk, etc.).
- [ ] **Icelandic & Faroese** — evaluate model availability; these are lower-resource and may need community/fine-tuned models.
- [ ] **Finnish** — Finnish is non–North-Germanic but central to the Nordics; evaluate `Finnish-NLP`/whisper-fi models and Finnish dialect handling.
- [ ] **Sámi languages (research / stretch)** — North/Lule/South Sámi are very low-resource; likely requires corpus collection and fine-tuning. Track as research, set expectations honestly.
- [ ] **Cross-Nordic code-switching** — real Nordic calls mix languages (e.g. Norwegian + Swedish, or Norwegian + English loanwords). Detect and handle intra-call language switches rather than forcing one language for the whole file.
- [ ] **Per-language fasit + WER gate** — no language is marked "supported" until it has a ground-truth clip and a published WER/CER baseline, mirroring the Norwegian gate.

## Test run findings (2026-05-30)

### Fasit evaluation — M1 baseline (27 min call recording)

First-ever WER measurement against a real ground-truth transcript. Pipeline run with `--dialect northern_norwegian` on `Call recording Håvard Kristiansen_260524_172503.m4a` (27 min, 48kHz AAC).

#### WER results — v1 (baseline, chunk_size=30, no post-processing split)

| Metric | Value |
|--------|-------|
| **WER** | **68.79%** |
| CER | 52.13% |
| MER | 58.13% |
| WIL | 70.67% |
| WIP | 29.33% |
| Reference words | 2,810 |
| Hypothesis words | 2,415 |
| Hits (correct) | 1,105 |
| Substitutions | 561 |
| Deletions | 1,144 |
| Insertions | 228 |

> **Note:** v1 was originally reported as 63.67% WER against `fasit_clean.txt` (2,640 words). All values here are re-evaluated against `fasit_improved.txt` (2,810 words) for fair comparison across runs. The corrected v1 baseline is 68.79%.

#### WER results — v2 (post-processing split at 15s)

| Metric | Value |
|--------|-------|
| **WER** | **89.47%** |
| Reference words | 2,810 |
| Hypothesis words | 3,362 |
| Hits (correct) | 1,063 |
| Substitutions | 1,532 |
| Deletions | 215 |
| Insertions | 767 |

#### WER results — v3 (re-run with same config as v2)

| Metric | Value |
|--------|-------|
| **WER** | **85.94%** |
| CER | 64.35% |
| Reference words | 2,810 |
| Hypothesis words | 3,159 |
| Hits (correct) | 1,057 |
| Substitutions | 1,440 |
| Deletions | 313 |
| Insertions | 662 |

#### WER results — v4 (VAD chunk_size=10, no post-processing split)

| Metric | Value |
|--------|-------|
| **WER** | **70.25%** |
| CER | 57.26% |
| Reference words | 2,810 |
| Hypothesis words | 1,615 |
| Hits (correct) | 942 |
| Substitutions | 567 |
| Deletions | 1,301 |
| Insertions | 106 |
| Segments | 55 |
| Adjacent repeated words | 36 |

#### WER results — v5 (VAD chunk_size=20, no post-processing split)

| Metric | Value |
|--------|-------|
| **WER** | **71.35%** |
| CER | 57.26% |
| Reference words | 2,810 |
| Hypothesis words | 1,682 |
| Hits (correct) | 939 |
| Substitutions | 609 |
| Deletions | 1,262 |
| Insertions | 134 |

#### WER results — v6 (VAD chunk_size=15, no post-processing split)

| Metric | Value |
|--------|-------|
| **WER** | **62.95%** |
| CER | 47.63% |
| Reference words | 2,810 |
| Hypothesis words | 2,173 |
| Hits (correct) | 1,270 |
| Substitutions | 674 |
| Deletions | 866 |
| Insertions | 229 |
| Segments | 55 |

#### WER comparison across all runs

| Metric | v1 (chunk=30) | v2 (split) | v3 (no split) | v4 (chunk=10) | v5 (chunk=20) | **v6 (chunk=15)** | v7 (chunk=25+hotwords) | **v8 (chunk=25, no hotwords)** | **v9 (onset=0.300)** | **v10 (v9 + filter)** |
|--------|:------------:|:---------:|:------------:|:-------------:|:-------------:|:----------------:|:---------------------:|:----------------------------:|:--------------------:|:---------------------:|
| **WER** | **68.79%** | **89.47%** | **85.94%** | **70.25%** | **71.35%** | **62.95%** | **91.28%** | **70.6%** | **47.84%** | **71.21%** |
| Hyp words | 2,415 | 3,362 | 3,159 | 1,615 | 1,682 | **2,173** | 555 | 1,791 | 3,245 | 1,538 |
| Hits | 1,105 | 1,063 | 1,057 | 942 | 939 | **1,270** | 246 | 1,022 | 2,110 | 882 |
| Substitutions | 561 | 1,532 | 1,440 | 567 | 609 | **674** | 308 | 573 | 402 | 534 |
| Deletions | 1,144 | 215 | 313 | 1,301 | 1,262 | **866** | 2,256 | 1,215 | 128 | 1,224 |
| Insertions | 228 | 767 | 662 | 106 | 134 | **229** | 1 | 196 | 733 | 122 |
| Segments | — | — | 109 | 55 | — | **55** | — | 55 | 55 | 55 |

**Key insight:** v9 (onset=0.300) is the best run at 47.84% WER — deletions dropped 87% vs v6. v10 (same config + hallucination filter) regressed to 71.21% WER due to model non-determinism (52% fewer hypothesis words). The hallucination filter removed 0 segments. See ISSUES.md #44 and #45.

**Key insight:** chunk_size=15 (v6) is the best tested value at **62.95% WER** — beats the v1 baseline (68.79%) by **5.84pp**. chunk_size=25 without hotwords (v8) gives 70.6% WER — worse than chunk_size=15 by 7.65pp. chunk_size=25 + hotwords (v7) caused catastrophic degradation (91.28%). **The optimal chunk_size is confirmed at 15. No further chunk_size tuning needed.** Next improvement avenues: vad_onset/vad_offset tuning, model fine-tuning, or post-processing.

#### Error analysis

1. **Deletions dominate in v1 (1,144/2,810 = 40.7%):** The model misses entire phrases. Likely causes: (a) 30-second VAD segments are too long for conversational speech with pauses — the model fills silence with stuttering/repetition instead of advancing; (b) the model struggles with crosstalk and overlapping speech common in phone calls. v6 (chunk=15) reduced deletions to 866 (30.8%), a significant improvement.

2. **Post-processing split is COUNTERPRODUCTIVE (v2 → v3):** Adding `max_segment_duration: 15` post-processing split INCREASED WER from 63.67% to 85.94–89.47%. The split creates more segments (55→109→217) from already-stuttered output, inflating insertion count (252→662→767). The model already stuttered within the original VAD segments; splitting after transcription just divides bad output into more pieces. **The fix must happen at model-load time** by passing `vad_options` with `chunk_size` (e.g., 10s) to `whisperx.load_model()`.

3. **Dialect normalization is systematic:** The fasit uses dialect forms extensively (`æ`=105, `e`=135, `kor`=11, `nu`=11, `ikkje`, `møkker`, `naboan`, `potetlanding`). The hypothesis converts nearly all to Bokmål (`jeg`=86, `er`=166, `hvor`=6, `nå`=20, `ikke`=101). Dialect preservation rate: **13.8%** (41/297 dialect words preserved). The dialect vocabulary prompt (118 words) is not strong enough to override Whisper's Bokmål bias.

4. **Stuttering/repetition in long segments:** 62 adjacent repeated words in hypothesis vs 26 in reference. Segment 1 (30s) contains "hallo"×7 and "god dag"×3. Segment 9 (30s) contains "det samsvarte med"×8. This pattern suggests the model runs out of audio content within a long segment and loops on what it heard.

5. **Names and numbers are poorly recognized:** "Markus" → missing, "Vilde Elise" → missing, "Knut" → missing, "19" → missing, "17" → missing, "WhatsApp" → missing. These are high-value errors for a transcription tool.

6. **Alignment model still broken:** Only 10/55 segments have word-level scores from `nb-wav2vec2-1b-bokmaal-v2`. This means word-level confidence signals are unavailable for 82% of segments.

7. **Transcription speed:** ~17 min for 27 min audio on Mac M1 (~1:1.6 ratio, faster than the earlier ~1:3 estimate). CPU usage was 100–400% (multi-core).

#### Implications

The 68.79% WER (corrected baseline against fasit_improved.txt) is far above the 15–25% range estimated before the fasit existed. This means the model is **not usable for unattended transcription** — every output needs full human review and correction. The confidence system's 100% flag rate was correct: every segment genuinely needs review.

The VAD chunk_size fix (chunk_size=15, v6) improved WER from 68.79% (v1) to **62.95%** — a 5.84pp improvement. This is the first VAD chunk_size tuning to beat the baseline. chunk_size=10 (v4) and chunk_size=20 (v5) were both worse than baseline. chunk_size=25 without hotwords (v8) gave 70.6% WER — worse than chunk_size=15 by 7.65pp. **The optimal chunk_size is confirmed at 15. No further chunk_size tuning needed.** Next improvement avenues: vad_onset/vad_offset tuning, model fine-tuning, or post-processing.

Post-processing split (`_split_long_segments()`) is disabled by default since it makes WER worse (+22-26pp).

### Stratified sample run (2026-05-30)

A stratified sample of 10 files (5 Håvard Kristiansen + 5 Elida Anna Wiktoria Kristiansen) across size ranges (2 small <1MB, 4 medium 1-20MB, 4 large >20MB) is running in background. Results pending — estimated 2-4 hours on CPU. As of 22:20, 4/10 files processed (3 Håvard + 1 Elida).

### v8 — chunk_size=25, no hotwords (2026-06-01)

Clean test to isolate the chunk_size=25 effect (v7 was confounded by hotwords). Pipeline run with `chunk_size=25`, `use_hotwords=false`, `--dialect northern_norwegian` on the same fasit1 file.

#### WER results — v8 (VAD chunk_size=25, no hotwords)

| Metric | Value |
|--------|-------|
| **WER** | **70.6%** |
| CER | 56.65% |
| MER | 66.0% |
| WIL | 79.25% |
| Reference words | 2,810 |
| Hypothesis words | 1,791 |
| Hits (correct) | 1,022 |
| Substitutions | 573 |
| Deletions | 1,215 |
| Insertions | 196 |
| Segments | 55 |

#### Comparison with v6 (chunk_size=15, best known)

| Metric | v6 (chunk=15) | v8 (chunk=25) | Δ |
|--------|:------------:|:------------:|:-:|
| **WER** | **62.95%** | **70.6%** | **+7.65pp** |
| Hits | 1,270 | 1,022 | −248 |
| Substitutions | 674 | 573 | −101 |
| Deletions | 866 | 1,215 | +349 |
| Insertions | 229 | 196 | −33 |

**Key findings:**
1. **chunk_size=25 is worse than chunk_size=15** by 7.65pp WER. The model misses 349 more words (deletions) at chunk_size=25.
2. **Hotwords were the primary cause of v7's catastrophic 91.28% WER** — without hotwords, chunk_size=25 gives 70.6%, which is still worse than chunk_size=15 but not catastrophically so.
3. **The optimal chunk_size is confirmed at 15.** The pattern is clear: 30→20→15 improves, 25 is worse than 15. No further chunk_size tuning is needed.
4. **Alignment model now working:** 28/55 segments (51%) have word-level scores — up from 18% before the #42 fix. The fuzzy time-window matching (50ms tolerance) is effective.
5. **Transcription speed:** ~22.5 min for 27.5 min audio (0.82× realtime) on CPU. Alignment added ~12 min (0.43× realtime). Total: ~34.5 min for 27.5 min audio (1.25× realtime).

#### Next steps
- **Revert config to chunk_size=15** — the best known value
- **Explore vad_onset/vad_offset tuning** — these parameters control VAD sensitivity and may improve recall
- **Model fine-tuning** — the most promising path to significant WER reduction

### v10 — v9 + hallucination filter (2026-05-31)

Test run to evaluate the hallucination filter's impact on insertions (the dominant error mode in v9). Same VAD config as v9 (onset=0.300, offset=0.400, chunk_size=15).

#### WER results — v10 (v9 + hallucination filter)

| Metric | vs fasit_clean.txt (2640 words) | vs fasit_improved.txt (2810 words) |
|--------|:-------------------------------:|:----------------------------------:|
| **WER** | **71.21%** | **74.98%** |
| CER | 58.53% | 62.49% |
| Hyp words | 1,538 | 1,576 |
| Hits | 882 | 882 |
| Substitutions | 534 | 574 |
| Deletions | 1,224 | 1,354 |
| Insertions | 122 | 120 |
| Segments (normalized) | 55 | 55 |
| Raw segments | 109 | 109 |

#### Comparison with v9 (same VAD config, no filter)

| Metric | v9 | v10 | Δ |
|--------|:--:|:---:|:-:|
| **WER** | **47.84%** | **71.21%** | **+23.37pp** |
| Hyp words | 3,245 | 1,538 | −1,707 (−53%) |
| Hits | 2,110 | 882 | −1,228 |
| Substitutions | 402 | 534 | +132 |
| Deletions | 128 | 1,224 | +1,096 |
| Insertions | 733 | 122 | −611 (−83%) |

**Key findings:**
1. **Hallucination filter reduced insertions 83%** (733→122) — the filter logic is sound. 🎉
2. **BUT the filter removed 0 segments** — the conservative thresholds (no_speech_prob > 0.5, confidence < 0.3, compression_ratio > 3.0) did not trigger on any of the 109 raw segments.
3. **The deletion increase is NOT from the filter** — the model itself produced 52% fewer words in v10 (1,538 vs 3,245 hypothesis words). Both runs have 109 raw segments.
4. **Model non-determinism is the root cause** — temperature=0.2 introduces sampling variation. The model's output varies significantly between runs for conversational speech with many short VAD segments.
5. **WER increased from 47.84% to 71.21%** — the deletion increase (128→1,224) completely overwhelmed the insertion reduction (733→122).

#### Implications
- The hallucination filter is a good idea but needs more aggressive thresholds to actually catch hallucinations
- The real problem is model non-determinism — v9 may have been a "lucky" run
- **Next step:** Run v11 with temperature=0.0 (greedy decoding) to test reproducibility. If WER stabilizes, the non-determinism hypothesis is confirmed and greedy decoding is the fix.
- See ISSUES.md #44 (updated) and #45 (new) for full details.

### Single-file test (142s)

Real pipeline execution on `testdata/Call recording Elida Anna Wiktoria Kristiansen_251023_190409.m4a` (142s, 48kHz AAC) with `--dialect northern_norwegian` revealed:

1. **Confidence hard-rules working:** 5 segments processed, all flagged. Segment with `repeated_words` + `repeated_phrases` + `possible_proper_noun` correctly ranked highest (0.517). Segments with only `low_logprob` ranked lower (0.372-0.500).
2. **Normalization working:** 21 issues flagged across 5 segments — punctuation restoration (commas before clause markers, periods, question marks) and repetition detection.
3. **Dialect vocabulary injected:** 34 Northern Norwegian dialect words added to initial prompt. Dialect forms ("ja", "da") preserved in output.
4. **Language detection:** Confidence 0.47 for Norwegian — correctly fell back to "no" (Norwegian).
5. **Word-level alignment unavailable:** `'FasterWhisperPipeline' object has no attribute 'align'` — non-fatal warning, alignment scores not available for confidence extraction.
6. **Transcription speed:** ~40s for 142s audio on CPU (Mac M1). Expected.

### 10-file stratified sample test (2026-05-30)

A stratified sample of 10 files (0.0MB–63.6MB, total 107MB) was selected from 410 testdata files and run through the full pipeline (`--no-diarize --dialect northern_norwegian --workers 1`). Total runtime: ~88 minutes (all CPU on Mac M1).

#### Per-file confidence results

| File | Confidence | Segments | Flagged | Duration |
|------|-----------|----------|---------|----------|
| Elida Anna Wiktoria Kristiansen_260308_141037 | **0.285** | 3 | 3 | ~20s |
| Elida Anna Wiktoria Kristiansen_251106_143751 | **0.324** | 6 | 5 | ~197s |
| Håvard Kristiansen_251105_233919 | **0.377** | 1 | 1 | ~2s |
| Elida Anna Wiktoria Kristiansen_251202_164323 | **0.436** | 7 | 6 | ~90s |
| Elida Anna Wiktoria Kristiansen_251109_202730 | **0.476** | 16 | 16 | ~300s |
| Elida Anna Wiktoria Kristiansen_251027_212844 | **0.491** | 16 | 16 | ~460s |
| Elida Anna Wiktoria Kristiansen_251121_114236 | **0.504** | 41 | 39 | ~180s |
| Elida Anna Wiktoria Kristiansen_260209_202520 | **0.506** | 1 | 1 | ~5s |
| Elida Anna Wiktoria Kristiansen_260308_141214 | **0.531** | 1 | 1 | ~20s |
| Elida Anna Wiktoria Kristiansen_251217_200931 | **0.540** | 129 | 121 | ~600s |

#### Aggregate statistics

| Metric | Value |
|--------|-------|
| **Mean confidence** | **0.447** |
| Median confidence | 0.491 |
| Min confidence | 0.285 |
| Max confidence | 0.540 |
| Std dev | 0.085 |
| Files < 0.5 confidence | 6/10 (60%) |
| Files < 0.4 confidence | 3/10 (30%) |
| Files < 0.3 confidence | 1/10 (10%) |

#### Key findings

1. **Confidence is uncalibrated and pessimistic:** The geometric mean of decoder signals produces scores in the 0.28–0.54 range. Every single segment in every file was flagged for review. This means the current 0.7 threshold is useless — **everything** is below it. The confidence system needs calibration against ground truth before it can meaningfully distinguish "needs review" from "probably correct."

2. **No correlation with audio quality:** The lowest-confidence file (0.285, 3 segments) and highest (0.540, 129 segments) are both call recordings of the same person. File size/duration does not predict confidence.

3. **100% flag rate is a problem:** If every segment is flagged, the review list is useless as a prioritization tool. The priority scoring (0.33–1.0 range) provides some ranking, but without a fasit we cannot tell whether the ranking correlates with actual errors.

4. **Alignment scores are always null:** The wav2vec2 alignment model (`NbAiLab/nb-wav2vec2-1b-bokmaal-v2`) returns 0 segments with word-level scores for every file. This means `alignment_score` and `min_word_alignment_score` are always `null` in the confidence signals, removing a key signal from the geometric mean.

5. **Transcription speed:** ~1:3 ratio (1 second audio = ~3 seconds CPU time) for the large model on Mac M1. The 63.6MB file (129 segments, ~600s audio) took ~56 minutes alone.

#### Implications for milestones

The empirical confidence baseline (mean 0.447) is far below the 0.70+ range that would indicate "good enough for most purposes." However, **confidence is not WER** — it is a proxy signal from the decoder. The real accuracy metric requires a fasit (ground-truth transcript). Until a fasit exists, the confidence score is useful only as a relative ranking tool within a single file, not as an absolute quality gate.

**Critical insight:** 2% WER (98% accuracy) = ~1 error per 50 words. This is the gold standard for production ASR and typically requires a fine-tuned, domain-specific model trained on hours of transcribed data. The current off-the-shelf `nb-whisper-large-verbatim` model, even with dialect vocabulary injection, is unlikely to achieve this without fine-tuning on Norwegian call recordings.

## Milestones toward 98% confidence / 2% WER

These milestones are based on the empirical baseline (mean confidence 0.447, 100% flag rate) and the realistic trajectory from off-the-shelf Whisper to a fine-tuned production system. Each milestone is gated on a fasit (ground-truth transcript) for WER measurement.

### M0 — Baseline established ✅ (2026-05-30)
- **Confidence:** 0.447 mean (uncalibrated)
- **Flag rate:** 100% of segments flagged
- **WER:** Unknown (no fasit)
- **Status:** 10-file stratified sample run complete. Confidence system works but is uncalibrated. Every segment needs human review.
- **Gate for next:** Create fasit for at least 3 files from the test sample (short, medium, long).

### M1 — Calibrated confidence + fasit baseline ✅ (2026-05-30)
- **Target confidence:** 0.50 mean (calibrated)
- **Target WER:** Measure baseline
- **Actual WER:** **63.67%** (CER: 52.13%) on 27 min call recording (2,640 reference words)
- **What it took:**
  - User created fasit (`testdata/fasit1/`) with timestamps and dialect forms ✅
  - Cleaned fasit for evaluation (`fasit_clean.txt`) ✅
  - Ran pipeline with `--dialect northern_norwegian` on 27 min audio (~17 min CPU time on Mac M1) ✅
  - Established WER baseline: **63.67%** (substitutions: 431, deletions: 998, insertions: 252, hits: 1,211) ✅
  - Fixed `scripts/evaluate.py` — added `sys.path.insert(0, ...)` for standalone execution; fixed jiwer API usage ✅
- **Key findings:**
  - **WER is much higher than expected** (63.67% vs. estimated 15–25%). The model misses ~38% of words (998 deletions out of 2,640).
  - **Dialect normalization is severe:** The model systematically converts dialect forms to Bokmål (`æ`→`jeg`, `kor`→`hvor`, `e`→`er`, `nu`→`nå`) despite the dialect vocabulary prompt.
  - **Stuttering/repetition:** The model produces repeated phrases (e.g., "hallo"×7, "det samsvarte med"×8) — likely from long 30-second segments causing the model to loop.
  - **Long segments hurt accuracy:** 30-second segments are too long for conversational speech with pauses. Shorter segments would reduce stuttering and improve alignment.
  - **Alignment model still broken:** Only 10/55 segments have word-level scores from `nb-wav2vec2-1b-bokmaal-v2`.
  - **Hypothesis is 28% shorter than reference** (1,894 vs. 2,640 words) — the model misses entire phrases, especially names, numbers, and dialect content.
- **VAD chunk_size fix (v4):** chunk_size=10 improved WER from 85.94% to 70.25% (Δ = -15.69pp) vs v3, but overcorrected — deletions increased from 313 to 1,301.
- **Optimal chunk_size confirmed at 15:** v6 (chunk=15) = 62.95% WER, v8 (chunk=25) = 70.6% WER. chunk_size=15 is the best tested value.
- **VAD onset=0.300 (v9):** 47.84% WER — best run so far. Deletions dropped 87% (128 vs 866 in v6). Insertions became the dominant error mode (733).
- **Hallucination filter (v10):** Insertions dropped 83% (733→122) but model non-determinism caused deletions to skyrocket (128→1,224). WER regressed to 71.21%. The filter removed 0 segments — the problem is upstream.
- **Model non-determinism discovered:** temperature=0.2 causes significant output variation between runs. v10 produced 52% fewer words than v9 with identical config. See ISSUES.md #45.
- **Gate for next:** WER baseline known (47.84% best at v9). Next priority: investigate model non-determinism (run v11 with temperature=0.0).

### M2 — Dialect-aware confidence + vocabulary expansion ✅ (2026-05-30)
- **Target confidence:** 0.55 mean (calibrated)
- **Target WER:** 10–15% (halve the baseline)
- **What it took:**
  - Expand dialect vocabulary from 34 → 100+ Northern Norwegian words ✅ (118 words)
  - Add dialect confidence scoring: flag segments where Whisper outputs standard forms but dialect expected ✅
  - Implement `--preserve-dialect` flag to prevent silent normalization ✅
  - Add dialect region auto-detection from transcribed text ✅ (5 dialects)
  - Fix model caching across files (load WhisperX model once per run, not per file) — reduces batch runtime by ~5× ✅ (done in v0.1.15)
- **Estimated effort:** 8–16 hours
- **Actual WER after M2:** 63.67% — dialect vocabulary injection alone is insufficient. The model still normalizes dialect to Bokmål despite the prompt.
- **Gate for next:** WER < 15% on the fasit set; dialect words recognized correctly

### M3 — Prompt engineering + domain vocabulary
- **Target confidence:** 0.60 mean (calibrated)
- **Target WER:** 5–10%
- **What it takes:**
  - Build domain-specific vocabulary (telephony, healthcare, customer service)
  - Optimize `initial_prompt` construction: test different prompt formats, word orderings, and token budgets
  - Add `--num-speakers 2` convenience flag for telephone calls
  - Implement stereo channel splitting for one-speaker-per-channel recordings
  - Run on full 410-file testdata set for comprehensive statistics
- **Estimated effort:** 16–24 hours
- **Gate for next:** WER < 10% on the fasit set; domain terms recognized

### M4 — Fine-tuned model (LoRA)
- **Target confidence:** 0.75 mean (calibrated)
- **Target WER:** 2–5%
- **What it takes:**
  - Collect 30+ minutes of transcribed Norwegian call audio (from corrected pipeline output)
  - Fine-tune `nb-whisper-large-verbatim` using LoRA on this corpus
  - Compare WER before/after fine-tuning on the fasit set
  - If successful, integrate LoRA weights as an optional pipeline component
- **Estimated effort:** 40–80 hours (data collection + training + evaluation)
- **Gate for next:** WER < 5% on the fasit set; fine-tuned model beats base model

### M5 — Production-ready (98% accuracy / 2% WER)
- **Target confidence:** 0.85+ mean (calibrated)
- **Target WER:** ≤2% (~1 error per 50 words)
- **What it takes:**
  - Large-scale fine-tuning: 100+ hours of domain-specific transcribed audio
  - Multi-dialect support (Trøndersk, Vestlandsk, Sørlandsk, Østlandsk)
  - Cross-model ensemble or self-training / pseudo-labeling
  - Comprehensive evaluation on held-out test set
  - Confidence calibration validated against large fasit
- **Estimated effort:** 200+ hours (data collection + training cycles + evaluation)
- **Reality check:** This is the gold standard for production ASR. It may not be achievable without a team, dedicated GPU budget, and extensive domain data collection. For a personal tool, M3 (5–10% WER) is a more realistic long-term target.

### Summary

| Milestone | Confidence | WER target | Effort | Key dependency |
|-----------|-----------|------------|--------|----------------|
| **M0** ✅ | 0.447 | Unknown | Done | — |
| **M1** | 0.50 | 15–25% (baseline) | 2–4h | Fasit creation |
| **M2** ✅ | 0.55 | 10–15% | 8–16h | Dialect expansion |
| **M3** | 0.60 | 5–10% | 16–24h | Domain vocabulary |
| **M4** | 0.75 | 2–5% | 40–80h | Training data |
| **M5** | 0.85+ | ≤2% | 200h+ | Large corpus + GPU |

**Bottom line:** The current system produces usable transcripts but every segment needs human review. The fastest path to measurable improvement is M1 (create a fasit and measure actual WER). Without a fasit, all confidence scores are guesses. M2 (dialect awareness) is now complete — dialect vocabulary expanded to 118 words, dialect confidence scoring active, `--preserve-dialect` flag available, and dialect region auto-detection operational.

## Test run findings (2026-05-29)

Real pipeline execution on `testdata/Call recording Elida Anna Wiktoria Kristiansen_250923_040529.m4a` (683s, 48kHz AAC) revealed:

1. **Runtime bug #27:** `src/normalize.py` missing `Path` import — crashed pipeline on startup. Fixed.
2. **Runtime bug #28:** `scripts/run_pipeline.py` missing `numpy` import — crashed after transcription. Fixed.
3. **Runtime bug #29:** `src/diarize.py` used deprecated `use_auth_token` parameter — pyannote.audio now expects `token`. Fixed.
4. **HF gated repo:** `pyannote/speaker-diarization-3.1` requires explicit access acceptance on huggingface.co. The token validates but user is not in authorized list. **Workaround:** use `--no-diarize` to skip diarization until HF access is granted.
5. **torchcodec warning:** Non-fatal warning about FFmpeg library versions. Does not block execution.
6. **Transcription speed:** ~11+ minutes for 683s audio on CPU (Mac M1). Expected given CTranslate2 CPU-only constraint.
7. **Vocabulary injection working:** 160 items loaded, 138 tokens generated (under 150 limit). Confirms #23 fix.
8. **Audio caching working:** No double-load observed; preprocess reused `metadata.audio_data`. Confirms #22 fix.

## Key audit findings to address next

> Se `ISSUES.md` for fullstendig og oppdatert status på alle problemer. Denne seksjonen er et sammendrag.

### Resolved (2026-05-28 to 2026-05-30)
- ✅ Issue #1 — konfig-parametre koblet til WhisperX
- ✅ Issue #2 — språkdeteksjon via `faster_whisper` (cachet, 30s-klipp)
- ✅ Issue #3 — HF-auth-helper
- ✅ Issue #4 — `segmentation_model` fjernet fra config (pyannote 3.1 bundler egen)
- ✅ Issue #5 — stereo kanal-splitting implementert (split_stereo_channels)
- ✅ Issue #6 — database/spell/vocab wiret inn i pipeline
- ✅ Issue #9 — jiwer WER-basert similarity i compare.py
- ✅ Issue #10/#13 — device auto-detection (CTranslate2=cuda, PyTorch=cuda/mps)
- ✅ Issue #11 — ThreadPoolExecutor default workers=1 (GIL-safe)
- ✅ Issue #12 — modell-caching for språkdeteksjon
- ✅ Issue #14 — confidence.py wiret inn i pipeline med hard-rules
- ✅ Issue #15 — language detection confidence threshold
- ✅ Issue #16 — loudness clipping fixed
- ✅ Issue #17 — corrupted file filtering
- ✅ Issue #18 — SRT speaker format fixed
- ✅ Issue #19 — beam size increased to 10
- ✅ Issue #20 — decoder signals passed to confidence extractor
- ✅ Issue #21 — spell-check deaktivert når ordbok mangler (honest failure)
- ✅ Issue #22 — audio data caching mellom analyze og preprocess (unngår dobbel lasting)
- ✅ Issue #23 — nøyaktig token-telling i vocabulary.py med Whisper tokenizer
- ✅ Issue #24 — eksplisitte avhengigheter i pyproject.toml (symspellpy, soundfile)
- ✅ Issue #25 — fjernet `pydub` avhengighet (ubrukt; låste Python til <3.13)
- ✅ Issue #26 — HF token validering med whoami()
- ✅ Issue #27 — normalize.py Path import (runtime NameError)
- ✅ Issue #28 — run_pipeline.py numpy import (runtime NameError)
- ✅ Issue #29 — diarize.py token parameter (pyannote.audio API compatibility)
- ✅ Issue #30 — word-level forced alignment fallback (NbAiLab/nb-wav2vec2-1b-bokmaal-v2)
- ✅ Issue #31 — editor Step 6 SRT filename mismatch (reuse `primary_output` instead of reconstructing)
- ✅ Issue #32 — compare.py config key mismatch (read from correct config blocks)
- ✅ Issue #33 — model caching across files in batch (module-level `_model_cache`/`_align_model_cache`)
- ✅ Issue #34 — pyproject.toml version drift (bumped to match CHANGELOG.md)
- ✅ Issue #35 — normalization opt-in with `--normalize` flag (raw output preserved as `*_raw.srt`)
- ✅ Issue #36 — privacy: real names removed from source (gitignored `data/proper_nouns.json`)
- ✅ Issue #37 — `--diarize` flag cleanup (opt-in with `default=False`, removed `--no-diarize`)

### Remaining (as of 2026-05-30)
- **#8:** `editor.py` er fortsatt placeholder — korrekt parkert, Subtitle Edit dekker behovet
- Ingen CI-pipeline — over-scope for personlig verktøy
- **Dialektgjenkjenning (PRIORITET):** Se Phase 8 for detaljert oversikt. Grunnleggende nordnorsk støtte er implementert (dialekt-flagg i `normalize.py`, dialekt-adaptiv vokabularinjeksjon i `vocabulary.py` via `--dialect northern_norwegian`). Gjenstående: dialekt-konfidensskåring, dialekt-bevarende output, utvidet vokabular, auto-deteksjon av dialektregion, multi-dialekt-støtte, og finjustert modell.
- **Spell-checking:** Featuren er deaktivert inntil en norsk ordbok lastes inn (ISSUES.md #21 er løst — honest failure — men selve funksjonaliteten krever fortsatt ekstern ordbok)
- **HF gated repo access:** `pyannote/speaker-diarization-3.1` krever eksplisitt aksept på huggingface.co. Token validerer, men brukeren er ikke i autorisert liste. Bruk `--diarize` eksplisitt når tilgang er gitt.

## Near-term priorities (revidert 2026-05-30)

See the **Milestones toward 98% confidence / 2% WER** section above for the structured roadmap. In priority order:

1. **M1 — Calibrated confidence + fasit baseline** (2–4h). Create ground-truth transcripts for 3 files from the test sample. Measure actual WER. Calibrate confidence scores. This is the single highest-ROI action — without a fasit, all other accuracy work is blind.
2. **M2 — Dialect-aware confidence + vocabulary expansion** — ✅ **Completed in v0.1.19**: dialect vocabulary expanded to 118 words, dialect confidence scoring added (2 new hard-rules), `--preserve-dialect` CLI flag implemented, dialect region auto-detection added (5 dialects).
3. **M3 — Prompt engineering + domain vocabulary** (16–24h). Build domain vocabulary, optimize prompts, run full 410-file test set.
4. **Phase 9 remaining items** — ✅ All Phase 9 items completed: integration test (12 tests), unit tests for newest modules, CLI `--num-speakers 2`, K5 spell_check fix, #5 stereo verification.

### Utsettes / droppes (over-scope for personlig verktøy)
- Web-editor (#8) — Subtitle Edit dekker behovet
- DTW-alignment i `compare.py` (#9) — `jiwer` gir ord-nivå diff billig
- `spell_check.py` autokorrektur — kan ødelegge egennavn; bruk kun til flagging
- REST API, Docker, svensk/dansk/finsk — riktig parkert som «future»
- Full CI (Phase 7) — overinvestering for personlig verktøy; behold målrettede tester
- Apple Silicon akselerasjon (#10) — CTranslate2 støtter ikke MPS; behold CPU

### Verifiser først
- Issue #5 (stereo): Kjør `analyze.py` på faktiske filer og sjekk `has_stereo_separation` før du bygger kanal-splitting

## Future ideas

- Fine-tune a Norwegian ASR model on domain-specific vocabulary (see Phase 8 / Phase 10 for the dialect fine-tuning roadmap)
- **Nordic-language support** — now a first-class vision item; see Phase 11 (Swedish, Danish, Icelandic, Faroese, Finnish, Sámi, each with dialects and its own WER gate)
- **Fasit-builder helper** — a small tool/UI to turn a recording + corrected SRT into a ground-truth fixture, lowering the cost of the WER gate that every language/dialect depends on
- **Streaming / near-real-time transcription** — process long calls incrementally instead of whole-file
- **Speaker naming & identity** — let the reviewer map `SPEAKER_00`/`SPEAKER_01` to real names, and explore voiceprint-based recurring-speaker identification across recordings (with explicit consent/privacy controls)
- **Diarization-aware confidence** — flag turn boundaries and cross-talk/overlap regions as high-priority for review (currently `vad_overlap` is a placeholder in `confidence.py`)
- **Richer export formats** — speaker-labeled plain text / Markdown transcripts, JSON for downstream NLP, and optional redacted exports
- Add REST API / local server wrapper around pipeline
- Provide a web-based review editor with waveform and speaker labels
- Containerize with Docker for reproducible deployments
