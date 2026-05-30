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
- [~] Model caching and memory optimization for batch jobs — language detection model cached (`_language_model`); transcription model loaded once per run. Full batch memory optimization is future work.
- [ ] Performance profiling and resource usage monitoring — **next priority after fasit exists**

### Phase 7: Quality & documentation
- [x] Add unit tests — 38 tests covering `analyze.py`, `preprocess.py`, `compare.py`, `diarize.py`
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
- [ ] **Dialect confidence scoring** — add dialect-specific confidence signals to `confidence.py`: flag segments where Whisper outputs standard forms but dialect forms are expected (and vice versa). Prioritize these for review. This catches "confidently wrong" dialect normalization where Whisper silently converts dialect to standard.
- [ ] **Dialect-preserving output** — ensure dialect features are preserved in SRT output and not silently normalized to Bokmål. Add `--preserve-dialect` CLI flag. The current approach (flag but don't correct) is the foundation; formalize as a configurable option.
- [ ] **Dialect-aware language model prompt** — expand `DIALECT_VOCABULARY` with more domain-specific dialect words (telephony, customer service, healthcare vocabulary in dialect form). Currently 34 words; target 100+.
- [ ] **Dialect region auto-detection** — analyze transcribed text for dialect markers and auto-select the appropriate dialect vocabulary, rather than requiring `--dialect northern_norwegian` to be passed manually.

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

> These are concrete defects and gaps found in a code/output audit. They are cheap, high-confidence fixes that protect the accuracy work — worth doing before piling on more features. See ISSUES.md #31–#37.

- [ ] **Fix editor Step 6 (ISSUES.md #31)** — Step 6 never runs because it reconstructs the SRT filename without the `_preprocessed` infix. Reuse `primary_output` from `transcribe_audio()` instead of rebuilding the path.
- [ ] **Fix comparison config keys (ISSUES.md #32)** — `compare.py` reads `min_agreement_threshold` / `low_confidence_threshold` that don't exist in `config.yaml` (which has `min_agreement_score`, and puts `low_confidence_threshold` under `transcription`). Thresholds silently fall back to defaults. Align the keys and add a test asserting a non-default YAML value reaches the comparer.
- [ ] **Cache models across files in batch (ISSUES.md #33)** — load the WhisperX, wav2vec2 alignment, and pyannote models once per run and reuse them; today every file reloads multi-GB models, dominating batch runtime.
- [ ] **Sync version (ISSUES.md #34)** — bump `pyproject.toml` (0.1.7) to match `CHANGELOG.md` (0.1.13) and add a release checklist / single source of truth.
- [ ] **Make normalization opt-in & preserve raw output (ISSUES.md #35)** — verbatim model output is currently overwritten in place by heuristic punctuation/capitalization. Add `--normalize` (default off for verbatim), always keep the raw model text, and treat normalization as a suggestion layer.
- [ ] **Privacy & data handling (ISSUES.md #36)** — remove real personal names from committed source (`NORWEGIAN_PROPER_NOUNS`); load proper-noun/vocabulary lists from a gitignored local data file. Document a data-retention policy and consider optional at-rest encryption and a `--redact` mode.
- [ ] **Clean up `--diarize` flag (ISSUES.md #37)** — it's a no-op redundant with the default-on behavior; either make diarization opt-in or drop the redundant flag.
- [ ] **Integration test on a tiny real/synthetic clip** — covers the orchestrator glue (currently untested): the path-building, config wiring, normalization, and confidence steps that unit tests don't touch.
- [ ] **Unit tests for the newest modules** — `normalize.py`, `vocabulary.py`, `spell_check.py`, `database.py`, `editor.py` have no tests; they are also where the most recent churn is.
- [ ] **CLI `--num-speakers 2` convenience** — lock 2-party telephone calls to two speakers (config already supports `num_speakers_override`; surface it on the CLI).

### Phase 10: Norwegian — all dialects (extends Phase 8)

> Vision step 1: a single Norwegian pipeline that handles any dialect with a measurable WER. Phase 8 covers Northern Norwegian specifically; this phase generalizes it.

- [ ] **Pluggable dialect packs** — refactor the Northern-Norwegian-specific maps (`NORWEGIAN_DIALECT_MAP`, `DIALECT_VOCABULARY`, place names) into per-dialect data files (`data/dialects/<region>.json`) with a common loader, so adding a dialect is data, not code.
- [ ] **All major Norwegian dialect regions** — Trøndersk, Vestlandsk, Sørlandsk, Østlandsk, Innlandet, plus finer-grained sub-regions, each as a dialect pack (vocabulary + standard-form mapping for flagging).
- [ ] **Automatic dialect-region detection** — classify a recording's dialect from distinctive markers (e.g. `eg`→Vestlandsk, `dæm`→Trøndersk, `æ`+`ikkje`→Nordnorsk) and auto-select the matching pack; fall back to a generic Norwegian pack.
- [ ] **Bokmål + Nynorsk alignment routing** — already partially present (`nb-wav2vec2-1b-bokmaal` vs `-nynorsk`); make the written-standard target selectable per file/segment.
- [ ] **Dialect WER tracking** — once fasits exist, report WER per dialect so we can see which dialects the base model handles well vs. poorly and target effort accordingly.

### Phase 11: Nordic languages (long-term vision)

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

### M1 — Calibrated confidence + fasit baseline
- **Target confidence:** 0.50 mean (calibrated)
- **Target WER:** Measure baseline (likely 15–25% WER for call recordings)
- **What it takes:**
  - Manually transcribe 3 files from the test sample (short ~5s, medium ~90s, long ~460s) → create ground-truth fasit
  - Run `jiwer` to establish actual WER baseline
  - Calibrate confidence scores: fit a logistic regression on `priority_score` → probability-of-error, using fasit as labels
  - Fix alignment model: investigate why `nb-wav2vec2-1b-bokmaal-v2` returns 0 word-level scores for all files
- **Estimated effort:** 2–4 hours (manual transcription + calibration code)
- **Gate for next:** WER baseline known; confidence scores correlate with actual errors

### M2 — Dialect-aware confidence + vocabulary expansion
- **Target confidence:** 0.55 mean (calibrated)
- **Target WER:** 10–15% (halve the baseline)
- **What it takes:**
  - Expand dialect vocabulary from 34 → 100+ Northern Norwegian words
  - Add dialect confidence scoring: flag segments where Whisper outputs standard forms but dialect expected
  - Implement `--preserve-dialect` flag to prevent silent normalization
  - Add dialect region auto-detection from transcribed text
  - Fix model caching across files (load WhisperX model once per run, not per file) — reduces batch runtime by ~5×
- **Estimated effort:** 8–16 hours
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
| **M2** | 0.55 | 10–15% | 8–16h | Dialect expansion |
| **M3** | 0.60 | 5–10% | 16–24h | Domain vocabulary |
| **M4** | 0.75 | 2–5% | 40–80h | Training data |
| **M5** | 0.85+ | ≤2% | 200h+ | Large corpus + GPU |

**Bottom line:** The current system produces usable transcripts but every segment needs human review. The fastest path to measurable improvement is M1 (create a fasit and measure actual WER). Without a fasit, all confidence scores are guesses.

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

### Resolved (2026-05-28 to 2026-05-29)
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

### Remaining (as of 2026-05-30)
- **#8:** `editor.py` er fortsatt placeholder — korrekt parkert, Subtitle Edit dekker behovet
- Ingen CI-pipeline — over-scope for personlig verktøy
- **Dialektgjenkjenning (PRIORITET):** Se Phase 8 for detaljert oversikt. Grunnleggende nordnorsk støtte er implementert (dialekt-flagg i `normalize.py`, dialekt-adaptiv vokabularinjeksjon i `vocabulary.py` via `--dialect northern_norwegian`). Gjenstående: dialekt-konfidensskåring, dialekt-bevarende output, utvidet vokabular, auto-deteksjon av dialektregion, multi-dialekt-støtte, og finjustert modell.
- **Spell-checking:** Featuren er deaktivert inntil en norsk ordbok lastes inn (ISSUES.md #21 er løst — honest failure — men selve funksjonaliteten krever fortsatt ekstern ordbok)
- **HF gated repo access:** `pyannote/speaker-diarization-3.1` krever eksplisitt aksept på huggingface.co. Token validerer, men brukeren er ikke i autorisert liste. Bruk `--no-diarize` inntil tilgang er gitt.

## Near-term priorities (revidert 2026-05-30)

See the **Milestones toward 98% confidence / 2% WER** section above for the structured roadmap. In priority order:

1. **M1 — Calibrated confidence + fasit baseline** (2–4h). Create ground-truth transcripts for 3 files from the test sample. Measure actual WER. Calibrate confidence scores. This is the single highest-ROI action — without a fasit, all other accuracy work is blind.
2. **M2 — Dialect-aware confidence + vocabulary expansion** (8–16h). Expand dialect vocabulary, add dialect confidence scoring, implement `--preserve-dialect`, fix model caching.
3. **M3 — Prompt engineering + domain vocabulary** (16–24h). Build domain vocabulary, optimize prompts, run full 410-file test set.
4. **Phase 9 engineering hardening** — fix ISSUES.md #31–#37 (editor step 6, config keys, model caching, version sync, normalization opt-in, privacy, diarize flag cleanup). These are cheap, high-confidence fixes that protect accuracy work.

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
