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
- [ ] **Word-level forced alignment** — nb-whisper-large-verbatim via faster-whisper lacks `align()`. Consider using a separate wav2vec2 alignment model (e.g., `NbAiLab/nb-wav2vec2-1b-bokmaal`) for forced alignment after transcription, or switch to a WhisperX-compatible model for the alignment step only. Tracked as ISSUES.md #30.

### Phase 8: Dialect recognition & adaptation
- [~] **Dialect-aware normalization** (`src/normalize.py`) — basic dialect word flagging implemented for Northern Norwegian (Nordland, Troms, Finnmark). Dialect words are flagged for awareness but NOT auto-corrected. See `NORWEGIAN_DIALECT_MAP`.
- [x] **Dialect-adaptive vocabulary** — `src/vocabulary.py` extended with `DIALECT_VOCABULARY` (30+ Northern Norwegian words across 6 categories: pronouns, negation, question words, adverbs, verbs, expressions). `load_vocabulary()` accepts `dialect="northern_norwegian"` parameter. Pipeline CLI has `--dialect northern_norwegian` flag. Dialect words are injected into Whisper's `initial_prompt` to improve recognition of non-standard forms.
- [ ] **Dialect-specific language model** — evaluate whether fine-tuning or LoRA adapters on a Norwegian dialect corpus (e.g., Nordic Dialect Corpus, NB Whisper) improves WER for Northern Norwegian speech vs. the generic nb-whisper-large-verbatim model.
- [ ] **Multi-dialect support** — extend dialect map to cover other Norwegian dialects (Trøndersk, Vestlandsk, Sørlandsk, Østlandsk) with region detection heuristics based on distinctive word patterns.
- [ ] **Dialect confidence scoring** — add dialect-specific confidence signals: flag segments where Whisper outputs standard forms but dialect forms are expected, and vice versa. Prioritize these for review.
- [ ] **Dialect-preserving output** — ensure that dialect features are preserved in SRT output and not silently normalized to Bokmål. The current approach (flag but don't correct) is the foundation; formalize as a configurable option (`--preserve-dialect`).

## Test run findings (2026-05-30)

Real pipeline execution on `testdata/Call recording Elida Anna Wiktoria Kristiansen_251023_190409.m4a` (142s, 48kHz AAC) with `--dialect northern_norwegian` revealed:

1. **Confidence hard-rules working:** 5 segments processed, all flagged. Segment with `repeated_words` + `repeated_phrases` + `possible_proper_noun` correctly ranked highest (0.517). Segments with only `low_logprob` ranked lower (0.372-0.500).
2. **Normalization working:** 21 issues flagged across 5 segments — punctuation restoration (commas before clause markers, periods, question marks) and repetition detection.
3. **Dialect vocabulary injected:** 34 Northern Norwegian dialect words added to initial prompt. Dialect forms ("ja", "da") preserved in output.
4. **Language detection:** Confidence 0.47 for Norwegian — correctly fell back to "no" (Norwegian).
5. **Word-level alignment unavailable:** `'FasterWhisperPipeline' object has no attribute 'align'` — non-fatal warning, alignment scores not available for confidence extraction.
6. **Transcription speed:** ~40s for 142s audio on CPU (Mac M1). Expected.

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

### Remaining (as of 2026-05-29)
- **#8:** `editor.py` er fortsatt placeholder — korrekt parkert, Subtitle Edit dekker behovet
- Ingen CI-pipeline — over-scope for personlig verktøy
- **Dialektgjenkjenning:** Grunnleggende nordnorsk dialekt-flagg i `normalize.py` og dialekt-adaptiv vokabularinjeksjon i `vocabulary.py` (via `--dialect northern_norwegian`) er implementert. Gjenstående: finjustert modell for nordnorsk, multi-dialekt-støtte, dialekt-konfidensskåring, og dialekt-bevarende output.
- **Spell-checking:** Featuren er deaktivert inntil en norsk ordbok lastes inn (ISSUES.md #21 er løst — honest failure — men selve funksjonaliteten krever fortsatt ekstern ordbok)
- **HF gated repo access:** `pyannote/speaker-diarization-3.1` krever eksplisitt aksept på huggingface.co. Token validerer, men brukeren er ikke i autorisert liste. Bruk `--no-diarize` inntil tilgang er gitt.

## Near-term priorities (revidert etter AUDIT.md §Strategisk gjennomgang)

1. **Ground-truth + WER-harness** (`jiwer`). Transkriber 5–10 minutter manuelt → fasit. Mål WER før alle andre endringer. ✅ `scripts/evaluate.py` på plass.
2. **Issue #1** — koble konfig-parametrene til WhisperX-kallet. ✅ Løst.
3. **Issue #2 (forbedret fiks)** — bruk `faster-whisper` sin innebygde språkdeteksjon i stedet for å laste en hel modell. ✅ Løst med modell-caching og 30s-klipp.
4. **Issue #6 (vocabulary via `initial_prompt`)** — høyest ROI for nøyaktighet. ✅ Integrert.
5. **Issue #3** — HF-auth-helper. ✅ Løst.
6. **Issue #10 (device auto-detection)** — `cuda` for transkripsjon, `cuda`/`mps` for diarization. ✅ Løst.
7. **Speaker diarization verification** — `src/diarize.py` er implementert med pyannote/speaker-diarization-3.1, og `src/transcribe.py` har `align_with_diarization()` som tildeler `SPEAKER_00`, `SPEAKER_01`, etc. til hvert segment. SRT/JSON/VTT-output inkluderer inline speaker labels. **Må verifiseres på ekte data** for å bekrefte at tildelingen er korrekt (spesielt ved kryssprat og overlap). CLI-støtte for `--min-speakers` / `--max-speakers` er ønskelig for å låse antall talere på 2 for telefonsamtaler.
8. **Mål, mål, mål** — kjør WER mot fasiten for hver endring.

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

- Fine-tune a Norwegian ASR model on domain-specific vocabulary
- Add REST API / local server wrapper around pipeline
- Provide a web-based review editor with waveform and speaker labels
- Containerize with Docker for reproducible deployments
- Add support for Swedish / Danish / Finnish in addition to Norwegian
