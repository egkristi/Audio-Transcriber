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

### Remaining (as of 2026-05-29)
- **#8:** `editor.py` er fortsatt placeholder — korrekt parkert, Subtitle Edit dekker behovet
- Ingen CI-pipeline — over-scope for personlig verktøy
- **Spell-checking:** Featuren er deaktivert inntil en norsk ordbok lastes inn (ISSUES.md #21 er løst — honest failure — men selve funksjonaliteten krever fortsatt ekstern ordbok)

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
