# Audio-Transcriber

Et Python-basert transkripsjonssystem for norsk tale, med særlig vekt på **nordnorske dialekter** (Nordland, Troms, Finnmark). Prosjektet kombinerer en flerfase-pipeline for analyse, adaptiv forhåndsbehandling, speaker diarization, flermodell-transkripsjon, sammenligning og manuell sluttkorrektur.

## Hva er på plass nå

Denne repoen inneholder en fungerende proof-of-concept pipeline med de viktigste modulene implementert, men også flere kjente begrensninger og tekniske gap som må adresseres før produksjon.

**Implementert:**
- `src/analyze.py` — lydanalyse, metadata, VAD og språkdeteksjon
- `src/preprocess.py` — resampling, mono-konvertering, filtering, loudness-normalisering
- `src/diarize.py` — pyannote-baserte speaker segments
- `src/transcribe.py` — WhisperX-transkripsjon og SRT/JSON/VTT-eksport
- `src/compare.py` — modell-sammenligning og avviksmarkering
- `scripts/run_pipeline.py` — CLI/orkestrering av steg og batch-prosessering

**Delvis implementert / placeholder:**
- `src/editor.py` — SRT-eksport og manuelle instruksjoner, ikke web-editor
- `src/database.py` — SQLite-modul for jobblogg, men ikke koblet til pipeline
- `src/spell_check.py` — norsk stavekontroll-prototype
- `src/vocabulary.py` — egendefinert ordliste / prompt-generator

## Viktige auditfunn

### Løst (2026-05-28 til 2026-05-29)
- ✅ `transcribe.py` sender nå `beam_size`, `vad_filter`, `condition_on_previous_text` og `initial_prompt` videre til WhisperX-kallet.
- ✅ `analyze.py` bruker nå `faster_whisper` direkte for språkdeteksjon (cachet, 30s-klipp) i stedet for standalone `whisper`.
- ✅ `diarize.py` har nå `check_hf_auth()`-hjelper med graceful feilmelding ved manglende HF-token.
- ✅ `database.py`, `spell_check.py` og `vocabulary.py` er nå integrert i pipeline via `--use-database`, `--spell-check` og `--vocabulary-file`.
- ✅ 38 enhetstester er på plass for `analyze`, `preprocess`, `compare`, `diarize` og `confidence`.
- ✅ `scripts/evaluate.py` — WER/CER-evalueringsharness med `jiwer` for ground-truth-sammenligning.
- ✅ Device auto-detection: `cuda` for transkripsjon (CTranslate2), `cuda`/`mps` for diarization (PyTorch).
- ✅ `src/confidence.py` — konfidens-flagging med 20 hard-rules, wiret inn i pipeline; eksporterer `*_review_list.txt` med alle segmenter.
- ✅ `segmentation_model` fjernet fra `config.yaml` (pyannote 3.1 bundler egen segmentering).
- ✅ `ThreadPoolExecutor` default `--workers` endret fra 4 til 1 (GIL-safe for CPU-bound inferens).
- ✅ `compare.py` bruker nå `jiwer.wer()` for ord-nivå similarity (bedre enn `SequenceMatcher`).
- ✅ Stereo kanal-splitting implementert (`split_stereo_channels()`), med advarsel når `has_stereo_separation=True`.
- ✅ Loudness clipping fikset (pre-clipping + -20 LUFS target).
- ✅ Språkdeteksjon med confidence threshold (fallback til "no" når confidence < 0.5).
- ✅ Korrupte filer filtreres ut (<1KB) i batch-modus.
- ✅ SRT speaker format fikset (inline `SPEAKER_00: tekst`).
- ✅ Per-segment og per-file confidence levels implementert.
- ✅ Norsk tekst-normalisering (`src/normalize.py`) med automatisk SRT-regenerering.
- ✅ Default norsk vokabular (`data/norwegian_vocabulary.json`) auto-lastet for `initial_prompt`.

### Gjenstående
Se `ISSUES.md` for fullstendig og oppdatert liste over åpne og løste problemer. Nedenfor er et sammendrag:

- **#8:** `editor.py` er fortsatt kun en SRT-eksportfunksjon, ikke en ekte web-editor — korrekt parkert, Subtitle Edit dekker behovet.
- **#5:** Stereo kanal-splitting er implementert, men pipeline kjører fortsatt på averaged mono. Full kanal-integrasjon er fremtidig arbeid.
- **#21:** Norsk stavekontroll krever ekstern ordbok (NST/UiB) — feature er deaktivert inntil ordbok er på plass.
- Ingen CI-pipeline (GitHub Actions) — over-scope for personlig verktøy.

## Rask installasjon

### Forutsetninger
- macOS
- Python 3.11
- Homebrew (valgfritt)
- `ffmpeg`
- `uv`

### Installer og kjør

```bash
brew install uv ffmpeg
cd /Users/erling/code/Audio-Transcriber
uv sync
uv run python scripts/run_pipeline.py --help
```

### Hugging Face-krypteringsnøkkel for pyannote

```bash
uv run huggingface-cli login
```

## Brukseksempler

### Kjør hele pipeline på én fil

```bash
uv run python scripts/run_pipeline.py \
  --input recording.m4a \
  --output-dir ./output \
  --diarize \
  --compare-models
```

### Kjør batch på en mappe

```bash
uv run python scripts/run_pipeline.py \
  --input ./recordings \
  --output-dir ./output \
  --diarize \
  --compare-models \
  --workers 1
```

### Kjør enkeltsteg

```bash
uv run python scripts/run_pipeline.py --input file.m4a --step analyze
uv run python scripts/run_pipeline.py --input file.m4a --step preprocess
uv run python scripts/run_pipeline.py --input file.m4a --step diarize
uv run python scripts/run_pipeline.py --input file.m4a --step transcribe --primary-model NbAiLab/nb-whisper-large-verbatim
```

## Konfigurasjon

Konfigurasjonen er i `config.yaml`.

Viktige seksjoner:
- `analysis` — VAD, språkdeteksjon, metadata
- `preprocessing` — sample rate, filtre, loudness, denoising
- `diarization` — pyannote-modell og speaker-innstillinger
- `transcription` — modellnavn, språk, confidence thresholds
- `comparison` — avtalenivå og flagglogikk
- `output` — format og eksportinnstillinger
- `performance` — enheter og compute type

## Filstruktur

```
Audio-Transcriber/
├── src/
│   ├── analyze.py
│   ├── config.py
│   ├── compare.py
│   ├── database.py
│   ├── diarize.py
│   ├── editor.py
│   ├── preprocess.py
│   ├── spell_check.py
│   ├── transcribe.py
│   ├── utils.py
│   └── vocabulary.py
├── scripts/
│   └── run_pipeline.py
├── config.yaml
├── pyproject.toml
├── README.md
├── ROADMAP.md
└── CHANGELOG.md
```

## Videre utvikling

Se `ROADMAP.md` for detaljer om neste prioriteringer og oppgaver.  
Se `AUDIT.md` §«Strategisk gjennomgang» for strategisk gjennomgang med anbefalt rekkefølge og elementer som bør utsettes.

## Lisens

MIT
