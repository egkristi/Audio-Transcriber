# Audio-Transcriber

Et Python-basert transkripsjonssystem for norsk tale. Prosjektet kombinerer en flerfase-pipeline for analyse, adaptiv forhåndsbehandling, speaker diarization, flermodell-transkripsjon, sammenligning og manuell sluttkorrektur.

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

### Løst (2026-05-28)
- ✅ `transcribe.py` sender nå `beam_size`, `vad_filter`, `condition_on_previous_text` og `initial_prompt` videre til WhisperX-kallet.
- ✅ `analyze.py` bruker nå `whisperx` for språkdeteksjon i stedet for standalone `whisper`.
- ✅ `diarize.py` har nå `check_hf_auth()`-hjelper med graceful feilmelding ved manglende HF-token.
- ✅ `database.py`, `spell_check.py` og `vocabulary.py` er nå integrert i pipeline via `--use-database`, `--spell-check` og `--vocabulary-file`.
- ✅ 31 enhetstester er på plass for `analyze`, `preprocess`, `compare` og `diarize`.

### Gjenstående
- `config.yaml` inneholder `segmentation_model`, men koden bruker ikke dette feltet i diarization-kallet.
- Stereo-innhold behandles ved å slå sammen kanaler til mono. Dette kan redusere nøyaktigheten for ekte flere-høyttaler stereo-opptak.
- `ThreadPoolExecutor` i batch-modus gir ikke ekte parallellisme for CPU-tunge oppgaver (transkripsjon, diarization) pga. Python GIL. Vurder `ProcessPoolExecutor` eller begrensning til én worker per GPU.
- `analyze.py` laster en hel `whisperx` tiny-modell (~39 MB) kun for språkdeteksjon. Dette er tungvint for et metadatasteg. Vurder caching eller lettere deteksjon.
- `device="cpu"` og `compute_type="int8"` er hardkodet i `transcribe.py` og `diarize.py`. Ingen auto-deteksjon av `mps` (Apple Silicon) eller `cuda`.
- `compare.py` bruker fortsatt enkel tids-overlap-alignment (>50 %) og `SequenceMatcher`. Ingen ord-nivå WER-beregning.
- `editor.py` er fortsatt kun en SRT-eksportfunksjon, ikke en ekte web-editor.
- Ingen CI-pipeline (GitHub Actions) eller integrasjonstester enda.

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
  --workers 4
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

## Nåværende kjente begrensninger

- `editor.py` er kun en SRT-funksjon og ikke et ekte web-UI
- `config.yaml` inneholder `segmentation_model`, men koden bruker ikke dette feltet
- Ekte stereo med én taler per kanal håndteres ikke optimalt (kanaler slås sammen til mono)
- `ThreadPoolExecutor` i batch-modus gir ikke ekte parallellisme for CPU-tunge oppgaver pga. Python GIL
- `analyze.py` laster en hel `whisperx` tiny-modell (~39 MB) kun for språkdeteksjon
- `device="cpu"` og `compute_type="int8"` er hardkodet — ingen auto-deteksjon av `mps` eller `cuda`
- `compare.py` bruker fortsatt enkel tids-overlap-alignment og `SequenceMatcher`, ikke ord-nivå WER
- Ingen CI-pipeline (GitHub Actions) eller integrasjonstester enda

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

## Lisens

MIT
