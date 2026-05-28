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

- `transcribe.py` bygger en `Transcriber` med konfigurasjon for `beam_size`, `word_timestamps`, `condition_on_previous_text` og `initial_prompt`, men sender ikke disse parameterne videre til WhisperX-kallet.
- `analyze.py` bruker `whisper` for språkdeteksjon, mens `pyproject.toml` kun har `whisperx`. Dette kan gi et avhengighetsgap ved installasjon.
- `diarize.py` bruker `Pipeline.from_pretrained(..., use_auth_token=True)`, så Hugging Face-pålogging er nødvendig for pyannote-modellen.
- `config.yaml` inneholder `segmentation_model`, men koden bruker ikke dette feltet i diarization-kallet.
- Stereo-innhold behandles ved å slå sammen kanaler til mono. Dette kan redusere nøyaktigheten for ekte flere-høyttaler stereo-opptak.
- `database.py`, `spell_check.py` og `vocabulary.py` er ikke aktivt integrert i `scripts/run_pipeline.py`.
- Det finnes ingen enhetstester eller integrasjonstester i repoet.

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

- Ingen tests eller `tests/`-mappe i repoet enda
- `editor.py` er kun en SRT-funksjon og ikke et ekte web-UI
- `database.py`, `spell_check.py` og `vocabulary.py` er ikke integrert i pipeline
- `transcribe.py` bruker ikke alle konfigurasjonsparametre for WhisperX i nåværende implementasjon
- `analyze.py` har et potensielt avhengighetsgap mellom `whisper` og `whisperx`
- Diarization krever aktiv Hugging Face-pålogging for `pyannote/speaker-diarization-3.1`
- Ekte stereo med én taler per kanal håndteres ikke optimalt

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
