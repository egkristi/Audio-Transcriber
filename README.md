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
- ✅ `analyze.py` bruker nå `faster_whisper` direkte for språkdeteksjon (cachet, 30s-klipp) i stedet for standalone `whisper`.
- ✅ `diarize.py` har nå `check_hf_auth()`-hjelper med graceful feilmelding ved manglende HF-token.
- ✅ `database.py`, `spell_check.py` og `vocabulary.py` er nå integrert i pipeline via `--use-database`, `--spell-check` og `--vocabulary-file`.
- ✅ 31 enhetstester er på plass for `analyze`, `preprocess`, `compare` og `diarize`.
- ✅ `scripts/evaluate.py` — WER/CER-evalueringsharness med `jiwer` for ground-truth-sammenligning.
- ✅ Device auto-detection: `cuda` for transkripsjon (CTranslate2), `cuda`/`mps` for diarization (PyTorch).
- ✅ `src/confidence.py` — konfidens-flagging med flere deterministiske signaler (alignment score, avg_logprob, no_speech_prob, compression_ratio, temperature, word probability, model disagreement, SNR). Prioriterer segmenter for manuell review.

### Gjenstående
Se `ISSUES.md` for fullstendig og oppdatert liste over åpne og løste problemer. Nedenfor er et sammendrag:

- **#4:** `config.yaml` inneholder `segmentation_model`, men koden bruker ikke dette feltet.
- **#5:** Stereo-innhold behandles ved å slå sammen kanaler til mono — verifiser på faktiske filer først.
- **#11:** `ThreadPoolExecutor` i batch-modus gir ikke ekte parallellisme for CPU-tunge oppgaver pga. Python GIL.
- **#14:** `src/confidence.py` er designet og testet, men ikke wiret inn i pipeline.
- **#8:** `editor.py` er fortsatt kun en SRT-eksportfunksjon, ikke en ekte web-editor.
- **#9:** `compare.py` bruker fortsatt enkel tids-overlap-alignment og `SequenceMatcher`.
- Ingen CI-pipeline (GitHub Actions) eller integrasjonstester.

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
