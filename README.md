# Audio-Transcriber

Et verktøy for å transkribere lydfiler til tekst med høy nøyaktighet. Systemet bruker en flerfase-pipeline som kombinerer flere transkripsjonsmodeller, speaker-diarization og manuell sluttkorrektur for å oppnå best mulig resultat.

---

## Pipeline (6 steg per lydfil)

For hver lydfil følges stegene nedenfor. De kan kjøres enkeltvis, i batch, eller på hele mapper.

### Steg 1: Analyse og metadata

Lydfilen analyseres og all metadata lagres:

- Filnavn, varighet, sample rate, kanaler, bit rate, codec
- Volumprofil (LUFS, peak, dynamikk)
- Støy- og talende-segmenter (VAD)
- Språkgjetting (hvis ukjent)

Metadata lagres som JSON ved siden av lydfilen og brukes i alle senere steg.

### Steg 2: Forhåndsbehandling

Lyden konverteres og transformeres for å optimalisere transkribering:

- Konvertering til 16 kHz mono WAV
- High-pass filter (80 Hz) for å fjerne rumling
- Low-pass filter (8000 Hz) for å fjerne unødvendig høyfrekvent støy
- Loudness-normalisering (ITU-R BS.1770-4)
- Eventuell støyreduksjon (valgfritt)

> **Merk:** Å forhåndsbehandle lyden gir tydelig bedre resultat enn å transkribere rå `.m4a` direkte.

### Steg 3: Transkripsjon fase 1 (med speaker-diarization)

Første transkripsjonsrunde kjøres med speaker-separasjon:

- **Primær modell:** `Necklace/faster-nb-whisper-large` (CT2-konvertert, rask)
- **Alternativ modell:** `NbAiLab/nb-whisper-large` (original norsk Whisper, marginalt bedre nøyaktighet)
- **Speaker-diarization:** `pyannote/speaker-diarization-3.1`
- **Segmentering:** `pyannote/segmentation-3.0`
- **Output:** SRT/JSON med tidsstempler og `SPEAKER_00`, `SPEAKER_01`

Resultatet fra fase 1 lagres og brukes som utgangspunkt for sammenligning.

### Steg 4: Transkripsjon fase 2 (alternativ modell)

Samme lydfil transkriberes på nytt med en annen modell for å sammenligne resultater:

- **Alternativ modell:** `NbAiLab/nb-whisper-large` (hvis fase 1 brukte `Necklace/faster-nb-whisper-large`), eller `openai/whisper-large-v3`
- Samme speaker-diarization som i fase 1, eller uten hvis fokus er ren tekstnøyaktighet
- Output lagres separat

### Steg 5: Konsensus og re-transkripsjon

Systemet sammenligner resultatene fra fase 1 og fase 2:

- Segmenter med lav konfidens (< 0.85) eller uenighet mellom modellene identifiseres
- Disse segmentene transkriberes på nytt med begge modeller (eventuelt en tredje som tie-breaker)
- Beste resultat velges per segment basert på konfidens og semantisk likhet
- Et endelig, konsolidert transkripsjonsdokument genereres

### Steg 6: Sluttkorrektur (menneskelig interaksjon)

Uansett modell vil egennavn, tall, adresser og fagord ha noen feil. Derfor er siste steg manuell gjennomgang:

- Åpne den genererte SRT-filen i **Subtitle Edit for Mac** (gratis) eller en teksteditor
- Spill av lyden parallelt og rett opp feil
- Dette er der du henter de siste prosentene mot 100 %

> **Anbefaling:** Vurder å sette opp en lokal webbasert tekst-lyd-editor (f.eks. basert på [wavesurfer.js](https://wavesurfer-js.org/)) for en mer integrert opplevelse direkte i prosjektet.

---

## Oppsett

**Forutsetninger:** Homebrew, Python 3.11, ffmpeg og uv må være installert.

### 1. Sett opp prosjektmappe og miljø med uv

```bash
git clone <repo-url> ~/Audio-Transcriber
cd ~/Audio-Transcriber
uv sync
```

`pyproject.toml` definerer avhengighetene, inkludert:

```toml
[project]
dependencies = [
    "faster-whisper",
    "whisperx",
    "pyannote.audio",
    "ffmpeg-python",
    "pydub",
    "numpy",
]
```

> **Merk:** `uv` håndterer automatisk virtuelt miljø, låsing av avhengigheter (`uv.lock`) og installasjon. Du trenger ikke å aktivere et `venv` manuelt — bruk `uv run` for å kjøre kommandoer i prosjektets miljø.

### 4. Hugging Face-token (for speaker-diarization)

1. Lag bruker på [huggingface.co](https://huggingface.co)
2. Godkjenn vilkårene for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generer token: **Settings → Access Tokens → New token (Read)**
4. Logg inn lokalt:
   ```bash
   huggingface-cli login
   ```

---

## Bruk

### Enkeltfil

```bash
uv run python scripts/run_pipeline.py \
  --input ~/Documents/CallRecordings/Call_recording_XXX.m4a \
  --output-dir ~/transcripts \
  --diarize \
  --compare-models
```

### Batch / hel mappe

```bash
uv run python scripts/run_pipeline.py \
  --input ~/Documents/CallRecordings/ \
  --output-dir ~/transcripts \
  --diarize \
  --compare-models \
  --workers 4
```

### Kjør enkeltsteg

```bash
# Kun analyse
uv run python scripts/run_pipeline.py --input file.m4a --step analyze

# Kun forhåndsbehandling
uv run python scripts/run_pipeline.py --input file.m4a --step preprocess

# Kun transkripsjon fase 1 (rask modell)
uv run python scripts/run_pipeline.py --input file.m4a --step transcribe --model faster-nb-whisper-large

# Kun transkripsjon fase 1 (original norsk modell)
uv run python scripts/run_pipeline.py --input file.m4a --step transcribe --model nb-whisper-large
```

---

## Filstruktur

```
~/Audio-Transcriber/
├── .venv/                  # Virtuelt miljø (håndteres av uv)
├── src/
│   ├── analyze.py          # Steg 1: Metadata og analyse
│   ├── preprocess.py       # Steg 2: Lydforberedelse
│   ├── transcribe.py       # Steg 3 & 4: Transkripsjon
│   ├── compare.py          # Steg 5: Sammenligning og konsensus
│   └── editor.py           # Steg 6: Web-editor for sluttkorrektur
├── scripts/
│   └── run_pipeline.py     # Hovedscript som orkestrerer alle steg
├── pyproject.toml          # Prosjekt- og avhengighetskonfigurasjon
├── uv.lock                 # Låst avhengighetsgraf (autogenerert)
├── config.yaml             # Innstillinger for modeller, thresholds, etc.
└── README.md
```

---

## Anbefalinger for forbedringer

1. **Konfidensbasert filtrering:** Lagre konfidens-score per ord/segment fra Whisper. Bruk dette til å markere usikre områder visuelt i editoren.

2. **Tredje modell som tie-breaker:** Ved uenighet mellom fase 1 og 2, kjør en tredje modell (f.eks. `NbAiLab/nb-whisper-large`) og bruk majoritetsstemme.

3. **Word-level timestamps:** Aktiver `--word_timestamps True` i Whisper for mer presis synkronisering i editoren.

4. **Automatisk stavekontroll:** Kjør norsk stavekontroll (f.eks. via `symspellpy` eller `transformers`-basert modell) på transkripsjonen før sluttkorrektur for å fange åpenbare feil.

5. **Web-basert editor:** Bytt ut ekstern SRT-editor med en innebygget web-editor (f.eks. React + wavesurfer.js) som viser bølgeform, speaker-farger og konfidens-markeringer direkte i nettleseren.

6. **Konfigurasjonsfil:** Flytt alle innstillinger (modellnavn, thresholds, filterparametre) til `config.yaml` slik at pipeline-en er reproduserbar og enkel å justere.

7. **Logging og sporbarhet:** Logg alle kjøringer med parametre, modellversjoner og resultater til en SQLite-database eller JSON-lines-fil. Dette gjør det mulig å sammenligne resultater over tid.

8. **GPU-støtte (valgfritt):** På maskiner med NVIDIA GPU kan `faster-whisper` kjøres med `device="cuda"` og `compute_type="float16"` for betydelig hastighetsøkning. På Apple Silicon brukes fortsatt `device="cpu"` med `int8`.

9. **Parallellprosessering:** For batch-kjøringer, bruk `multiprocessing` eller `concurrent.futures` for å transkribere flere filer samtidig (begrenset av RAM og CPU-kjerner).

10. **VAD-forhåndsfilter:** Bruk `silero-vad` eller innebygd WhisperX-VAD for å fjerne lange stillhetssegmenter før transkripsjon. Dette sparer tid og reduserer hallucinering.

---

## Ytelse

- **Første kjøring:** Modellen lastes ned (~3 GB) til `~/.cache/huggingface/`
- **Påfølgende kjøringer:** På M1 Max kan du regne med ca. **5–10× sanntid** (10 minutter lyd ⇒ 1–2 minutter transkripsjon)
- **Speaker-diarization:** Legger typisk til 20–40 % ekstra tid

---

## Lisens

MIT
