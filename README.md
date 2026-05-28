# Audio-Transcriber

Et verktøy for å transkribere lydfiler til tekst med høy nøyaktighet. Systemet bruker en flerfase-pipeline som kombinerer flere transkripsjonsmodeller, speaker-diarization og manuell sluttkorrektur for å oppnå best mulig resultat.

---

## Status: Pipeline Fully Implemented ✓

The complete 6-stage pipeline has been implemented and is ready for testing:

- ✅ **Step 1 - Analyze**: Audio metadata extraction with VAD, language detection, bandwidth detection
- ✅ **Step 2 - Preprocess**: Adaptive audio conditioning with filtering, resampling, loudness normalization  
- ✅ **Step 3 - Diarize**: Speaker diarization using PyAnnote
- ✅ **Steps 3 & 4 - Transcribe**: Multi-model transcription with SRT/JSON/VTT output
- ✅ **Step 5 - Compare**: Multi-model comparison with deviation marking
- ✅ **Step 6 - Editor**: Export for manual review in external editors
- ✅ **Orchestration**: Full CLI with batch processing and worker pools

### Quick Start

```bash
# Single file with all steps
uv run python scripts/run_pipeline.py \
  --input recording.m4a \
  --output-dir ./output \
  --diarize --compare-models

# Batch process folder with 4 parallel workers  
uv run python scripts/run_pipeline.py \
  --input ./recordings \
  --output-dir ./output \
  --workers 4
```

---

## Pipeline (6 steg per lydfil)

For hver lydfil følges stegene nedenfor. De kan kjøres enkeltvis, i batch, eller på hele mapper.

### Steg 1: Analyse og metadata

Lydfilen analyseres og all metadata lagres:

- Filnavn, varighet, sample rate, kanaler, bit rate, codec
- **Reell båndbredde** (smalbånd ~8 kHz vs. bredbånd ~16 kHz) — styrer forhåndsbehandlingen i Steg 2
- **Antall kanaler** — ekte stereo med én part per kanal gir et bedre alternativ enn diarization
- Volumprofil (LUFS, peak, dynamikk)
- Støy- og talende-segmenter (VAD)
- Språkgjetting (hvis ukjent)

Metadata lagres som JSON ved siden av lydfilen og brukes i alle senere steg.

### Steg 2: Forhåndsbehandling

Lyden konverteres og transformeres for å optimalisere transkribering. Behandlingen tilpasses opptaket basert på metadata fra Steg 1 — den kjøres ikke blindt:

- Konvertering til 16 kHz mono WAV
- High-pass filter (80 Hz) for å fjerne rumling — trygt for alle opptak
- Low-pass filter (8000 Hz) **kun for smalbånds-telefoni.** Hvis Steg 1 oppdager reell bredbåndslyd (VoLTE/VoWiFi), dropp lavpass — du kaster ellers bort nyttig signal
- Loudness-normalisering (ITU-R BS.1770-4, mål −16 LUFS)
- Støyreduksjon: **av som default.** Aggressiv denoising lager artefakter som kan forverre Whisper-resultatet. Slå kun på ved tydelig konstant bakgrunnsstøy, og test med/uten

> **Merk:** Å forhåndsbehandle lyden gir tydelig bedre resultat enn å transkribere rå `.m4a` direkte.

### Steg 3: Diarization og primærtranskripsjon

Speaker-diarization kjøres **én gang** her og gjenbrukes på alle senere transkripsjoner — den er uavhengig av ASR-modellen, så det er sløsing å kjøre den per modell-fase.

- **Speaker-diarization:** `pyannote/speaker-diarization-3.1`
- **Segmentering:** `pyannote/segmentation-3.0`
- For to-parts samtaler: sett `min_speakers=2`, `max_speakers=2`
- **Snarvei:** Hvis Steg 1 viser ekte stereo med én part per kanal, splitt kanalene og transkriber hver for seg — mer presist enn diarization, og dropper pyannote helt

Primærmodell (velg bevisst — se merknad under):

- **`NbAiLab/nb-whisper-large-verbatim`** — ordrett, gjengir det som faktisk ble sagt (anbefalt for samtaleopptak der ordlyd har betydning)
- **`NbAiLab/nb-whisper-large`** (main) — vasker muntlig språk og retter grammatikk, mer lesbart, men endrer ordlyden

Kjør med `word_timestamps=True` fra start (editoren i Steg 6 trenger det), `beam_size=5` og `vad_filter=True`.

- **Output:** SRT/JSON med tidsstempler og `SPEAKER_00`, `SPEAKER_01`

> **Viktig om modellvalg:** `Necklace/faster-nb-whisper-large` (CT2) og `NbAiLab/nb-whisper-large` (PyTorch) er *samme vekter* — kun ulik motor/presisjon. De gir tilnærmet identisk transkript, så å sammenligne dem mot hverandre gir ingen reell gevinst. CT2-versjonen brukes fordi den er raskere og fordi WhisperX bygger på den.

### Steg 4: Andre mening med uavhengig modell (valgfri)

Samme lydfil transkriberes på nytt med en **arkitektonisk forskjellig** modell — ikke samme NB-Whisper på en annen motor:

- **Alternativ modell:** `openai/whisper-large-v3` (nyere multilingual base), eller NB-Whisper main hvis primær var verbatim
- **Gjenbruk speaker-tidslinjen fra Steg 3** — ikke kjør diarization på nytt
- Output lagres separat for sammenligning i Steg 5

### Steg 5: Avviksmarkering (ikke automatisk konsensus)

Systemet sammenligner fase 1 og fase 2 for å **styre** den menneskelige gjennomgangen — ikke for å avgjøre fasit automatisk:

- Transkriptene aligns på ord/segment-nivå (WER-stil diff)
- Segmenter med lav konfidens (`avg_logprob` < 0.85) eller uenighet mellom modellene flagges
- De flaggede segmentene blir en **prioritert arbeidsliste** for Steg 6

> **Merk:** Ikke la maskinen auto-velge «beste» segment. Whisper sin konfidens er ikke kalibrert, og to modeller bommer ofte korrelert på det vanskeligste (navn, tall, overlappende tale) — der hjelper ikke konsensus. Hopp gjerne over hele dette steget hvis tid er knapp; Steg 6 fanger uansett opp feilene.

### Steg 6: Sluttkorrektur (menneskelig interaksjon)

Uansett modell vil egennavn, tall, adresser og fagord ha noen feil. Derfor er siste steg manuell gjennomgang:

- Åpne den genererte SRT-filen i **Subtitle Edit for Mac** (gratis) eller en teksteditor
- Spill av lyden parallelt, prioriter de flaggede segmentene fra Steg 5, og rett opp feil
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

`pyproject.toml` definerer avhengighetene. La WhisperX styre versjonene av faster-whisper/ctranslate2/pyannote — disse listes ikke som egne topp-nivå-deps, fordi whisperx pinner kompatible versjoner og du ellers får resolver-konflikter:

```toml
[project]
requires-python = ">=3.11,<3.13"
dependencies = [
    "whisperx",          # drar inn faster-whisper, ctranslate2 og pyannote.audio
    "ffmpeg-python",
    "pydub",
    "numpy",
    "pyyaml",
]
```

> **Merk:** `uv` håndterer automatisk virtuelt miljø, låsing av avhengigheter (`uv.lock`) og installasjon. Du trenger ikke å aktivere et `venv` manuelt — bruk `uv run` for å kjøre kommandoer i prosjektets miljø.

> **Merk om `openai/whisper-large-v3` (Steg 4):** Hvis den krever en annen ctranslate2-versjon enn whisperx tåler, kjør den i et separat uv-miljø i stedet for å tvinge alt inn i ett.

### 2. Hugging Face-token (for speaker-diarization)

1. Lag bruker på [huggingface.co](https://huggingface.co)
2. Godkjenn vilkårene for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generer token: **Settings → Access Tokens → New token (Read)**
4. Logg inn lokalt:
   ```bash
   uv run huggingface-cli login
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

# Kun diarization
uv run python scripts/run_pipeline.py --input file.m4a --step diarize

# Kun transkripsjon (ordrett verbatim-modell)
uv run python scripts/run_pipeline.py --input file.m4a --step transcribe --primary-model NbAiLab/nb-whisper-large-verbatim

# Kun transkripsjon (lesbar main-modell)
uv run python scripts/run_pipeline.py --input file.m4a --step transcribe --primary-model NbAiLab/nb-whisper-large
```

---

## Filstruktur

```
~/Audio-Transcriber/
├── .venv/                  # Virtuelt miljø (håndteres av uv)
├── src/
│   ├── analyze.py          # Steg 1: Metadata og analyse
│   ├── preprocess.py       # Steg 2: Adaptiv lydforberedelse
│   ├── diarize.py          # Steg 3: Diarization (kjøres én gang)
│   ├── transcribe.py       # Steg 3 & 4: Transkripsjon
│   ├── compare.py          # Steg 5: Align og avviksmarkering
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

2. **Andre mening fremfor «tie-breaker»:** En tredje modell løser sjelden uenighet automatisk, fordi konfidens-score ikke er kalibrert og modeller feiler korrelert. Bruk heller den ekstra modellen til å flagge avvik for menneskelig review (Steg 5).

3. **Word-level timestamps:** Hold `word_timestamps=True` på som default — editoren trenger det for presis synkronisering.

4. **Automatisk stavekontroll:** Kjør norsk stavekontroll (f.eks. via `symspellpy` eller en `transformers`-basert modell) på transkripsjonen før sluttkorrektur for å fange åpenbare feil.

5. **Egendefinert ordliste:** Mat kjente egennavn og fagord inn via Whisper sin `initial_prompt` for å redusere feil på akkurat de ordene som ellers bommer.

6. **Web-basert editor:** Bytt ut ekstern SRT-editor med en innebygget web-editor (f.eks. React + wavesurfer.js) som viser bølgeform, speaker-farger og konfidens-/avviksmarkeringer direkte i nettleseren.

7. **Konfigurasjonsfil:** Flytt alle innstillinger (modellnavn, thresholds, filterparametre) til `config.yaml` slik at pipeline-en er reproduserbar og enkel å justere.

8. **Logging og sporbarhet:** Logg alle kjøringer med parametre, modellversjoner og resultater til en SQLite-database eller JSON-lines-fil. Dette gjør det mulig å sammenligne resultater over tid.

9. **Maskinvareakselerasjon (Apple Silicon):** `faster-whisper` kjører kun på CPU på Mac og bruker verken GPU eller Neural Engine. For betydelig hastighetsøkning kan NB-Whisper konverteres til whisper.cpp (CoreML) eller MLX, som utnytter ANE/Metal. På maskiner med NVIDIA GPU brukes `device="cuda"` med `compute_type="float16"`.

10. **VAD-forhåndsfilter:** Bruk `silero-vad` eller innebygd WhisperX-VAD for å fjerne lange stillhetssegmenter før transkripsjon. Dette sparer tid og reduserer hallusinering.

---

## Ytelse

- **Første kjøring:** Modellen lastes ned (~3 GB) til `~/.cache/huggingface/`
- **Påfølgende kjøringer (faster-whisper, CPU, int8):** På M1 Max er realistisk hastighet ca. **1–3× sanntid** for large-modellen — `faster-whisper` bruker ikke GPU/ANE på Mac
- **Med CoreML/MLX (krever egen konvertering):** ca. **3–10× sanntid** ved å utnytte ANE/Metal
- **Speaker-diarization:** Legger typisk til 20–40 % ekstra tid
- **Batch:** large-modellen er minnetung — hold `--workers` på 2–4 på 32 GB RAM for å unngå swapping

---

## Lisens

MIT

> Ved bruk av NB-Whisper i Norge oppfordrer Nasjonalbiblioteket til å merke output med «Transkribert med NB-Whisper Large», bl.a. for å unngå at framtidige ASR-modeller trenes på maskingenerert tekst.