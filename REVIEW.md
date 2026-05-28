# Strategisk gjennomgang av Audio-Transcriber

**Dato:** 28 May 2026  
**Kilde:** Ekstern AI-gjennomgang (Claude) basert på audit av repoet

---

## Hovedfunn: Prosjektet har drevet fra «transkriber opptakene mine nøyaktig» mot «bygg en transkripsjonsplattform»

Mange av elementene i Phase 5–7 (web-editor, REST API, Docker, flere språk, fine-tuning) er plattformbygging du sannsynligvis aldri trenger for å transkribere dine egne opptak godt. Den røde tråden må være: **mål først, fiks den døde konfigurasjonen, injiser vocabulary — så vurder resten mot tall i stedet for magefølelse.**

---

## 1. Du måler ikke nøyaktighet — det er det største hullet

Hele premisset var «så nær 100 % som mulig», men det finnes ingen måte i repoet å vite hvor nær du er. Uten en fasit er ROADMAP-en ren gjetning — «bedre alignment», «stavekontroll», «andre modell» er alle uverifiserbare påstander.

**Dette bør være oppgave #1, før alt annet:**

1. Transkriber 5–10 minutter av dine *faktiske* opptak manuelt og perfekt → en liten ground-truth-fasit.
2. Regn WER (Word Error Rate) mot pipeline-output med `jiwer`.
3. Nå er hver eneste endring målbar: du ser svart på hvitt om verbatim-modellen slår main, om `initial_prompt` hjelper, om den andre modellen i Steg 4 i det hele tatt bidrar.

**Status:** Ikke påbegynt. Står ikke i roadmap i det hele tatt.

---

## 2. Konfig-knottene er koblet fra (Issue #1) — fiks før du måler

`beam_size`, `word_timestamps`, `condition_on_previous_text` og `initial_prompt` blir tatt imot men aldri sendt til WhisperX. Det betyr at alt du justerer i `config.yaml` i dag ikke har effekt. Dette må fikses *før* ground-truth-målingen, ellers måler du en pipeline der knottene er døde.

**Advarsel:** `condition_on_previous_text=True` gir mer sammenheng, men på samtaleopptak med kryssprat kan en dårlig segment forgifte de neste (hallusinering forplanter seg). Test begge veier mot fasiten — ikke sett-og-glem.

**Status:** Løst i commit `155513b`.

---

## 3. Språkdeteksjon: Bedre fiks enn å bytte til `whisperx`

`analyze.py` brukte `whisper` (OpenAI), mens `pyproject.toml` kun hadde `whisperx`. Fiksen i commit `155513b` byttet til `whisperx.load_model("tiny")`, men dette laster fortsatt en hel ~39 MB-modell kun for språkdeteksjon.

**Bedre fiks:** `faster-whisper` (som WhisperX bygger på) returnerer allerede `info.language` og `info.language_probability` fra `transcribe()`. Bruk den til språkdeteksjon og **fjern `whisper`-importen helt** — da forsvinner avhengighetsgapet i stedet for å vokse.

**Status:** Delvis løst, men kan forbedres.

---

## 4. Issue #10 (device auto-detection) vil krasje slik den er beskrevet

«Auto-detect `mps`/`cuda`» fungerer ikke for transkripsjonen. CTranslate2 (motoren under faster-whisper og WhisperX) **støtter ikke Apple Metal/MPS** — `device="mps"` gir bokstavelig talt `ValueError: unsupported device mps`. På Mac har du bare `cpu`.

**Reelle alternativer:**
- **Behold CPU.** For et personlig verktøy med en håndfull opptak er 1–3× sanntid helt greit. Ikke invester i akselerasjon før volumet faktisk plager deg.
- **Bytt motor** for transkripsjonssteget til whisper.cpp+CoreML eller MLX hvis du vil ha ~10× — men det er en egen motor, ikke et device-flagg, og NB-Whisper må konverteres dit.
- **pyannote** (diarization) *kan* bruke `mps`, siden det er PyTorch. Så `mps`-deteksjon hører hjemme i `diarize.py`, ikke `transcribe.py`.

**Status:** Må revideres i ISSUES.md og ROADMAP.md.

---

## 5. Issue #5 (stereo) — verifiser før du bygger

Samsung sine samtaleopptak er som regel **mono** (mikset kanal), ikke ekte stereo med én part per kanal. Kjør `analyze.py` på de faktiske filene dine og sjekk `has_stereo_separation` *før* du bygger kanal-splitting. Hvis de er mono, er #5 bortkastet arbeid. Du har allerede verktøyet til å avgjøre dette — bruk det.

**Status:** Åpen, men bør verifiseres på faktiske filer først.

---

## 6. Vocabulary via `initial_prompt` (Issue #6) — høyest ROI

Dette er den høyeste ROI-en for nøyaktighet på *dine* opptak: mat inn kjente egennavn, stedsnavn og fagord. `vocabulary.py` finnes allerede; den henger bare på at #1 fikses først.

**Status:** Integrert i commit `155513b` via `--vocabulary-file`.

---

## Anbefalt rekkefølge fremover

1. **Ground-truth + WER-harness** (`jiwer`). Forutsetning for alt annet.
2. **Issue #1** — koble konfig-parametrene til WhisperX-kallet. ✅ Løst.
3. **Issue #2 (forbedret fiks)** — bruk `faster-whisper` sin innebygde språkdeteksjon i stedet for å laste en hel modell.
4. **Issue #6 (vocabulary via `initial_prompt`)** — høyest ROI for nøyaktighet. ✅ Integrert.
5. **Issue #3** — HF-auth-helper. ✅ Løst.
6. **Mål, mål, mål** — kjør WER mot fasiten for hver endring.

---

## Drop eller utsett (over-scope for et personlig verktøy)

| Element | Anbefaling | Begrunnelse |
|---------|-----------|-------------|
| **Web-editor (#8)** | Utsett | Subtitle Edit gjør allerede dette bra og gratis. FastAPI + wavesurfer.js er mye arbeid for marginal gevinst. |
| **DTW-alignment i `compare.py` (#9)** | Utsett | `jiwer` gir ord-nivå diff billig. Ground-truth-harnessen vil fortelle deg om to-modell-sammenligningen i det hele tatt hjelper. |
| **`spell_check.py` autokorrektur** | Unngå | Norsk stavekontroll på Whisper-output kan «rette» korrekte egennavn til feil vanlige ord. Bruk kun til å *flagge*. Prioriter `initial_prompt` (forhindrer feil oppstrøms). |
| **REST API, Docker, svensk/dansk/finsk** | Riktig parkert som «future» | Ikke rør før kjernen leverer. |
| **Full test-suite + CI (Phase 7)** | Delvis | Noen få målrettede tester rundt `compare.py` og konfig-gjennomføringen er verdt det; full CI er overinvestering for et personlig verktøy. |
| **Apple Silicon akselerasjon (#10)** | Utsett / behold CPU | CTranslate2 støtter ikke MPS. whisper.cpp+CoreML krever egen konvertering. 1–3× sanntid på CPU er greit for et personlig verktøy. |

---

## Oppsummert

Den røde tråden: **mål først, fiks den døde konfigurasjonen, injiser vocabulary — så vurder resten mot tall i stedet for magefølelse.** Det meste i Phase 5–7 er plattformbygging du sannsynligvis aldri trenger for å transkribere dine egne opptak godt.
