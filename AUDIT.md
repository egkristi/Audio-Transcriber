# Audio-Transcriber — Omfattende Auditrapport
**Dato:** 28. mai 2026  
**Revisjon:** v0.1.3  
**Auditor:** GitHub Copilot (kimi-k2.6:cloud)

---

## Sammendrag

Prosjektet er en solid proof-of-concept med god arkitektur og modulær design. 38 enhetstester passerer. Kjerne-pipeline (analyse → forhåndsbehandling → diarisering → transkripsjon) er funksjonell. **Men:** prosjektet har ennå ikke møtt virkelige data — `ffmpeg` mangler, ingen ground-truth fasit eksisterer, og `evaluate.py` (WER-harness) har aldri blitt kjørt på et reelt opptak. Dette er den kritiske gjenstående oppgaven.

---

## Kritiske funn (må fikses før produksjon)

### K1: `ffmpeg` / `ffprobe` er ikke installert
- **Status:** Blokkerer all pipeline-kjøring på testdata
- **Bevis:** `which ffmpeg` → `not found`; `analyze_audio()` på testdata krasjer med `FileNotFoundError: ffprobe`
- **Impact:** Hele pipeline avhengig av ffmpeg (ffprobe for metadata, pydub for lasting, librosa for analyse)
- **Fix:** `brew install ffmpeg` (allerede dokumentert i README, men ikke utført)
- **Prioritet:** 🔴 **BLOCKER**

### K2: `confidence.py` er designet og testet, men ikke wiret inn i pipeline
- **Status:** Issue #14 — Åpen
- **Bevis:** `src/confidence.py` har 7 tester, alle passerer. `run_pipeline.py` importerer aldri `confidence`. Ingen `--confidence` CLI-flagg.
- **Impact:** Bruker får ingen prioritert review-liste. Manuell korrektur er uniform i stedet for fokusert.
- **Fix:** Wire `extract_confidence_signals()` inn etter transkripsjon i `process_single_file()`. Legg til `--export-confidence` CLI-flagg.
- **Prioritet:** 🔴 **HØY**

### K3: `ThreadPoolExecutor` med 4 workers for CPU-bound inferens
- **Status:** Issue #11 — Åpen
- **Bevis:** `run_pipeline.py` linje ~320: `ThreadPoolExecutor(max_workers=workers)` der default er 4. Transkripsjon og diarisering er CPU-tunge. Python GIL forhindrer ekte parallellisme.
- **Impact:** Batch-modus er ikke raskere enn sekvensiell; flere modellkopier i minne kan gi OOM (nb-whisper large ~3GB + pyannote ~1GB per tråd).
- **Fix:** Sett default `--workers 1` for CPU-only. Dokumenter at >1 kun gir mening på CUDA med nok VRAM.
- **Prioritet:** 🟠 **MEDIUM**

### K4: `spell_check.py` har ingen faktisk norsk ordbok
- **Status:** Uavklart — ikke sporet i ISSUES.md
- **Bevis:** `NorwegianSpellChecker._init_symspell()` oppretter `SymSpell`-objektet, men laster **ingen** ordbok. `check_word()` vil alltid returnere `True, None` fordi det ikke finnes noe å sammenligne mot.
- **Impact:** `--spell-check` flagget gjør ingenting. Bruker får falsk trygghet.
- **Fix:** Enten (a) last en norsk ordbok (f.eks. fra NST/UiB), eller (b) fjern `--spell-check` fra CLI inntil ordbok er på plass, eller (c) bruk `transformers`-basert modell som faktisk fungerer.
- **Prioritet:** 🟠 **MEDIUM**

### K5: `config.yaml` `segmentation_model` ignoreres fullstendig
- **Status:** Issue #4 — Åpen
- **Bevis:** `config.yaml` har `diarization.segmentation_model: "pyannote/segmentation-3.0"`. `Diarizer._load_model()` leser kun `diarization.model` og aldri `segmentation_model`. pyannote 3.1 bundler egen segmentering.
- **Impact:** Bruker kan ikke overstyre segmenteringsmodell. Feltet villeder.
- **Fix:** Enten (a) pass `segmentation_model` til pyannote Pipeline hvis API støtter det, eller (b) fjern feltet fra `config.yaml` og dokumenter at det ikke er konfigurerbart.
- **Prioritet:** 🟡 **LOW**

---

## Høy-prioritet funn (påvirker nøyaktighet eller brukeropplevelse)

### H1: Ingen ground-truth fasit — WER-harness er ubrukt
- **Status:** REVIEW.md Tier 1 — den eneste tingen som betyr noe
- **Bevis:** `scripts/evaluate.py` eksisterer og er komplett (jiwer-basert). Ingen `testdata/*.txt` fasit-fil finnes. Ingen WER-måling har blitt kjørt.
- **Impact:** All "forbedring" er uverifiserbar. Man vet ikke om vocabulary injection, spell-check, eller model A vs B faktisk senker WER.
- **Fix:** Manuelt transkriber 5–10 minutter av `testdata/*.m4a`. Lagre som `testdata/fasit.txt`. Kjør pipeline + `evaluate.py`. Dette er nå den viktigste oppgaven.
- **Prioritet:** 🔴 **BLOCKER for all videre tuning**

### H2: `preprocess.py` laster lyd to ganger
- **Status:** Uavklart
- **Bevis:** `analyze_audio()` kjører `librosa.load(..., mono=True)`. Deretter `preprocess_audio()` kjører `librosa.load(..., mono=False)`. Samme fil lastes to ganger.
- **Impact:** Unødvendig I/O og minnebruk. For 11MB fil er det ubetydelig, men for batch av lange opptak er det merkbart.
- **Fix:** Cache `audio_data` i `metadata`-objektet, eller pass det eksplisitt fra analyse til forhåndsbehandling.
- **Prioritet:** 🟡 **LOW**

### H3: SRT-format i `transcribe.py` er ugyldig
- **Status:** Uavklart
- **Bevis:** `_segments_to_srt()` legger speaker-label (`SPEAKER_00`) på egen linje mellom tidsstempel og tekst. Gyldig SRT har kun én tekstlinje (eller flere) etter tidsstempel. Speaker-label på egen linje tolkes som tekst av de fleste SRT-parserere.
- **Impact:** Eksterne editorer (Subtitle Edit, VLC, etc.) viser `SPEAKER_00` som tekst i stedet for metadata.
- **Fix:** Formatér som `SPEAKER_00: tekst` på samme linje, eller bruk SRT-cue settings (`<v SPEAKER_00>tekst`) hvis editor støtter det.
- **Prioritet:** 🟠 **MEDIUM**

### H4: `vocabulary.py` token-estimat er feil
- **Status:** Uavklart
- **Bevis:** `generate_initial_prompt()` estimerer "2 tokens per word". Whisper bruker BPE-tokenisering — vanlige ord kan være 1 token, sjeldne ord/subwords kan være 3–5 tokens. `max_tokens=100` er hardkodet.
- **Impact:** Prompt kan overstige Whisper's 224-token grense for `initial_prompt`, eller være mye kortere enn nødvendig.
- **Fix:** Bruk `transformers.AutoTokenizer` for faktisk token-telling, eller sett en konservativ grense (~150 tokens) og dokumenter begrensningen.
- **Prioritet:** 🟡 **LOW**

### H5: `detect_stereo_separation()` bruker `np.corrcoef` med identiske kanaler gir NaN
- **Status:** Uavklart — test passerer med advarsel
- **Bevis:** Test `test_identical_channels_returns_false` triggerer `RuntimeWarning: invalid value encountered in divide` fra numpy. Når to kanaler er identiske (`np.ones`), standardavviket er 0, og korrelasjon blir NaN.
- **Impact:** For mono-opptak som konverteres til stereo (identiske kanaler), returnerer funksjonen potensielt `NaN` i stedet for `False`.
- **Fix:** Håndter `stddev == 0` tilfelle eksplisitt: returner `False` (ingen separasjon) når korrelasjon er NaN.
- **Prioritet:** 🟠 **MEDIUM**

---

## Medium-prioritet funn (teknisk gjeld)

### M1: Manglende eksplisitte avhengigheter i `pyproject.toml`
- **Status:** Uavklart
- **Bevis:** `pyloudnorm`, `symspellpy`, `soundfile` importeres i koden men er ikke listet i `dependencies`. De er sannsynligvis transitive avhengigheter (via librosa, whisperx), men bør være eksplisitte.
- **Impact:** Potensiell `ModuleNotFoundError` på friske miljøer hvis transitiv avhengighet endres.
- **Fix:** Legg til i `pyproject.toml`: `pyloudnorm>=0.1.0`, `symspellpy>=6.7.0`, `soundfile>=0.12.0`.
- **Prioritet:** 🟡 **LOW**

### M2: `audioop` deprekert i Python 3.13
- **Status:** Uavklart
- **Bevis:** `pydub` importerer `audioop` som er "deprecated and slated for removal in Python 3.13". `requires-python = ">=3.11,<3.13"` i `pyproject.toml` beskytter mot dette, men er en tidsinnstilt bombe.
- **Impact:** Prosjektet er låst til Python 3.11–3.12. Oppgradering til 3.13 krever pydub-erstatning.
- **Fix:** Overvåke pydub-oppdateringer; vurdere `pydub`-alternativ (f.eks. `ffmpeg-python` direkte, eller `soundfile`+`librosa` for all lyd-I/O).
- **Prioritet:** 🟡 **LOW**

### M3: Ingen integrasjonstester
- **Status:** Uavklart
- **Bevis:** 38 enhetstester dekker isolerte funksjoner. Ingen test kjører `run_pipeline.py` end-to-end. Ingen test verifiserer at `analyze → preprocess → transcribe` fungerer sammen.
- **Impact:** Regresjoner i pipeline-orkestrering oppdages ikke automatisk.
- **Fix:** Legg til minst én integrasjonstest som kjører hele pipeline på en kort syntetisk lydfil (eller mock).
- **Prioritet:** 🟡 **LOW**

### M4: `diarize.py` `check_hf_auth()` sjekker ikke faktisk gyldighet
- **Status:** Uavklart
- **Bevis:** `check_hf_auth()` sjekker at token *finnes* (fil eller env-var), men verifiserer ikke at tokenet er gyldig (ikke utløpt, ikke revoked, har tilgang til pyannote-modellen).
- **Impact:** Bruker får "auth OK" men pyannote-kallet feiler likevel med 403/401.
- **Fix:** Gjør et lightweight API-kall (f.eks. `huggingface_hub.whoami()`) for å verifisere token-gyldighet.
- **Prioritet:** 🟡 **LOW**

### M5: `compare.py` alignment er for simpel
- **Status:** Issue #9 — Åpen
- **Bevis:** `align_segments()` bruker kun tids-overlap >50%. Ingen ord-nivå WER-diff. To segmenter med samme tidsvindu men ulik tekst anses som "match".
- **Impact:** Falske positive/negative i disagreement-deteksjon.
- **Fix:** Bruk `jiwer.process_words()` for ord-nivå diff innenfor overlappende segmenter. Dette er billigere enn DTW og mer nøyaktig enn overlap.
- **Prioritet:** 🟡 **LOW**

---

## Lav-prioritet funn (forbedringer)

### L1: `editor.py` er fortsatt placeholder
- **Status:** Issue #8 — Åpen, men korrekt parkert per REVIEW.md
- **Bevis:** Kun SRT-eksport og print-instruksjoner. Ingen web-UI.
- **Impact:** Bruker må bruke ekstern editor. Subtitle Edit dekker behovet.
- **Fix:** Ingen — korrekt parkert som "future".

### L2: `database.py` har ingen query-API for korrektur-analyse
- **Status:** Uavklart
- **Bevis:** `TranscriptionDatabase` logger jobber og transkripsjoner, men har ingen metode for å hente "mest korrigerte ord" eller "WER per modell over tid".
- **Impact:** Feedback-loop for modell-forbedring er manuell.
- **Fix:** Legg til query-metoder: `get_corrections_by_type()`, `get_wer_trend()`, `get_most_corrected_words()`.
- **Prioritet:** 🟢 **FUTURE**

### L3: `config.yaml` `performance.device` hardkodet til "cpu"
- **Status:** Uavklart
- **Bevis:** `performance.device: "cpu"` i config. Koden auto-detekterer uansett (transcribe.py sjekker `torch.cuda.is_available()`, diarize.py sjekker `cuda`/`mps`). Config-verdien overstyres av auto-deteksjon.
- **Impact:** Config-verdien er meningsløs. Bruker kan ikke overstyre via config.
- **Fix:** Enten respekter config-verdien (med validering), eller fjern feltet og dokumenter auto-deteksjon.
- **Prioritet:** 🟢 **FUTURE**

---

## Testdekning

| Modul | Tester | Dekning | Kommentar |
|-------|--------|---------|-----------|
| `analyze.py` | 8 | Delvis | `detect_language`, `detect_speech_vad`, `calculate_loudness` ikke testet (krever tunge modeller) |
| `preprocess.py` | 8 | God | Alle hovedfunksjoner testet med syntetisk lyd |
| `compare.py` | 7 | God | `SequenceMatcher`-basert, enkel men dekket |
| `diarize.py` | 5 | Delvis | `_load_model`, `diarize` ikke testet (krever HF-auth og modell) |
| `confidence.py` | 7 | God | Alle signaler og prioritering testet |
| **Total** | **38** | **~60%** | Mangler: integrasjonstester, modell-lastetester, end-to-end |

---

## Dokumentasjonsstatus

| Fil | Status | Kommentar |
|-----|--------|-----------|
| `README.md` | ✅ Oppdatert | Lenker til ISSUES.md som sannhetskilde. Ingen dupliserte "Begrensninger"-seksjoner lenger. |
| `ROADMAP.md` | ✅ Oppdatert | "Remaining" renset for løste items. ISSUES.md referanse på plass. |
| `ISSUES.md` | ✅ Oppdatert | #12 lukket, #10 merget inn i #13. Eneste sannhetskilde for status. |
| `REVIEW.md` | ✅ Oppdatert | Tier 1–5 prioritering med "mål først"-budskap. |
| `CHANGELOG.md` | ✅ Oppdatert | v0.1.0–v0.1.3 med alle endringer. |

**Ingen dokumentasjonsdrift oppdaget.** Alle filer er konsistente og peker til ISSUES.md.

---

## Anbefalt rekkefølge (basert på REVIEW.md + denne auditen)

1. **🔴 Installer ffmpeg** (`brew install ffmpeg`) — blocker for all testing
2. **🔴 Lag ground-truth fasit** — manuelt transkriber 5–10 min av testdata-opptaket
3. **🔴 Kjør pipeline + evaluate.py** — få første reelle WER-tall
4. **🔴 Wire confidence.py inn i pipeline** — høy ROI for review-prioritering
5. **🟠 Fiks SRT-format** — påvirker alle eksterne editorer
6. **🟠 Sett `--workers 1` default** — unngå OOM og falsk parallellisme
7. **🟠 Fiks spell_check.py** — enten last ordbok eller fjern flagget
8. **🟡 Fiks `detect_stereo_separation()` NaN-håndtering** — robusthet
9. **🟡 Fiks `segmentation_model` i config** — dokumenter eller fjern
10. **🟡 Legg til integrasjonstest** — en end-to-end test på syntetisk lyd

---

## Konklusjon

Audio-Transcriber er et godt designet proof-of-concept med solid arkitektur og god testdekning for enhetstester. De største gjenstående oppgavene er **operasjonelle, ikke kode-messige**: installer ffmpeg, lag en fasit, og kjør WER-måling. Når det er på plass, er `confidence.py` den høyest ROI kode-oppgaven. Alt annet — stereo-håndtering, web-editor, CI — er korrekt parkert.

**Prosjektet trenger nå mindre kode og mer måling.**
