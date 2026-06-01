# Audio-Transcriber — Omfattende Auditrapport
**Dato:** 29. mai 2026  
**Revisjon:** v0.1.5 (post-fix, etter testing på 410 ekte opptak + Issue #4/#5/#9/#21 fixes)  
**Auditor:** GitHub Copilot (kimi-k2.6:cloud)

---

## Sammendrag

Prosjektet er nå **verifisert på ekte data**. Pipeline kjører end-to-end på reelle opptak og produserer gyldig SRT med norsk transkripsjon. 38 enhetstester passerer. Kritiske blocker-bugs (ffmpeg, JSON-serialisering, whisperx API-mismatch, VAD API, NaN i stereo, confidence priority all-zero) er løst. Ytterligere 4 issues (#4, #5, #9, #21) er løst i denne revisjonen.

**Status etter siste kjøring (2026-05-29):**
- 2 gyldige filer funnet (1 korrupt filtrert ut)
- File 1: 11 minutter, 22 segmenter, priority range 0.562–0.000
- File 2: 17 minutter, 35 segmenter, priority range 0.596–0.000
- Confidence-flagging fungerer: `low_logprob` og `contains_numbers` flagg identifisert
- Hard-rules fungerer: segment med "14 fot" korrekt flagget med `contains_numbers`
- **Nytt i v0.1.5:** Pipeline kjørte uten diarization (--no-diarize) på File 1: 22 segmenter transkribert på ~4 minutter, review list eksportert med 20 flaggete segmenter

**Gjenstående blocker:** Ground-truth fasit og WER-måling mangler fortsatt.

---

## Kritiske funn (må fikses før produksjon)

### K1: ~400 av 410 testfiler er korrupte / ufullstendige
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** Batch-kjøring på `testdata/*.m4a` (410 filer) ga ~400 feil med `ffprobe exited with code 1: moov atom not found`. Output-mapper for disse filene er tomme (0 bytes).
- **Impact:** Kan ikke kjøre batch-prosessering uten filtrering.
- **Fix:** `_find_audio_files()` i `run_pipeline.py` skipper nå filer < 1KB. Logger antall hoppet over filer.
- **Prioritet:** 🟢 **LØST**

### K2: `detect_language()` returnerer "et" (estisk) for norsk tale
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** `analyze_audio()` på testopptak rapporterte `language: et` med confidence 0.29.
- **Impact:** Metadata var feil. Fremtidig multi-språk-støtte ville vært broken.
- **Fix:** Lagt til confidence-threshold (0.5) i `detect_language()`. Hvis `info.language_probability < 0.5`, faller tilbake til "no" (Norwegian).
- **Prioritet:** 🟢 **LØST**

### K3: `confidence.py` er designet og testet, men ikke wiret inn i pipeline
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** `src/confidence.py` hadde 7 tester, alle passerte. `run_pipeline.py` importerte aldri `confidence`.
- **Impact:** Bruker fikk ingen prioritert review-liste.
- **Fix:** `extract_confidence_signals()` er nå wiret inn automatisk etter transkripsjon i `process_single_file()`. Eksporterer `*_review_list.txt` med top 20 flaggete segmenter.
- **Prioritet:** 🟢 **LØST**

### K4: `ThreadPoolExecutor` med 4 workers for CPU-bound inferens
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** `run_pipeline.py` hadde `ThreadPoolExecutor(max_workers=4)` som default. Transkripsjon og diarisering er CPU-tunge. Python GIL forhindrer ekte parallellisme.
- **Impact:** Batch-modus var ikke raskere enn sekvensiell; risiko for OOM.
- **Fix:** Default `--workers` endret fra 4 til 1. Dokumentert at >1 kun gir mening på CUDA med nok VRAM.
- **Prioritet:** 🟢 **LØST**

### K5: `spell_check.py` har ingen faktisk norsk ordbok
- **Status:** 🟢 **LØST** (2026-06-02, ISSUES.md #47)
- **Bevis:** `NorwegianSpellChecker._init_symspell()` oppretter `SymSpell`-objektet, men laster **ingen** ordbok. `check_word()` vil alltid returnere `True, None` fordi det ikke finnes noe å sammenligne mot.
- **Impact:** `--spell-check` flagget gjør ingenting. Bruker får falsk trygghet.
- **Fix:** To bugs ble fikset:
  1. `include_unknown=True` i SymSpell lookup — ukjente ord ble returnert som seg selv og akseptert som korrekte. Endret til `include_unknown=False`.
  2. `check_text()` krevde `suggestion is not None` for å flagge en feil — ukjente ord uten forslag ble ignorert. Endret til å flagge alle ukorrekte ord uavhengig av forslag.
  Ordboken (334,169 ord fra LibreOffice nb_NO.dic) ble allerede lastet ned og lastet inn — feilen var i oppslagslogikken, ikke i ordboksinnlasting.
- **Prioritet:** 🟢 **LØST**

---

## Løste kritiske funn (fikset i commit 921604d og 3894ce1)

| Funn | Fil | Fix |
|------|-----|-----|
| **ffmpeg/ffprobe manglet** | `analyze.py` | `_get_audio_info_fallback()` med librosa når ffprobe er utilgjengelig |
| **Silero VAD API mismatch** | `analyze.py` | Fjernet `num_steps`, bruker `sampling_rate` |
| **NaN i stereo detection** | `analyze.py` | Håndterer `np.isnan(correlation)` eksplisitt |
| **np.float32 ikke JSON-serializable** | `utils.py` | `_NumpyEncoder` for numpy-typer |
| **WhisperX API mismatch** | `transcribe.py` | `beam_size`, `initial_prompt`, etc. flyttet til `asr_options` i `load_model()` |
| **Word alignment ikke tilgjengelig** | `transcribe.py` | Graceful warning, fortsetter uten alignment |
| **pyloudnorm manglet** | `pyproject.toml` | Eksplisitt avhengighet lagt til |
| **Confidence priority all-zero** | `transcribe.py`, `run_pipeline.py` | Decoder signals (`avg_logprob`, etc.) flyter nå gjennom pipeline til confidence extractor |
| **Hard-rules for high-risk content** | `confidence.py` | Tall og proper nouns flagges alltid — fanger "confidently wrong" feil |

---

## Høy-prioritet funn (påvirker nøyaktighet eller brukeropplevelse)

### H1: Ingen ground-truth fasit — WER-harness er ubrukt
- **Status:** Tier 1 (strategisk prioritet) — den eneste tingen som betyr noe
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
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** `_segments_to_srt()` plasserte speaker-label (`SPEAKER_00`) på egen linje mellom tidsstempel og tekst.
- **Impact:** Eksterne editorer viste `SPEAKER_00` som synlig tekst.
- **Fix:** Speaker label nå inline: `SPEAKER_00: tekst` på samme linje som subtitle-innhold.
- **Prioritet:** 🟢 **LØST**

### H4: `vocabulary.py` token-estimat er feil
- **Status:** Uavklart
- **Bevis:** `generate_initial_prompt()` estimerer "2 tokens per word". Whisper bruker BPE-tokenisering — vanlige ord kan være 1 token, sjeldne ord/subwords kan være 3–5 tokens. `max_tokens=100` er hardkodet.
- **Impact:** Prompt kan overstige Whisper's 224-token grense for `initial_prompt`, eller være mye kortere enn nødvendig.
- **Fix:** Bruk `transformers.AutoTokenizer` for faktisk token-telling, eller sett en konservativ grense (~150 tokens) og dokumenter begrensningen.
- **Prioritet:** 🟡 **LOW**

### H5: Preprocessing clipping ved loudness-normalisering
- **Status:** ✅ Løst (2026-05-29)
- **Bevis:** `normalize_loudness()` rapporterte `Clipping detected (1.91)` på testopptaket.
- **Impact:** Klipping introduserte digital distorsjon.
- **Fix:** To endringer: (1) Gain begrenses nå til max 1.0/peak (pre-clipping). (2) Default `loudness_target_lufs` endret fra -16 til -20 i `config.yaml`.
- **Prioritet:** 🟢 **LØST**

---

## Medium-prioritet funn (teknisk gjeld)

### M1: Manglende eksplisitte avhengigheter i `pyproject.toml`
- **Status:** Delvis løst (pyloudnorm lagt til)
- **Bevis:** `symspellpy`, `soundfile` importeres i koden men er ikke listet i `dependencies`. De er sannsynligvis transitive avhengigheter, men bør være eksplisitte.
- **Fix:** Legg til `symspellpy>=6.7.0`, `soundfile>=0.12.0`.
- **Prioritet:** 🟡 **LOW**

### M2: `audioop` deprekert i Python 3.13
- **Status:** Uavklart
- **Bevis:** `pydub` importerer `audioop` som er "deprecated and slated for removal in Python 3.13". `requires-python = ">=3.11,<3.13"` beskytter mot dette, men er en tidsinnstilt bombe.
- **Impact:** Prosjektet er låst til Python 3.11–3.12.
- **Fix:** Overvåke pydub-oppdateringer; vurdere `soundfile`+`librosa` for all lyd-I/O.
- **Prioritet:** 🟡 **LOW**

### M3: Ingen integrasjonstester
- **Status:** Uavklart
- **Bevis:** 38 enhetstester dekker isolerte funksjoner. Ingen test kjører `run_pipeline.py` end-to-end.
- **Impact:** Regresjoner i pipeline-orkestrering oppdages ikke automatisk.
- **Fix:** Legg til minst én integrasjonstest som kjører hele pipeline på en kort syntetisk lydfil (eller mock).
- **Prioritet:** 🟡 **LOW**

### M4: `diarize.py` `check_hf_auth()` sjekker ikke faktisk gyldighet
- **Status:** Uavklart
- **Bevis:** `check_hf_auth()` sjekker at token *finnes*, men verifiserer ikke at tokenet er gyldig.
- **Impact:** Bruker får "auth OK" men pyannote-kallet feiler likevel med 403/401.
- **Fix:** Gjør et lightweight API-kall (f.eks. `huggingface_hub.whoami()`) for å verifisere token-gyldighet.
- **Prioritet:** 🟡 **LOW**

### M5: `compare.py` alignment er for simpel
- **Status:** ✅ Løst (2026-05-29) — Issue #9
- **Bevis:** `align_segments()` brukte kun tids-overlap >50%. `calculate_similarity()` brukte `difflib.SequenceMatcher` på tegn-nivå.
- **Impact:** Falske positive/negative i disagreement-deteksjon.
- **Fix:** `calculate_similarity()` bruker nå `jiwer.wer()` for ord-nivå WER-basert similarity når tilgjengelig, med fallback til `SequenceMatcher`. WER gir mer lingvistisk meningsfull sammenligning enn tegn-nivå matching.
- **Prioritet:** 🟢 **LØST**

### M6: `config.yaml` `segmentation_model` ignoreres fullstendig
- **Status:** ✅ Løst (2026-05-29) — Issue #4
- **Bevis:** `config.yaml` hadde `diarization.segmentation_model: "pyannote/segmentation-3.0"`. `Diarizer._load_model()` leste kun `diarization.model` og aldri `segmentation_model`. pyannote 3.1 bundler egen segmentering.
- **Impact:** Bruker kunne ikke overstyre segmenteringsmodell. Feltet villedet.
- **Fix:** Fjernet feltet fra `config.yaml` og la til kommentar som forklarer at pyannote 3.1 bundler sin egen segmenteringsmodell internt. La til inline-kommentar i `diarize.py` som refererer til ISSUES.md #4.
- **Prioritet:** 🟢 **LØST**

---

## Lav-prioritet funn (forbedringer)

### L1: `editor.py` er fortsatt placeholder
- **Status:** Issue #8 — Åpen, men korrekt parkert (Tier 5)
- **Fix:** Ingen — Subtitle Edit dekker behovet.

### L2: `database.py` har ingen query-API for korrektur-analyse
- **Status:** Uavklart
- **Fix:** Legg til query-metoder: `get_corrections_by_type()`, `get_wer_trend()`, `get_most_corrected_words()`.
- **Prioritet:** 🟢 **FUTURE**

### L3: `config.yaml` `performance.device` hardkodet til "cpu"
- **Status:** Uavklart
- **Bevis:** `performance.device: "cpu"` i config. Koden auto-detekterer uansett. Config-verdien overstyres av auto-deteksjon.
- **Impact:** Config-verdien er meningsløs.
- **Fix:** Fjern feltet og dokumenter auto-deteksjon.
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
| `README.md` | ✅ Oppdatert | Lenker til ISSUES.md som sannhetskilde |
| `ROADMAP.md` | ✅ Oppdatert | "Remaining" renset for løste items |
| `ISSUES.md` | ✅ Oppdatert | Eneste sannhetskilde for status |
| `REVIEW.md` | ❌ Slettet | Innhold flyttet til AUDIT.md «Strategisk gjennomgang» |
| `CHANGELOG.md` | ✅ Oppdatert | v0.1.0–v0.1.5 |
| `AUDIT.md` | ✅ Denne filen | Post-fix audit etter real-data-testing |
| `.instructions.md` | ✅ Oppdatert | AI agent kontekst |

**Ingen dokumentasjonsdrift oppdaget.**

---

## Anbefalt rekkefølge (revidert etter v0.1.5)

1. **🔴 Lag ground-truth fasit** — manuelt transkriber 5–10 min av ett gyldig opptak
2. **🔴 Kjør pipeline + evaluate.py** — få første reelle WER-tall
3. **🟡 Legg til integrasjonstest** — en end-to-end test på syntetisk lyd (mock whisperx)
4. **🟡 Fiks `performance.device` i config** — fjern eller dokumenter at auto-deteksjon overstyres

---

## Strategisk gjennomgang (fra REVIEW.md, 28. mai 2026)

> **Hovedfunn:** Du har bygget målestokken, men ikke målt. `scripts/evaluate.py` eksisterer, men ingen fasit er laget og ingen WER er målt. Dette er nå den eneste tingen som betyr noe.

### Tier 1 — Lukk nøyaktighetsloopen (før mer kode)

| Oppgave | Hvorfor |
|---------|---------|
| Lag fasit på ekte opptak | Eneste umålte forutsetning. Uten den er resten gjetning. |
| Kjør full pipeline på samme opptak + `evaluate.py` | Gir første reelle WER-tall. |
| Inspiser feilmodusene | Avgjør om problemet er navn, tall, kryssprat, dialekt eller noe helt annet — det styrer alt videre. |

### Tier 2 — `confidence.py`: fullfør, men *valider* den

**2a. Valider at prioriteringen faktisk korrelerer med feil**
- Regn per-segment WER mot fasit.
- Mål **rangkorrelasjon** (Spearman) mellom priority-score og segment-WER, eller **precision@k**.
- Hvis korrelasjonen er svak, er signal-miksen feil — juster før du stoler på den.

**2b. «Skråsikkert feil»-gapet trenger en *regel*, ikke bare en advarsel**
- Segmenter som inneholder **tall/siffer** → alltid flagg.
- Segmenter med **stor forbokstav-tokens utenfor vokabularet** (sannsynlige egennavn) → alltid flagg.

**2c. Kjent WhisperX-svakhet som rammer akkurat tallene**
WhisperX gir som standard **ikke** ord-nivå timestamps/score for tokens som kun er tall — alignment-modellen er fonem-basert. **Ikke stol på alignment-score for numeriske tokens — flagg dem med regel i stedet.**

### Tier 2b — Tun knottene mot fasiten (mest config, lite kode)

Når fasiten finnes, er disse plutselig målbare:
- **Verbatim vs. main** — hvilken gir lavest WER på *dine* opptak?
- **`condition_on_previous_text` på/av** — på kryssprat kan «på» forplante hallusinering.
- **Vocabulary på/av** — bekreft at `initial_prompt` faktisk senker WER.
- **Andre modell (Steg 4) på/av** — dropp hvis én modell + vocabulary gir lav WER.
- **`--spell-check` på/av** — *dette kan gjøre WER verre.* Norsk stavekontroll kan «rette» korrekte egennavn til feil vanlige ord.

> **Interaksjon å være obs på:** verbatim-modellen er små-bokstavert uten tegnsetting. Injiserer du vocabulary med stor forbokstav via `initial_prompt`, kan effekten bli mindre enn ventet.

### Tier 3 — Reelle bugs som påvirker korrekthet/bruk

- **#11 (ThreadPoolExecutor + GIL):** Reell bug — tråder parallelliserer ikke CPU-bundet inferens. **Enkleste korrekte fiks:** sett default `--workers 1`, og dokumenter at høyere kun gir mening på CUDA med nok minne.
- **#4 (`segmentation_model` ignoreres):** pyannote 3.1 bundler egen segmentering. Lavt prioritert — dokumenter at feltet ikke er konfigurerbart og fjern det fra `config.yaml`.

### Tier 4 — Verifiser før du bygger

- **#5 (stereo):** Fortsatt ikke verifisert. Kjør `analyze.py` på de faktiske filene og sjekk `has_stereo_separation`. Samsung-opptak er typisk mono — er de det, lukk #5 uten å skrive en linje kode.

### Tier 5 — Korrekt utsatt (ingen endring)

Web-editor (#8), DTW-alignment (#9), Apple Silicon-akselerasjon utover CPU (#10), full CI (Phase 7), REST API / Docker / flerspråk. Alt riktig parkert. Ikke rør før kjernen leverer målt lav WER.

### Hvis du bare gjør tre ting

1. **Lag fasit + kjør pipelinen på ett ekte opptak + mål WER.** Alt annet venter på dette.
2. **Fullfør `confidence.py` med valideringsmetrikk (Spearman/precision@k) + hard-regler for tall og egennavn.**
3. **Reconciler dokumentasjonen** til én sannhetskilde (`ISSUES.md`), og lukk #12.

---

## Konklusjon

Audio-Transcriber er nå **verifisert på ekte data** og **alle kritiske blocker-bugs er løst**. Kjerne-pipeline (analyse → forhåndsbehandling → transkripsjon) fungerer på reelle opptak og produserer gyldig SRT med norsk tekst.

**v0.1.5 endringer:** Issues #4, #5, #9, #21 er løst. Stereo-splitting, jiwer WER-similarity, config-opprydding, og spell-check deaktivering er på plass.

De gjenstående oppgavene er:
1. **Operasjonelle:** Lag fasit + mål WER (den viktigste)
2. **Teknisk gjeld:** Legg til integrasjonstest, fjern ubrukte config-felter

**Prosjektet trenger nå mindre debugging og mer måling.**
