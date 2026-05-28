# Strategisk gjennomgang av Audio-Transcriber

**Dato:** 28. mai 2026  
**Kontekst:** Oppfølging etter v0.1.1–v0.1.3. Det meste fra forrige runde er adressert; dette dokumentet handler om hva som faktisk gjenstår og hva som bør gjøres nå.

---

## Hovedfunn: Du har bygget målestokken, men ikke målt

Forrige gjennomgang sa: *mål først.* Du har bygget `scripts/evaluate.py` (WER/CER med `jiwer`) — det er målestokken. Men ingenting i repoet tyder på at du faktisk har:

1. Laget en **fasit** (manuelt transkribert 5–10 min av dine *ekte* opptak), eller
2. **Kjørt pipelinen på et reelt opptak** og sett på resultatet.

Roadmap krysser av «Ground-truth + WER-harness ✅», men en harness uten fasit måler ingenting. Å bygge målestokken er ikke det samme som å måle.

**Dette er nå den eneste tingen som betyr noe.** Alt annet — confidence-flagging, stereo-håndtering, knott-tuning — er uverifiserbart inntil du har kjørt loopen én gang på virkelige data. Du har bygget bredt og kompetent, men prosjektet har fortsatt ikke møtt virkeligheten.

> **Konkret, denne uken:** Ta ett ekte opptak. Transkriber 5–10 min av det manuelt og perfekt. Kjør full pipeline på samme opptak. Kjør `evaluate.py`. Se på *hvor* og *hvordan* den feiler. Den ene øvelsen vil omrokere hele resten av roadmapen basert på tall i stedet for antakelser.

---

## Dokumentasjonen har drevet ut av synk

Etter de raske commitene motsier filene nå hverandre. Dette er et reelt problem, fordi dokumentene er kilden til prioritering — er de utdaterte, blir arbeid re-gjort eller feilprioritert.

**Anbefaling:** Gjør `ISSUES.md` til eneste sannhetskilde for status. La `README.md` og `ROADMAP.md` *lenke* til den i stedet for å duplisere «Gjenstående»/«Begrensninger» i flere seksjoner som så råtner.

---

## Tier 1 — Lukk nøyaktighetsloopen (før mer kode)

| Oppgave | Hvorfor |
|---------|---------|
| Lag fasit på ekte opptak | Eneste umålte forutsetning. Uten den er resten gjetning. |
| Kjør full pipeline på samme opptak + `evaluate.py` | Gir første reelle WER-tall. |
| Inspiser feilmodusene | Avgjør om problemet er navn, tall, kryssprat, dialekt eller noe helt annet — det styrer alt videre. |

---

## Tier 2 — `confidence.py` (#14): fullfør, men *valider* den

Designet er ferdig og stubben finnes. Dette er riktig neste kodeoppgave, fordi den treffer review-prioriteringsmålet direkte. To ting er kritiske når du wirer den inn:

### 2a. Valider at prioriteringen faktisk korrelerer med feil

En prioriteringsrangering som ikke er bedre enn tilfeldig er verre enn ingenting — den gir falsk trygghet. Med fasiten kan du måle det:

- Regn per-segment WER mot fasit.
- Mål **rangkorrelasjon** (Spearman) mellom priority-score og segment-WER, eller **precision@k**: av de k høyest flaggede segmentene, hvor mange inneholder faktisk feil?
- Hvis korrelasjonen er svak, er signal-miksen feil — juster før du stoler på den.

Dette er suksesskriteriet som binder #14 til harnessen. Uten det er confidence-flagging bare en udokumentert magefølelse-vekt.

### 2b. «Skråsikkert feil»-gapet trenger en *regel*, ikke bare en advarsel

ISSUES #14 noterer ærlig at confidence bommer på plausible substitusjoner av navn og tall. Men en advarsel i dokumentasjonen fanger ingen feil. Legg inn deterministiske **hard-regler** som flagger uansett score:

- Segmenter som inneholder **tall/siffer** → alltid flagg.
- Segmenter med **stor forbokstav-tokens utenfor vokabularet** (sannsynlige egennavn) → alltid flagg.

Dette treffer nøyaktig feiltypen confidence-scoren ikke kan se.

### 2c. Kjent WhisperX-svakhet som rammer akkurat tallene

WhisperX gir som standard **ikke** ord-nivå timestamps/score for tokens som kun er tall (f.eks. «1,5» eller «2024») — alignment-modellen er fonem-basert. Det betyr at det akustiske «lyd-mot-tekst»-signalet ditt er svakest nettopp der feilene er farligst (tall). Praktisk konsekvens: **ikke stol på alignment-score for numeriske tokens — flagg dem med regel i stedet** (samme som 2b). Verifiser oppførselen på dine faktiske data.

---

## Tier 2b — Tun knottene mot fasiten (mest config, lite kode)

Når fasiten finnes, er disse plutselig målbare i stedet for teoretiske. Kjør WER for hver:

- **Verbatim vs. main** — hvilken gir lavest WER på *dine* opptak?
- **`condition_on_previous_text` på/av** — på kryssprat kan «på» forplante hallusinering. Mål det.
- **Vocabulary på/av** — bekreft at `initial_prompt` faktisk senker WER (forventet høy ROI, men verifiser).
- **Andre modell (Steg 4) på/av** — hvis én god modell + vocabulary allerede gir lav WER, dropp den andre modellen helt.
- **`--spell-check` på/av** — *dette kan gjøre WER verre.* Norsk stavekontroll kan «rette» korrekte egennavn til feil vanlige ord. Nå som den er wiret inn, mål effekten — ikke anta at den hjelper.

> **Interaksjon å være obs på:** verbatim-modellen er små-bokstavert uten tegnsetting. Injiserer du vocabulary med stor forbokstav via `initial_prompt`, kan effekten bli mindre enn ventet, og outputen får uansett ikke kapitaliseringen. Test kombinasjonen.

---

## Tier 3 — Reelle bugs som påvirker korrekthet/bruk

- **#11 (ThreadPoolExecutor + GIL):** Reell bug — tråder parallelliserer ikke CPU-bundet inferens. Men ikke bygg elaborate `ProcessPoolExecutor`-maskineri: hver prosess laster sin egen modellkopi (NB-Whisper int8 ~1,5–3 GB + alignment + pyannote), så på 32 GB får du uansett bare plass til 2–3 samtidig. **Enkleste korrekte fiks:** sett default `--workers 1`, og dokumenter at høyere kun gir mening på CUDA med nok minne. For en håndfull personlige opptak er sekvensiell kjøring helt greit.
- **#4 (`segmentation_model` ignoreres):** pyannote 3.1-pipelinen bundler egen segmentering; å overstyre den er ikke nødvendigvis støttet rent. Lavt prioritert — enten wire det eller **dokumenter at feltet ikke er konfigurerbart** og fjern det fra `config.yaml` så det ikke villeder.

---

## Tier 4 — Verifiser før du bygger

- **#5 (stereo):** Fortsatt ikke verifisert. Kjør `analyze.py` på de faktiske filene og sjekk `has_stereo_separation`. Samsung-opptak er typisk mono — er de det, lukk #5 uten å skrive en linje kode.

---

## Tier 5 — Korrekt utsatt (ingen endring)

Web-editor (#8), DTW-alignment (#9), Apple Silicon-akselerasjon utover CPU (#10), full CI (Phase 7), REST API / Docker / flerspråk. Alt riktig parkert. Ikke rør før kjernen leverer målt lav WER.

---

## Hvis du bare gjør tre ting

1. **Lag fasit + kjør pipelinen på ett ekte opptak + mål WER.** Alt annet venter på dette.
2. **Fullfør `confidence.py` med valideringsmetrikk (Spearman/precision@k) + hard-regler for tall og egennavn.**
3. **Reconciler dokumentasjonen** til én sannhetskilde (`ISSUES.md`), og lukk #12.

Den røde tråden er den samme som sist, bare skarpere nå: du har sluttet å mangle verktøy og begynt å mangle *data*. Mål på ekte opptak, så lar du tallene bestemme resten.
