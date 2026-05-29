# AGENT.md — Audio-Transcriber AI Agent Instructions

> Read this file first. Then read `AUDIT.md`, `ISSUES.md`, and the latest `CHANGELOG.md` entry. **Verify the current state against the actual code before trusting any document — drift has been observed in this repo, including in audit reports.**

---

## 1. Mission

Drive the project toward measurably accurate Norwegian transcription of real call recordings on the operator's machine. Strategic guidance lives in `AUDIT.md` §«Strategisk gjennomgang» and is canonical.

The non-negotiable goal: a fasit (ground-truth transcript) plus a baseline WER number on a real recording. Without those, every "improvement" is speculation.

---

## 2. Project Identity

- **Name:** Audio-Transcriber
- **Purpose:** Norwegian audio transcription pipeline with speaker diarization, multi-model comparison, and confidence-based review prioritization
- **Primary language:** Norwegian (Bokmål/Nynorsk). Language detection currently falls back to `"no"` when confidence < 0.5 — a deliberate hardcode for a Norwegian-only tool.
- **License:** MIT
- **Repository:** https://github.com/egkristi/Audio-Transcriber

---

## 3. Tech Stack

- **Python:** 3.11, locked to `<3.13` due to `audioop` deprecation in `pydub` (tracked tech debt — AUDIT M2)
- **Package manager:** `uv` (never `pip` directly — bypasses lockfile)
- **OS:** macOS primary; Linux/Windows for CUDA paths
- **Core dependencies:** `whisperx`, `pyannote.audio`, `torch`, `librosa`, `soundfile`, `pydub`, `jiwer`, `pyloudnorm`, `pyyaml`, `numpy`, `symspellpy`
- **External tools:** `ffmpeg` and `ffprobe` (install via `brew install ffmpeg`)

---

## 4. Architecture

6-stage pipeline orchestrated by `scripts/run_pipeline.py`:

```
analyze → preprocess → diarize → transcribe → compare (optional) → editor
```

Supporting modules:
- `confidence.py` — **wired in**; auto-runs after transcription; exports `*_review_list.txt` with top 20 flagged segments. Validation against fasit still pending.
- `database.py` — opt-in via `--use-database`
- `spell_check.py` — opt-in via `--spell-check` **but currently a silent no-op** (no dictionary loaded — see open finding K5 in §6)
- `vocabulary.py` — opt-in via `--vocabulary-file`; generates Whisper `initial_prompt`
- `config.py`, `utils.py` — config loading, logging, file helpers

---

## 5. Critical Technical Constraints

These are constraints of the environment and will not change soon:

1. **CTranslate2 does not support Apple Metal/MPS.** Transcription is CPU-only on Mac. `device="mps"` raises `ValueError: unsupported device mps`. Diarization (PyTorch) *can* use MPS.
2. **WhisperX does not produce alignment scores for pure-number tokens** (e.g., "2024", "1,5"). Acoustic confidence is weakest exactly where errors are most costly. Compensate with hard-rule flagging, not acoustic confidence.
3. **Whisper `initial_prompt` has a 224-token hard limit.** The current vocabulary estimator assumes ~2 BPE tokens per Norwegian word — naive for compound words. Risk of silent prompt truncation.
4. **pyannote requires HF authentication.** Run `uv run huggingface-cli login` once. The current `check_hf_auth()` only checks that a token *exists*; it does not verify validity (AUDIT M4).
5. **Whisper confidence scores are not calibrated.** Useful for ranking, not for probability thresholding.

---

## 6. Current Reality (verified against code, not document claims)

### Open issues tracked in `ISSUES.md`
- **#8** `editor.py` is a placeholder — correctly parked (Subtitle Edit covers the need)
- **#25** `pydub` uses deprecated `audioop` — Python 3.13 time bomb

### Resolved issues (still worth knowing about)
- **#4** `segmentation_model` removed from `config.yaml` — pyannote 3.1 bundles its own
- **#5** Stereo channel-splitting implemented (`split_stereo_channels()`), but pipeline still processes averaged mono. Full channel-aware orchestration is future work.
- **#9** `compare.py` now uses `jiwer.wer()` for word-level similarity — time-overlap alignment remains appropriate for segment-level pairing
- **#21** `spell_check.py` explicitly disabled when no dictionary loaded — honest failure instead of silent no-op
- **#22** `preprocess.py` no longer loads audio twice — `AudioMetadata.audio_data` caches the loaded array (ephemeral, excluded from JSON)
- **#23** `vocabulary.py` now uses `transformers.AutoTokenizer` for accurate token counting; default `max_tokens=150` stays well under Whisper's 224-token limit

### Critique of AUDIT.md fixes worth knowing
- **K1 fix (skip files < 1 KB) is a weak heuristic.** `moov atom not found` can occur on larger corrupted files too. A more robust fix is per-file ffprobe error handling that skips on failure rather than relying on size. Track as a follow-up.
- **K2 fix (language fallback to "no" on low confidence)** bakes Norwegian into the pipeline. Acceptable for current scope; revisit if multi-language is ever in scope.

### `confidence.py` limitations (intentional gaps)
Wired into the pipeline, but two known gaps remain:
- **Validation pending.** Priority ranking has never been compared to actual segment-WER. Cannot be done without fasit. First task once fasit exists: compute Spearman correlation or precision@k between priority score and per-segment WER.
- **Hard-rules pending.** Segments containing digits or capitalized OOV tokens should always be flagged regardless of score — this covers "confidently wrong" failures that scores cannot see.

### Recently resolved (per `CHANGELOG.md` v0.1.4–v0.1.6)
#11, #12, #14, #15, #16, #17, #18, #19, #20, #21. See `ISSUES.md` for details.

---

## 7. Single Source of Truth — and observed drift

`ISSUES.md` is canonical for issue status. `README.md`, `ROADMAP.md`, and `AUDIT.md` may reference it but must not contradict it.

**Drift currently in the repo (fix before any new feature work):**
- `README.md` "Gjenstående" lists #14 as "designet og testet, men ikke wiret inn" — but #14 is resolved.
- `ROADMAP.md` "Remaining" lists #11 — but #11 is resolved.
- `ROADMAP.md` "Resolved (2026-05-28)" stops at #13. Missing: #11, #14, #15, #16, #17, #18, #19.
- `README.md` batch example uses `--workers 4`, but default and recommendation is `1` (CPU-bound, GIL).
- `AUDIT.md` claims "Ingen dokumentasjonsdrift oppdaget" — false. The items above prove it. **Treat audit summaries with the same skepticism as any other source.**

---

## 8. Operating Principles (non-negotiable)

1. **Measurement before features.** No new feature work begins until fasit exists and `evaluate.py` has produced a baseline WER. If those don't exist, stop and report — do not invent test data, do not fabricate numbers.
2. **Bias toward closing, not opening.** New entries in `ISSUES.md` require justification tied to an observed failure mode on real audio.
3. **No self-marking-done.** Mark an item resolved only when the verification in §10 passes. Cite the evidence (commit hash, test names, WER numbers) in `CHANGELOG.md`.
4. **Honest reporting.** Regressions, failed tests, or surprising behavior on real audio are stated explicitly. Never paper over.
5. **Distrust audits — including your own.** Verify against code, not against doc summaries. An audit claiming "no drift" is not evidence of no drift.
6. **Findings introduced during an audit must be added to `ISSUES.md` before being declared addressed.** Floating findings (like AUDIT K5) are not tracked work.

---

## 9. Priority Queue

At session start, pick the next task using this order. Stop at the first item that applies:

1. **Tier 1 — fasit + first WER measurement on a real recording.** Blocks all other work. Without it, every change below is unverifiable.
2. **Confidence validation** — once fasit exists, compute Spearman / precision@k between `confidence.py` priority score and per-segment WER. If correlation is weak, fix the signal mix before trusting the review list.
3. **Confidence hard-rules** — always-flag segments containing digit tokens or capitalized OOV tokens, regardless of score.
4. **K5 (spell_check) decision** — either load a real Norwegian dictionary, or remove `--spell-check` from the CLI. Current state is silent false trust. Add to `ISSUES.md` first.
5. **Knob tuning against fasit** — each experiment must produce before/after WER in `CHANGELOG.md`:
   - main vs verbatim model
   - `condition_on_previous_text` on/off
   - vocabulary on/off
   - second comparison model on/off
   - `--spell-check` on/off (only meaningful after K5 is resolved)
6. **Documentation reconciliation pass** — fix the drift listed in §7.
7. **#4** — remove `segmentation_model` from `config.yaml` and document that pyannote 3.1 bundles its own.
8. **AUDIT findings not yet tracked** (H2, H4, M1, M2, M4, plus follow-up on K1's weak heuristic) — add to `ISSUES.md` first, then schedule.
9. **#5 stereo** — first run `analyze.py` on real files and check `has_stereo_separation`. If mono, close without writing code.

Anything not on this list and not in `ISSUES.md` is out of scope. Items explicitly deferred in `AUDIT.md` Tier 5 (web editor #8, DTW alignment #9, Apple Silicon GPU for transcription, REST API, Docker, multi-language, fine-tuning, full CI) are not to be touched.

---

## 10. Definition of Done (by task type)

- **Bug fix:** root cause identified; fix applied; unit test added or updated; full suite passes; behavior verified on at least one real recording when relevant.
- **Feature:** wired into pipeline or CLI; unit tests added; run on at least one real recording; output inspected; inspection summarized in commit message and `CHANGELOG.md`.
- **Accuracy change:** before/after WER computed against the fasit; both numbers cited in `CHANGELOG.md`. If WER did not improve, the change is reverted or explicitly justified with a non-WER reason.
- **Documentation:** no contradictions remain between `ISSUES.md`, `README.md`, `ROADMAP.md`, `AUDIT.md`, and `CHANGELOG.md`. Verify with §12.

---

## 11. What "Test Run" Means

`pytest` alone is not a test run. The full run is all four steps:

1. `uv run pytest -q` — all tests pass.
2. Run the full pipeline on at least one real recording from the operator's working set.
3. Run `scripts/evaluate.py` against the fasit if it exists. Record WER and CER.
4. Manually inspect the first 30 seconds of output. Note any visible failure modes: hallucinations, dropped speaker labels, misaligned timestamps, missing scores on digit tokens.

---

## 12. Audit Checklist (run before declaring session complete)

Do the actual checks — do not trust summaries:

1. **Stale "Open" issues.** Grep the code. Any issue marked Open but already fixed? Mark resolved, cite the commit.
2. **Stale "Resolved" issues.** Any issue marked Resolved where the bug is still present in code? Reopen.
3. **README drift.** Does `README.md` "Gjenstående" or "Limitations" mention anything `ISSUES.md` marks resolved?
4. **ROADMAP drift.** Does `ROADMAP.md` "Remaining" mention anything resolved? Does "Resolved" include all resolved issues?
5. **Example drift.** Do CLI examples in `README.md` match current defaults (e.g., `--workers`)?
6. **Roadmap checkbox honesty.** Any `[x]` whose underlying artifact does not exist?
7. **WER trend.** If this session touched the accuracy path, record before/after in `CHANGELOG.md`.
8. **Scope creep.** Did this session touch anything on the deferred list? Justify or revert.
9. **Floating audit findings.** Were any findings from an audit (yours or another agent's) added to `ISSUES.md` as tracked entries before being declared addressed?

---

## 13. Git Discipline

- One logical change per commit. Use conventional prefixes: `fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Commit message states what changed, why, and — for accuracy changes — before/after WER.
- `uv run pytest -q` before every commit. Do not commit on a red suite.
- Push after each commit unless explicitly told otherwise.
- Never force-push.

---

## 14. Ground-Truth Workflow (the #1 priority)

To measure transcription quality:

1. Manually transcribe 5–10 minutes of a real recording, perfectly, to `testdata/fasit.txt`.
2. Run the full pipeline on the same recording.
3. `uv run python scripts/evaluate.py --reference testdata/fasit.txt --hypothesis output/<run>/recording.srt`.
4. Record WER, CER, and a brief error-mode summary (names, numbers, dialect, crosstalk, hallucinations).
5. Use those numbers as the gate for every subsequent change.

Without this loop, all optimization is guesswork. This is the single highest-leverage action available right now.

---

## 15. What NOT to Do

- Do not enable `--spell-check` by default — it currently does nothing (AUDIT K5).
- Do not build a web editor (#8) — Subtitle Edit covers the need.
- Do not add DTW or other fancy alignment in `compare.py` (#9) — `jiwer` is sufficient.
- Do not add REST API, Docker, multi-language support, or fine-tuning.
- Do not build elaborate CI/CD — targeted unit tests are sufficient for a personal tool.
- Do not chase Apple Silicon GPU acceleration for transcription — CTranslate2 limitation, not a code fix.
- Do not build more tools or modules before fasit + first WER measurement exist.
- Do not trust any document — verify against code.
- Do not mark an audit finding "addressed" without first adding it to `ISSUES.md` as a tracked entry.

---

## 16. End-of-Session Report

Produce a short written summary covering:

- What was attempted and what is actually done (with evidence: commit hash, test names, WER numbers).
- WER before/after if applicable.
- Issues opened, closed, or reconciled, with reasons.
- Drift caught and fixed during the audit step.
- What you would pick next, per §9.
- Anything you were unsure about or chose not to do, and why.

If any operating principle in §8 was bent or violated during the session, state that at the top of the report.
