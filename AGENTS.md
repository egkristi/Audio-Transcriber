# AGENTS.md — Audio-Transcriber AI Agent Instructions

> Read this file first. Then read `ISSUES.md` (Open section), `ROADMAP.md` (near-term), and the latest `CHANGELOG.md` entry. **Verify the current state against the actual code before trusting any document — drift has been observed in this repo, including in audit reports.**
>
> **This file is a model-agnostic contract.** It is written to be followed identically by any coding agent — DeepSeek, Kimi, Claude (Opus/Sonnet), GPT, Gemini, Qwen, local models — driven from **VS Code** (Continue, Cline, Roo Code, Copilot Chat, Cursor, or the integrated terminal). Whatever model you are, obey these rules and leave the canonical trackers (`ISSUES.md`, `ROADMAP.md`, `CHANGELOG.md`) updated so the *next* agent — likely a different model — can resume cleanly. The trackers are the shared memory between models. See **§18 (VS Code environment)** and **§19 (multi-model guidance)** for setup and per-model expectations.

---

## 1. Mission

Drive the project toward measurably accurate Norwegian transcription of real call recordings on the operator's machine. Strategic guidance lives in `AUDIT.md` §«Strategisk gjennomgang» and is canonical.

The non-negotiable goal: a fasit (ground-truth transcript) plus a baseline WER number on a real recording. Without those, every "improvement" is speculation.

**Session Workflow**

1. Solve issues in ISSUES.md, then keep working on ROADMAP.md
2. If the need for new features are identified, add to ROADMAP.md
3. If any issues are identified, add to ISSUES.md
4. For each new features/change update CHANGELOG.md, then commit to git and push.
5. Do a test run and then do an extensive analysis and audit of the results.
6. Add suggested improvemnts to ROADMAP.md
7. Add any issues to ISSUES.md

---

## 2. Project Identity

- **Name:** Audio-Transcriber
- **Purpose:** Norwegian audio transcription pipeline with speaker diarization, multi-model comparison, and confidence-based review prioritization
- **Primary language:** Norwegian (Bokmål/Nynorsk). Language detection currently falls back to `"no"` when confidence < 0.5 — a deliberate hardcode for a Norwegian-only tool.
- **Dialect:** Northern Norwegian (Nordland, Troms, Finnmark). The audio recordings feature Northern Norwegian dialects with characteristic features: "æ" (jeg), "ikkje" (ikke), "ka" (hva), "kor" (hvor), "mæ" (meg), "dæ" (deg), "sæ" (seg), "dokker" (dere), "no" (noe). The normalize module flags dialect-standard mismatches for awareness but does NOT auto-correct — dialect is valid Norwegian.
- **License:** MIT
- **Repository:** https://github.com/egkristi/Audio-Transcriber

---

## 3. Tech Stack

- **Python:** 3.11+ (previously locked to `<3.13` due to `audioop` in `pydub`; resolved by removing unused `pydub` dependency)
- **Package manager:** `uv` (never `pip` directly — bypasses lockfile)
- **OS:** macOS primary; Linux/Windows for CUDA paths
- **Core dependencies:** `whisperx`, `pyannote.audio`, `torch`, `librosa`, `soundfile`, `jiwer`, `pyloudnorm`, `pyyaml`, `numpy`, `symspellpy`
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

### Resolved issues (still worth knowing about)
- **#4** `segmentation_model` removed from `config.yaml` — pyannote 3.1 bundles its own
- **#5** Stereo channel-splitting implemented (`split_stereo_channels()`), but pipeline still processes averaged mono. Full channel-aware orchestration is future work.
- **#9** `compare.py` now uses `jiwer.wer()` for word-level similarity — time-overlap alignment remains appropriate for segment-level pairing
- **#21** `spell_check.py` explicitly disabled when no dictionary loaded — honest failure instead of silent no-op
- **#22** `preprocess.py` no longer loads audio twice — `AudioMetadata.audio_data` caches the loaded array (ephemeral, excluded from JSON)
- **#23** `vocabulary.py` now uses `transformers.AutoTokenizer` for accurate token counting; default `max_tokens=150` stays well under Whisper's 224-token limit
- **#25** `pydub` dependency removed — unused; `audioop` deprecation no longer blocks Python 3.13

### Critique of AUDIT.md fixes worth knowing
- **K1 fix (skip files < 1 KB) is a weak heuristic.** `moov atom not found` can occur on larger corrupted files too. A more robust fix is per-file ffprobe error handling that skips on failure rather than relying on size. Track as a follow-up.
- **K2 fix (language fallback to "no" on low confidence)** bakes Norwegian into the pipeline. Acceptable for current scope; revisit if multi-language is ever in scope.

### `confidence.py` limitations (intentional gaps)
Wired into the pipeline, but two known gaps remain:
- **Validation pending.** Priority ranking has never been compared to actual segment-WER. Cannot be done without fasit. First task once fasit exists: compute Spearman correlation or precision@k between priority score and per-segment WER.
- **Hard-rules pending.** Segments containing digits or capitalized OOV tokens should always be flagged regardless of score — this covers "confidently wrong" failures that scores cannot see.

### Recently resolved (per `CHANGELOG.md` v0.1.4–v0.1.7)
#11, #12, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23, #24, #25, #26, #27, #28, #29. See `ISSUES.md` for details.

---

## 7. Single Source of Truth — and observed drift

`ISSUES.md` is canonical for issue status. `README.md`, `ROADMAP.md`, and `AUDIT.md` may reference it but must not contradict it.

**Drift history (all items below have been resolved as of 2026-05-29):**
- ~~`README.md` "Gjenstående" listed #14 as "designet og testet, men ikke wiret inn" — but #14 is resolved.~~ ✅ Fixed.
- ~~`ROADMAP.md` "Remaining" listed #11 — but #11 is resolved.~~ ✅ Fixed.
- ~~`ROADMAP.md` "Resolved (2026-05-28)" stopped at #13. Missing: #11, #14, #15, #16, #17, #18, #19.~~ ✅ Fixed — now lists all issues through #29.
- ~~`README.md` batch example used `--workers 4`, but default and recommendation is `1` (CPU-bound, GIL).~~ ✅ Fixed.
- ~~`AUDIT.md` claimed "Ingen dokumentasjonsdrift oppdaget" — false.~~ ✅ Fixed.
- **Note:** `AUDIT.md` claims about no drift were false at the time. Treat audit summaries with the same skepticism as any other source.

**Current drift check (2026-05-29):** No drift detected between `ISSUES.md`, `README.md`, `ROADMAP.md`, and `CHANGELOG.md`. All resolved issues through #29 are correctly reflected across all documents. Verify with §12 before declaring session complete.

---

## 8. Operating Principles (non-negotiable)

1. **Measurement before features.** No new feature work begins until fasit exists and `evaluate.py` has produced a baseline WER. If those don't exist, stop and report — do not invent test data, do not fabricate numbers.
2. **Bias toward closing, not opening.** New entries in `ISSUES.md` require justification tied to an observed failure mode on real audio.
3. **No self-marking-done.** Mark an item resolved only when the verification in §10 passes. Cite the evidence (commit hash, test names, WER numbers) in `CHANGELOG.md`.
4. **Honest reporting.** Regressions, failed tests, or surprising behavior on real audio are stated explicitly. Never paper over.
5. **Distrust audits — including your own.** Verify against code, not against doc summaries. An audit claiming "no drift" is not evidence of no drift.
6. **Findings introduced during an audit must be added to `ISSUES.md` before being declared addressed.** Floating findings (like AUDIT K5) are not tracked work.
7. **Use subagents for leverage.** Delegate research, QA review, architecture validation, optimal-solution search, documentation updates, code generation, and parallel exploration to subagents (§20). A subagent call costs less than a full turn of sequential tool calls — use them aggressively for everything you can.
   - **Parallelize independent tasks** (§20.6) — fire multiple subagents simultaneously instead of sequentially.
   - **Route to the right model** (§20.7) — `cheap` for docs/research, `best-coding` for implementation, `best-chat` for QA.
   - **Use custom agents** (§20.9) for specialized, tool-restricted workflows.

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

---

## 17. Session Workflow (mandatory)

At the start of every session, follow this sequence:

1. **Read `ISSUES.md`** — identify any open issues.
2. **Solve open issues** — work through them in priority order (§9). For each issue:
   - Verify against code, not document claims.
   - Apply fix; add/update tests; run `uv run pytest -q`.
   - Update `ISSUES.md` status to Resolved with date and fix description.
   - Update `CHANGELOG.md` with the change.
   - Commit with conventional prefix (`fix:`, `feat:`, `docs:`, `test:`).
   - Push to origin.
3. **Read `ROADMAP.md`** — once all open issues are resolved, continue with ROADMAP work.
   - Reconcile checkboxes against actual code (§12 checklist).
   - Identify next items to implement or verify.
4. **Identify new issues/features** during work:
   - If a bug or gap is found, **add to `ISSUES.md` first** before declaring it addressed (§8.6).
   - If a new feature is needed, **add to `ROADMAP.md`** under the appropriate phase.
5. **Update `CHANGELOG.md`** for every new feature or change.
6. **Commit and push** after each logical change. One change per commit. Never force-push.
7. **Use subagents throughout** — see §20 for when and how. Delegate research, QA, documentation, and parallel exploration aggressively. This workflow ensures the canonical trackers (`ISSUES.md`, `ROADMAP.md`, `CHANGELOG.md`) stay in sync with the code at all times.
   - **Parallelize research** (§20.6): Fire subagents for independent file reads simultaneously.
   - **Route by model** (§20.7): `cheap` for docs, `best-coding` for code, `best-chat` for review.
   - **Custom agents** (§20.9): Define `.agent.md` files for specialized workflows.

---

## 18. VS Code Development Environment

This project is developed in **VS Code** with an AI coding assistant (Continue, Cline, Roo Code, Copilot Chat, Cursor, etc.) and/or the integrated terminal. The setup below is the expected baseline regardless of which model backs the assistant.

### One-time setup

```bash
brew install uv ffmpeg          # ffmpeg/ffprobe are required (analyze.py shells out to ffprobe)
cd /Users/erling/code/Audio-Transcriber
uv sync                          # creates .venv/ from uv.lock (Python 3.11 — see .python-version)
uv run huggingface-cli login     # once, for pyannote diarization (HF-gated model)
cp .env.example .env             # then fill in HF_TOKEN; .env is gitignored
```

The interpreter lives at `.venv/bin/python`. Point VS Code at it: **Cmd-Shift-P → Python: Select Interpreter → `./.venv/bin/python`**.

### Recommended workspace files

`.vscode/` is gitignored, so these are local-only and safe to create. Suggested `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": { "source.organizeImports": "explicit" }
  },
  "files.watcherExclude": {
    "**/testdata/**": true,
    "**/output/**": true,
    "**/.venv/**": true
  },
  "search.exclude": {
    "**/testdata": true,
    "**/output": true,
    "**/uv.lock": true,
    "**/.venv": true
  }
}
```

Suggested `.vscode/launch.json` for debugging the pipeline on one file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Pipeline: single file (no diarize)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/run_pipeline.py",
      "args": ["--input", "testdata/<recording>.m4a", "--output-dir", "./output",
               "--no-diarize", "--dialect", "northern_norwegian"],
      "console": "integratedTerminal",
      "python": "${workspaceFolder}/.venv/bin/python"
    }
  ]
}
```

### Recommended extensions

- **Python** + **Pylance** (Microsoft) — interpreter, IntelliSense, test discovery.
- **Ruff** (`charliermarsh.ruff`) — lint + format (both are dev deps in `pyproject.toml`).
- Your AI assistant of choice (Continue / Cline / Roo Code / Copilot Chat / Cursor). Configure it to read `AGENTS.md` as project context.

### Everyday commands (run in the VS Code integrated terminal)

```bash
uv run pytest -q                                   # full unit suite (must be green before commit)
uv run pytest tests/test_confidence.py -q          # one module
uv run ruff check . && uv run ruff format --check . # lint + format gate
uv run python scripts/run_pipeline.py --help        # CLI surface
uv run python scripts/run_pipeline.py --input testdata/<file>.m4a --no-diarize   # quick run
uv run python scripts/evaluate.py --reference testdata/fasit.txt --hypothesis output/<run>/<file>.srt
```

- **Always** invoke Python via `uv run …`, never a bare `python` / `pip` — that bypasses the lockfile (§3).
- `ffprobe` must be on `PATH`; if `analyze.py` falls back to librosa with a warning, ffmpeg isn't installed.
- Do **not** add `.vscode/` to git, and do **not** commit `.env`, `output/`, `testdata/`, or `*.db` (all gitignored — keep it that way; see ISSUES.md #36 on data handling).

---

## 19. Multi-Model Agent Guidance (DeepSeek, Kimi, Opus, GPT, …)

Different models will work on this repo over time, each with different context windows, tool-calling reliability, and reasoning depth. These rules keep results consistent and let any model hand off to any other.

### Model-agnostic rules (apply to every model)

1. **The trackers are shared memory.** You may be a different model than the one that worked last, with none of its chat history. Reconstruct state only from `ISSUES.md` / `ROADMAP.md` / `CHANGELOG.md` / git log — and from the code itself. Before finishing, write your state back so the next model can resume (§16, §17).
2. **Verify against code, never against prose** (§8.5). Model training data may predate the current `whisperx` / `pyannote.audio` / `faster-whisper` APIs — confirm signatures with `uv run python -c "import whisperx; help(...)"` or by reading the installed package, don't assume.
3. **Minimal diffs.** Prefer the smallest change that fixes the issue. One logical change per commit (§13). Don't reformat unrelated code or "drive-by" refactor.
4. **Determinism over cleverness.** Use the exact commands in §18. Don't invent paths, flags, or filenames — the SRT naming bug (ISSUES.md #31) is exactly the kind of error that comes from reconstructing strings instead of reusing returned values.
5. **No fabricated numbers.** Never invent WER/CER or test results. If the fasit doesn't exist, say so and stop (§8.1).
6. **Note the model in the commit body** (optional but encouraged), e.g. a trailer line `Agent: kimi-k2` — it makes drift and regressions traceable across models.

### Context budgeting (especially for smaller-context models)

Read order, cheapest-useful-first: **`AGENTS.md` → `ISSUES.md` (Open) → `ROADMAP.md` (near-term) → only the specific `src/*.py` you're editing → its test.** Then expand as needed.

**Never load these into context** (they will blow a small window for zero benefit):
- `uv.lock` (~750 KB), `.venv/`
- `testdata/` and `output/` (real recordings/transcripts — also a privacy concern, ISSUES.md #36)
- `*.db`, `*.wav`, `*.m4a`, `*_review_list.json`

The whole `src/` tree is ~3 k LOC and each module is self-contained — read one module at a time rather than the whole tree.

### Per-model expectations (capability, not hierarchy)

- **Large-context / strong-reasoning models (e.g. Opus, large frontier models):** OK to take on multi-file changes (e.g. the language-pack refactor in ROADMAP Phase 11), cross-module audits, and the §12 audit checklist in one pass. Still: one logical change per commit.
- **Strong coders with mid-size context (e.g. DeepSeek-Coder, Kimi, Qwen-Coder):** excellent for scoped, single-module fixes (ISSUES.md #31–#37) and test writing. Keep tasks to one module + its test per step; lean on the read-order above to stay within context.
- **Smaller / local models:** restrict to one well-specified issue with an explicit fix already described in `ISSUES.md`. Always run `uv run pytest -q` after editing; if the suite can't be reasoned about, stop and hand back rather than guessing.

If you are unsure whether a change is within your reliable capability, **split it smaller** and leave a `ROADMAP.md` note for the remainder rather than half-finishing a large refactor.

### Tool-calling / agent-mode notes

- Confirm the working directory is the repo root before running commands (paths in §18 are repo-relative).
- After any file edit, re-run the relevant `pytest` target before moving on — don't batch edits across modules without a green checkpoint.
- If your assistant supports it, add `AGENTS.md` (and `ISSUES.md`) to the always-included project context so these rules survive context truncation.
- Long pipeline runs (CPU transcription is minutes per recording, §5) will exceed short tool timeouts — run them in the integrated terminal, not as a blocking agent tool call, and inspect the output dir afterward.
- **Use subagents (§20) for any task that can be parallelized or delegated.** Do not do sequential research when a subagent can do it in one turn.
- **Use parallel subagents (§20.6) for independent research tasks** — fire 3 subagents at once instead of doing 3 sequential reads. VS Code 2026 supports this natively.
- **Route subagents to the right model (§20.7):** Use `cheap` for documentation and research, `best-coding` for implementation, `best-chat` for architecture review. Don't waste an expensive model on a grep task.
- **Consider custom agents (§20.9)** for specialized workflows — a read-only research agent, a security scanning agent, or a documentation agent. Define them in `.agent.md` files.

---

## 20. Subagent Workflow — The Core Productivity Lever

Subagents (the `runSubagent` tool) are your **primary productivity multiplier**. Use them aggressively for everything: research, quality assurance, architecture validation, finding optimal solutions, parallel work, documentation, cost reduction, and code generation. A subagent call costs less than a full turn of sequential tool calls — and produces better results.

**Core philosophy:** Subagents are not a luxury or an afterthought. They are how you work faster, better, cheaper, and more reliably. Every task that can be delegated, should be delegated. Every decision that can be validated, should be validated. Every doc that can be updated in parallel, should be.

**VS Code 2026 multi-agent support:** This environment supports parallel subagents (§20.6), model routing by alias (§20.7), and custom agents with handoffs (§20.9). Use all three aggressively — they are designed to work together.

### 20.1 When to use subagents — the complete list

| Situation | Why subagent | Example |
|---|---|---|
| **Research** | Don't burn your context on reading files you won't edit | "Read `src/transcribe.py` and tell me how `_split_long_segments` works" |
| **Quality assurance** | Get a second opinion on architecture decisions | "Review my planned change to `config.py` — is there a simpler approach?" |
| **Architecture validation** | Validate design decisions against constraints | "Will this approach work on CPU-only Mac? Check for MPS/CUDA assumptions" |
| **Finding optimal solutions** | Explore multiple approaches before committing | "Compare 3 ways to implement X: pros, cons, complexity. Recommend one." |
| **Parallel exploration** | Investigate multiple options simultaneously | "Search for the best way to implement X in 3 different approaches" |
| **Code review** | Catch bugs before they reach the test suite | "Review this diff for edge cases I might have missed" |
| **Test writing** | Write tests while you implement the feature | "Write pytest tests for the new `_split_long_segments` function" |
| **Documentation** | Update docs while you keep coding | "Update `CHANGELOG.md` with the changes I just made" |
| **Cost reduction** | Small models for simple tasks, big models for hard ones | "Use a cheap subagent to grep the codebase, then use the result yourself" |
| **Code generation** | Generate boilerplate or well-scoped code | "Write a Python dataclass for X with validation and serialization" |
| **Debugging** | Isolate root causes faster | "Read this traceback and the relevant source. Find the root cause." |
| **Performance analysis** | Profile and suggest optimizations | "Read this function. Where are the bottlenecks? Suggest 3 optimizations." |
| **Security review** | Catch vulnerabilities before they ship | "Review this file for injection risks, path traversal, or secret leaks" |
| **Decision records** | Document why a choice was made | "Write an ADR explaining why we chose approach A over B and C" |

### 20.2 When NOT to use subagents

- **Writing code that needs your current context.** If the subagent doesn't have the same context as you, it will produce wrong results. Pass enough context in the prompt.
- **Tiny tasks** (reading one file, making one edit). The overhead isn't worth it.
- **Tasks requiring the full conversation history.** Subagents are stateless — they don't see previous turns.

### 20.3 Subagent patterns

#### Pattern A: Research → You implement

Delegate research to a subagent, then implement based on its findings:

```
Subagent prompt: "Read src/transcribe.py and src/config.py. Find how
max_segment_duration is used. Return: (1) the exact line numbers where it's read,
(2) the default value if not set, (3) any validation logic. Do NOT edit any files."
```

#### Pattern B: You implement → Subagent QA

After making changes, have a subagent review them:

```
Subagent prompt: "Review this diff for src/transcribe.py. Check for:
(1) edge cases with empty segments, (2) type mismatches, (3) silent failures.
The function splits segments longer than max_duration into equal chunks.
Do NOT edit any files. Return a bullet list of concerns."
```

#### Pattern C: Parallel subagents

Fire multiple subagents at once for independent tasks:

```
Subagent 1: "Read ISSUES.md and return all open issues with their descriptions."
Subagent 2: "Read ROADMAP.md and return the next 3 uncompleted milestones."
Subagent 3: "Run `uv run pytest -q` and return the output."
```

Then combine their results in your main context.

#### Pattern D: Subagent for documentation

Delegate CHANGELOG.md / ROADMAP.md updates while you keep coding:

```
Subagent prompt: "Update CHANGELOG.md at /Users/erling/code/Audio-Transcriber/CHANGELOG.md.
Add a new entry for v0.1.22 with these changes: [list changes].
Use the same format as the previous entry. Do NOT modify any other files."
```

#### Pattern E: Architecture / design review

Before committing to a design, get a subagent to validate it:

```
Subagent prompt: "I plan to add a --max-segments CLI flag to run_pipeline.py that
overrides config.yaml's max_segment_duration. Read the relevant files and tell me:
(1) Is there already a pattern for CLI-overrides-config in this codebase?
(2) What's the simplest way to implement this?
(3) Any pitfalls with argparse + yaml override patterns?
Do NOT edit any files."
```

#### Pattern F: Optimal solution search

Before implementing, have a subagent explore the design space:

```
Subagent prompt: "I need to implement feature X in this Python codebase.
Read the relevant files and suggest 3 different approaches ranked by:
(1) Simplicity, (2) Performance, (3) Maintainability, (4) Fit with existing patterns.
For each approach, estimate LOC and risk. Recommend one. Do NOT edit any files."
```

#### Pattern G: Subagent generates code, you review

For well-scoped, self-contained code generation:

```
Subagent prompt: "Write a Python function that does X. It must:
- Accept parameters: [list]
- Return: [type]
- Handle edge cases: [list]
- Include type hints and docstring
- Follow the patterns in src/utils.py (read it first)
Write ONLY the function. Do NOT modify any existing files."
```

#### Pattern H: Multi-model quality stack

Use different models for different quality levels — cheap model for first pass, expensive model for review:

```
Turn 1 (cheap model): "Write a first draft of function X. Cover the happy path."
Turn 2 (expensive model): "Review this function for: edge cases, type safety,
performance, and correctness. The function does X. Return specific improvements."
Turn 3 (cheap model): "Apply the improvements from the review. Update the function."
```

This gives you near-expert quality at a fraction of the cost.

#### Pattern J: Parallel research burst

Fire multiple research subagents simultaneously to gather context fast:

```
Subagent 1 (fast): "Grep for 'vad_options' in src/transcribe.py. Return line numbers."
Subagent 2 (cheap): "Read config.yaml and return the transcription section."
Subagent 3 (cheap): "Read ISSUES.md and return all open issues."
```

Collect all results, combine insights, implement. Same cost as sequential, 3× faster.

#### Pattern K: Custom agent handoff pipeline

Use handoffs to create a plan → implement → review pipeline:

```
Step 1 (Research Agent): "Read the codebase and suggest 3 approaches for feature X."
Step 2 (You): Choose the best approach.
Step 3 (Implementation Agent): "Implement approach 2 for feature X."
Step 4 (QA Agent): "Review the implementation for edge cases and bugs."
Step 5 (You): Apply fixes, commit.
```

Each agent has the right tools and model for its job. The research agent is read-only (safe), the implementation agent has full editing, and the QA agent only reviews.

#### Pattern I: Subagent-as-linter

Before committing, have a subagent do a final quality pass:

```
Subagent prompt: "Read the diff between these two git refs (or the staged changes).
Check for: (1) debug code left in, (2) missing error handling, (3) silent failures,
(4) type mismatches, (5) violations of project conventions.
Return a bullet list of issues. Do NOT edit any files."
```

### 20.4 Writing effective subagent prompts

A good subagent prompt has four parts:

1. **Context** — what the subagent needs to know (2-3 sentences max)
2. **Task** — exactly what to do (research, review, write, edit)
3. **Scope** — what files to touch and what NOT to touch
4. **Output format** — what to return (bullet list, code, summary)

Bad: "Check the code for bugs."
Good: "Read `src/transcribe.py` lines 100-200. The `_split_long_segments` function splits segments >15s. Check for: (1) off-by-one errors in the split calculation, (2) whether word-level timestamps are preserved correctly, (3) what happens if a segment has 0 words. Return a bullet list. Do NOT edit any files."

**Prompt quality checklist:**
- ✅ Does the prompt include enough context for a stateless agent?
- ✅ Does it specify exactly what NOT to touch?
- ✅ Does it specify the output format?
- ✅ Does it include file paths and line numbers?
- ✅ Does it constrain the scope tightly enough?

### 20.5 Cost optimization — get more done for less

- **Use subagents for cheap research** instead of burning expensive context-window tokens on reading files you won't edit.
- **Batch independent reads** into one subagent call rather than multiple sequential `read_file` calls.
- **Use the `cheap` alias for simple tasks** (grep, file reading, formatting, documentation). Reserve `best-chat` or `claude-opus` for architecture, code review, and complex reasoning. The cost hierarchy is: `local` ≈ `fast` < `cheap` < `codestral` < `best-coding` < `best-chat` < `claude-opus`.
- **One subagent call is cheaper than 10 sequential tool calls** — prefer delegation over iteration.
- **Multi-model quality stack (Pattern H):** Use `cheap` for first drafts, `best-chat` for review, then `cheap` to apply fixes. This gives expert quality at budget prices.
- **Parallelize everything:** Instead of doing 3 research tasks sequentially, fire 3 subagents in parallel. Same cost, 3× faster wall-clock time.
- **Don't over-prompt:** A 50-word prompt to a `cheap` subagent costs less than a 500-word prompt. Be concise.
- **Use `fast` for throwaway tasks:** If you just need a quick grep or file listing and won't use the result for anything critical, `fast` (Groq Llama) gives the lowest latency.

### 20.6 Parallel execution — the force multiplier

VS Code now supports **running multiple subagents in parallel** (VS Code 2026-02 release). Fire off multiple tasks at once, get results faster, and save premium requests in the process. The main agent delegates work and only the final result flows back — intermediate exploration stays contained, keeping your primary context clean.

When you have multiple independent tasks, fire subagents simultaneously:

```
Turn 1: Fire Subagent A (research feature X) + Subagent B (research feature Y)
Turn 2: Collect both results, combine insights, implement
```

This cuts wall-clock time in half for research-heavy sessions.

**Parallel execution patterns:**

| Pattern | What to parallelize | Example |
|---|---|---|
| **Research split** | Independent file reads | Subagent A reads `transcribe.py`, B reads `config.py`, C reads `diarize.py` |
| **QA split** | Independent review dimensions | Subagent A checks correctness, B checks performance, C checks security |
| **Implementation split** | Independent code changes | Subagent A writes tests, B implements feature, C updates docs |
| **Validation split** | Independent validation | Subagent A runs tests, B reviews diff, C checks for drift |
| **Exploration split** | Independent approaches | Subagent A tries approach 1, B tries approach 2, C tries approach 3 |

**Rule of thumb:** If you can describe two tasks that don't depend on each other, fire them in parallel. Don't wait. VS Code shows which tasks are running, which agent is being used, and lets you expand any subagent to see the full prompt and result.

### 20.7 Model routing — use the right tool for each job

Different subagent models have different strengths. Route tasks to the model best suited for them.

#### Available models in this environment

All models are served through a local LiteLLM proxy (`http://localhost:4000/v1`) plus direct connections. The following named shortcuts are configured and should be used by alias:

| Alias | Model | Tool Calling | Vision | Context | Best for |
|---|---|---|---|---|---|
| `best-coding` | DeepSeek V4 Pro | ✅ | ❌ | 128K | Implementation, test writing, debugging |
| `best-chat` | Claude Sonnet 4.6 | ✅ | ✅ | 200K | Architecture, code review, QA, vision |
| `fast` | Groq Llama 4 Maverick | ✅ | ❌ | 128K | Quick research, grep, simple reads |
| `cheap` | DeepSeek V4 Flash | ✅ | ❌ | 128K | Documentation, boilerplate, cost-sensitive tasks |
| `local` | DeepSeek V4 (local) | ✅ | ❌ | 128K | Offline-capable tasks, sensitive data |
| `codestral` | Codestral 25.08 | ✅ | ❌ | 256K | Large-context code generation, refactoring |

Additional models available by full ID (use when specific capability needed):
- `claude-opus` — deepest reasoning, 200K ctx, vision
- `kimi-k2.6` — strong reasoning + vision, 128K ctx
- `gemini-3-flash` — fastest vision-capable, 128K ctx
- `qwen3.5-397b` — largest parameter count, no vision
- `ministral-3-8b` / `ministral-3-14b` — lightweight cloud models

#### Task-to-model routing table

| Task type | Best model alias | Why |
|---|---|---|
| **Grep / file search / simple reads** | `fast` | Lowest latency, good enough quality |
| **Boilerplate code generation** | `cheap` or `codestral` | Low cost, codestral for large outputs |
| **Test writing** | `best-coding` | Needs code understanding + tool calling |
| **Documentation updates** | `cheap` | Low complexity, high volume, minimal cost |
| **Architecture design** | `best-chat` or `claude-opus` | Deep reasoning, trade-off analysis |
| **Code review / QA** | `best-chat` | Catches subtle bugs, edge cases |
| **Debugging complex issues** | `best-coding` or `best-chat` | Cross-module causality tracing |
| **Security review** | `best-chat` | Security-aware reasoning |
| **Performance optimization** | `best-coding` | Algorithmic complexity understanding |
| **Vision tasks (images, UI)** | `best-chat` or `kimi-k2.6` | Vision-capable, strong reasoning |
| **Decision records (ADRs)** | `cheap` | Good writing, minimal cost |
| **Large refactoring (>500 LOC)** | `codestral` | 256K context, sees entire files |
| **Offline / sensitive data** | `local` | Never leaves your machine |

**Cost hierarchy (cheapest → most expensive):** `local` ≈ `fast` < `cheap` < `codestral` < `best-coding` < `best-chat` < `claude-opus`

**If your platform supports model selection per subagent call, use it.** Don't use an expensive reasoning model for a grep task, and don't use a cheap model for architecture validation. The `cheap` alias exists precisely for high-volume, low-complexity work — use it aggressively for documentation, research, and boilerplate.

### 20.8 Subagent safety rules

1. **Never pass secrets** in subagent prompts (they may be logged).
2. **Always specify "Do NOT edit any files"** for research-only subagents — otherwise they may make unwanted changes.
3. **Verify subagent output** — subagents can hallucinate just like any model. Cross-check file paths, line numbers, and code snippets.
4. **Subagents are stateless** — they don't see the conversation history. Include all necessary context in the prompt.
5. **Prefer read-only subagents** for research. Only use write-capable subagents when you explicitly want them to edit files.
6. **Pin file paths and line numbers** in your prompts — don't assume the subagent will find them correctly.
7. **Specify the output format explicitly** — "Return a bullet list" is better than "Tell me what you find."
8. **Set expectations about thoroughness** — say "quick scan" or "thorough review" so the subagent budgets its effort appropriately.

### 20.9 Custom agents and handoffs (VS Code 2026)

VS Code supports **custom agents** — specialized agent definitions with their own tools, instructions, and model. Combine with **handoffs** to create workflows that transition from plan → implement → review — all orchestrated from your main session.

#### When to create a custom agent

| Agent type | Tools | Model | Purpose |
|---|---|---|---|
| **Research agent** | Read-only (file search, grep, web fetch) | `cheap` or `fast` | Explore codebase without risk of edits |
| **Implementation agent** | Full editing + terminal | `best-coding` | Write code, run tests, apply fixes |
| **QA/review agent** | Read-only + test runner | `best-chat` | Review diffs, validate architecture |
| **Security agent** | Read-only + grep | `best-chat` | Scan for vulnerabilities, secrets |
| **Documentation agent** | Read-only + file edit | `cheap` | Update CHANGELOG, README, ROADMAP |

#### Handoff workflow pattern

```
Main session (you)
  │
  ├── Handoff to Research Agent ──► Returns findings
  │
  ├── You implement based on findings
  │
  ├── Handoff to QA Agent ──► Returns review
  │
  └── You apply QA feedback → commit
```

#### Handoff prompt template

```
Handoff to [Agent Name]: "I need you to [task]. 
Context: [2-3 sentences about what's happening].
Files to examine: [paths].
Output: [bullet list / code / summary].
Do NOT edit any files unless explicitly instructed."
```

#### Custom agent definition example (`.agent.md`)

```markdown
---
name: research-agent
model: cheap
instructions: |
  You are a read-only research agent. You can search files, read code,
  and fetch web pages. You must NEVER edit files. Return concise,
  well-organized findings.
tools:
  - read_file
  - grep_search
  - file_search
  - fetch_webpage
---
```

Create `.agent.md` files in the project root or `.vscode/` directory. VS Code auto-discovers them and makes them available for handoffs.

#### Key benefits of custom agents + handoffs

- **Specialized behavior per task** — each agent has the right tools and model for its job
- **Safety through tool restriction** — research agents can't accidentally edit files
- **Cost optimization** — cheap models for research, expensive only for implementation/review
- **Parallel handoffs** — fire a research agent and a documentation agent simultaneously
- **Traceability** — VS Code shows which agent ran, what prompt was used, and what it returned
