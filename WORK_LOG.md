# ETIB — Daily Work Log

Running record of daily tasks and hours worked.

| Day | Date | Task | Hours |
|-----|------|------|-------|
| 1 | 2026-08-03 | Excel ETIB validation / evaluation | 5 |
| 2 | 2026-08-04 | Corrected feedback from Madame Lina | 5 |
| 3 | 2026-08-05 | Python statistics pipeline + tested the adjusted features | 5 |
| 4 | 2026-08-06 | Madame Lina feedback round 2 (grounding, glossary, scenarios, evaluation) | 5 |
| 5 | 2026-09-16 | UN sourcing restored after the Digital Library went behind a WAF; mobile layout fixes | 5 |

---

## Day 1 — Excel ETIB validation / evaluation
- Built / ran the ETIB validation-evaluation Excel.

## Day 2 — Feedback corrections
- Corrected feedback from Madame Lina.

## Day 3 — 2026-08-05
- Wrote `analysis/etib_stats.py`: computes every statistic the Excel workbook left "for Python/R" (kappa, ICC, Krippendorff α, Pearson/Spearman, ANOVA/Kruskal-Wallis, paired t/Wilcoxon, Cronbach α, bootstrap CI), organised by the 6 research questions.
- Reads the raw tester-input columns from `ETIB_Validation_Evaluation.xlsx`, so it runs even before Excel recalculates. Degrades gracefully ("not enough data yet") on empty sheets.
- Verified on the real file (runs, reads the 2 sample rows) and in `--demo` mode (fabricated data, shows what filled output looks like). Writes a report to `analysis/ETIB_stats_results.md`.
- Ready to re-run unchanged once the test campaign fills the workbook.
- Tested the features adjusted in the 3–4 Aug Lina/ETIB round:
  - **Session resume** — localStorage snapshot restores the last generated speech + Continue/Start-fresh banner on reload.
  - **Speech-rate / WPM warning** — Module B shows measured words-per-minute under the player, warns above 120 wpm.
  - **Glossary edits & re-import** — user edits no longer overwritten; bilingual re-import fills Arabic + English.
  - **fetch-url** — web-page source fetch via curl_cffi Chrome impersonation; clear "save as PDF and upload" message on 401/403.
  - **Media upload** — unreadable formats mapped to a clear error; upload limit raised 50 → 120 MB.
  - **Groq TPM 429 retry** — waits Groq's suggested delay and retries once with a smaller token budget.

## Day 4 — 2026-08-06
Addressed Madame Lina's second feedback round (4 points). Backend `vite build` + Python syntax/unit checks all green. **DEPLOYED 2026-08-10**: main repo `e0fff71` → `deploy/main` (Vercel prod) + `origin/joe-main` (team); HF Space `1cab579` (module_a + module_d, also carried the pending materials-reset fix). Also replaced the ESIB footer logo with the new USJ–ESIB image.
1. **Source grounding ignored the added link** — root cause: editing the topic/domain after attaching a source silently wiped it ([App.jsx](Frontend/src/App.jsx) `updateField`), reverting to the auto UN-library lookup. Fix: sources now persist until removed via the chip ×; backend grounding block made MANDATORY so an attached source is actually used and its subject is never swapped ([module_a.py](backend/modules/module_a.py)).
2. **Glossary corrections not remembered** — new `glossary correction memory` (localStorage, per browser): every equivalent/definition the user edits is saved by term and auto-re-applied to future glossaries. UI hint + "forget saved corrections" button; labels in EN/AR/FR.
3. **Scenario options didn't change the output** (e.g. "Entrevue" wasn't an interview) — added `SCENARIO_FORMATS` in [module_a.py](backend/modules/module_a.py): the prompt structure now matches the setting (interview → Q&A exchange, press conference → statement + journalist questions, court/medical/consultation → two-speaker turns, etc.), overriding the default podium scaffold.
4. **Evaluation missed intentional errors (grammar, numbers)** — [module_d.py](backend/modules/module_d.py): TASK 5 grammar made assertive for FR/EN, with an Arabic honesty rule (unwritten case endings can't be confirmed from an undiacritized transcript → don't falsely pass, defer to Ali's acoustic module). Added a compromised-recording check so a truncated/garbled transcript (bad connection) is flagged as unreliable instead of scored leniently.

**Non-code follow-ups from her message:** she needs **editor access** to the shared validation Excel (she couldn't edit it); her daily Groq quota was exhausted (expected, nothing to fix).

**Follow-up (Kevin, same day) — false-positive number errors:** evaluation flagged `2015-16` said as `2015 2016` as "Incorrect / missing hyphen between years" — a formatting difference, not a value error. Fixed in [module_d.py](backend/modules/module_d.py): (a) prompt TASK 10 now states year ranges written differently ("2015-16" = "2015-2016" = "2015 2016" = "٢٠١٥-٢٠١٦") are the same, and a missing hyphen/dash/space is never a number error; (b) `_normalize_year_ranges` expands ranges before fingerprinting so all renderings compare equal and the equivalence filter auto-un-flags them. Verified: all range variants → correct; chapter-style "16-4" not treated as a range; a genuine year error still flags.


## Day 5 — 2026-09-16 — UN sourcing restored + mobile layout

**1. Speeches were being grounded in Wikipedia, not UN documents.**
Reported as "it was from UN, now it is showing wiki". Root cause is external,
not a regression in our code: the UN put the whole of `digitallibrary.un.org`
behind an AWS WAF JavaScript challenge. Every path on that host — home page,
`/record/<id>`, the PDF files and the `/search` API — answers `HTTP 202` with
`x-amzn-waf-action: challenge` and an empty body. Verified against all four
endpoints.

The three-tier bypass we already had (`curl_cffi` → `curl` → `requests`) was
built for TLS-fingerprint blocking; a JS challenge needs the challenge script
executed to earn an `aws-waf-token` cookie, which TLS impersonation cannot do.
So `_search_un_api` returned `[]` every time and Module A fell through to its
Wikipedia fallback on every generation. It was invisible because each failure
logged at DEBUG only, and the generated speech looked normal.

**Fix.** `documents.un.org` is a different host and is *not* challenged; its
symbol API serves the real PDFs in English, French and Arabic. No keyword
search survives anywhere (`search.un.org` 400s on every parameter spelling,
`documents.un.org/api/search` requires a token, and the reachable OAI-PMH
interface exposes only a `sanctions` set), so
[un_documents.py](backend/utils/un_documents.py) ships a catalog of **39
verified symbols** and ranks it locally instead. Every symbol was fetched on
16 Sep 2026 and its date read from page 1 of the PDF — nothing is guessed.
They are **verbatim records**: transcripts of speeches actually delivered at
the UN, which suits interpreter training better than the summary documents the
old keyword search often returned.

Ranking weights a title hit above a topic hit, and weights a general-debate
record's topic list lower still (that list spans the whole UN agenda by
design). That is what lets "women rights" reach the Council's *Women and peace
and security* record rather than a general-debate sitting that merely mentions
it. Extracted text is cached on disk — **cold 17.1s → warm 0.0s** — and the
topic-specific records are warmed on a daemon thread at boot (`UN_PREWARM=0`
to skip). Failures now log at WARNING so a silent regression cannot repeat.

**2. Mobile layout.** Three bugs from a phone screenshot of the live site:
the topic box clipped its own placeholder mid-word (an *inline* `min-height`
beat every media query, so no breakpoint could grow it); the assistant panel
was a hard-coded 340px, wider than a 360px phone's viewport once the dock
inset is subtracted, so it ran off-screen invisibly under `overflow-x:hidden`;
and the launcher sat under the iOS home indicator because `viewport-fit=cover`
was missing from the viewport meta. Desktop is untouched above 600px.

**Rollback point:** tag `pre-un-rewrite-2026-09-16`.
