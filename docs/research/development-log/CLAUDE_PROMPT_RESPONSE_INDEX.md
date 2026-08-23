# Claude Prompt/Response Index — Recording Policy (Phase 9B onward)

## Why this exists

Phases 0-9A were reconstructed for `000-project-start.md` through
`010-phase-9a-opssat.md` from repository artifacts, because the
original chat prompts and AI responses for those phases were not
preserved as standalone files (one exception: the Phase 9 dataset-search
step, see `docs/research/claude-research/phase-9-real-spacecraft-dataset-search-prompt.md`
and `...-response.md`, which ARE genuinely preserved). Every one of
those 11 entries says so explicitly rather than inventing the missing
content.

From Phase 9B onward, this project preserves the real material
directly, so no future entry needs to say "not preserved."

## What gets recorded for every future major task

For every major AI-assisted task from Phase 9B onward (a phase
implementation, a scientific audit, a design task, a documentation
task), the following are preserved:

1. **Prompt** — the user's actual instruction text for the task, as
   given.
2. **Claude response/report** — the substantive final response(s) —
   audit findings, design decisions, final reports — not routine tool
   output or intermediate exploration.
3. **Files changed** — exact list of files created/modified, taken
   from `git status`/`git diff --stat`, not summarized from memory.
4. **Tests** — which test suites were run, pass/fail counts, taken from
   actual `pytest` output, not estimated.
5. **Results** — real, measured metrics only. If a result is not yet
   available, it is recorded as "Not available" / "Awaiting
   experiment," matching the project's existing "Data & result
   integrity" policy stated in `README.md`.
6. **Problems** — errors, unexpected findings, or contradictions
   discovered, recorded honestly, including ones that reflect badly on
   an earlier phase.
7. **Decisions** — what was chosen and, critically, why — especially
   when an option that "sounds better" was rejected in favor of a
   simpler, more defensible one (as happened with Option A vs. Option B
   in the Phase 9A audit).
8. **Git state** — branch, HEAD commit, working-tree status, at both
   the start and end of the task.
9. **Commit** — exact commit hash and message, once (and only once)
   the user has explicitly approved committing.
10. **Push** — explicit push status; a task is never marked "pushed"
    unless a real `git push` happened, per this project's standing
    rule that commits/pushes require separate, explicit per-action
    approval.
11. **Next step** — what the task's own final report identified as the
    next planned step, so the chain from entry to entry stays legible
    without re-deriving it from context.

## Where this gets recorded

- **Every completed phase** gets its own numbered entry in this
  directory (`011-phase-9b-....md` onward), following the same
  27-point template used in `001-phase-0.md` through
  `010-phase-9a-opssat.md`.
- **`MASTER_PROJECT_JOURNAL.md`** gets a new row per completed phase,
  appended (never edited retroactively to change a historical row's
  content — corrections are new, dated notes referencing the old row,
  matching `docs/research/decisions/research_decision_log.md`'s own
  stated policy).
- **Design-only or audit-only tasks** that don't produce a "phase" in
  the Phase 0-9A sense (e.g. this documentation-system task itself, or
  the Phase 9B design-only audit) still get an entry, but are clearly
  labeled as design/audit entries rather than implementation entries.

## Non-negotiable rule

**No entry in this index or in any future development-log file may
fabricate a prompt, a response, a metric, a decision, or a date.** If
something is not recoverable at the time an entry is written, the entry
says so explicitly, exactly as every Phase 0-9A entry in this directory
does. This rule is permanent and applies to every future contributor —
human or AI — who adds to this log.
