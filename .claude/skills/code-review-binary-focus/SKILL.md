---
name: code-review-binary-focus
description: The workflow for fixing bugs surfaced by a code review (/code-review or manual) in this repo - confirm-before-fix, root-cause scoping, one-finding-one-commit, regate ladder, and findings-doc bookkeeping. Use when working through a code-review findings list or fixing any reported finding.
---

# Fixing code-review findings

A review finding is a CLAIM, not a diagnosis. The workflow below turns claims
into bisectable, gated fixes without trading one bug for another.

## HARD RULE: one at a time

Work on exactly ONE bug/finding/entry/task at a time - confirm it, fix it,
gate it, commit it, tick it off - before even READING the next one's code.
No "while I'm in this file", no fixing a second finding you noticed on the
way, no interleaving two half-done fixes. If you spot a new bug mid-fix,
append it to the findings doc as a new `[ ]` entry and return to the one in
flight. A finding is done when its commit exists and its checkbox is ticked;
only then does the next one start.

## Where findings live

- Each review run gets a doc: `research/code-review-<branch>-<date>.md` with
  per-finding `[ ]` checkboxes, file:line, failure scenario, and verdict
  (CONFIRMED / PLAUSIBLE), plus the refuted-candidates table for the record.
- Tick `[x]` fixed / `[-]` won't-fix-with-reason as you go. The doc is the
  walkthrough ledger; TASKS.md gets only the landed-fix summary per ledger
  conventions (see llamacpp-v100-ledger).
- If findings were reported to the host UI this session (ReportFindings),
  re-report them with `outcome:` immediately after fixes land - the UI's
  per-finding status only updates from that call.

## Per-finding workflow

1. **Confirm against real code first, not the report.** Re-read the actual
   lines; check the failure path still exists and the report didn't misread
   intent (deleted code is sometimes a deliberate fix - read the comments and
   `git log -L` before "restoring" anything). PLAUSIBLE verdicts need a
   code-trace or repro BEFORE any edit; CONFIRMED ones still get a re-read.
2. **Classify before fixing:**
   - *Mechanical bug* (wrong field, missing call, race, unvalidated input):
     fix directly.
   - *Deliberate behavior change flagged as a bug* (e.g. a warn->exit
     hardening): decide intent FIRST. Either mark `[-]` with the rationale in
     the findings doc, or change it - never "fix" a decision by accident.
   - *Topology/config-dependent* (only fires under a gate or fleet shape):
     verify it applies to the actual roster before building a fix or repro.
3. **Fix the root cause at the narrowest scope that covers it.** If the same
   defect pattern exists at sibling sites (the report often lists "same root
   cause also at:"), one change covering all sites beats N spot-patches.
   Don't widen into refactors - that's /simplify's job, separate commit.
4. **One finding = one commit.** Subject per ledger style
   (`area: what changed - key result`), body cites the finding number and the
   findings doc. Never batch findings - bisect must land on one finding.
5. **Add the regression check where the harness supports it.** Server-level
   findings: curl against a loopback serve or the existing gate harness.
   Backend/RPC findings: loopback A/B with the relevant env gate (see
   llamacpp-v100-dev-workflow). Skip only when the check would cost more than
   the bug class warrants - say so in the commit body.
6. **Regate before moving to the next finding** - build + the relevant
   loopback gate, not just "it compiles." A fix that changes RPC/meta/graph
   code re-runs the byte-identity protocol.

## Ordering the list

- Do NOT just go top-to-bottom by severity. Group by decision type:
  mechanical clearly-wrong fixes first (cheap, safe), intent-decision items
  (deliberate-change findings) as explicit decisions with the user,
  topology-dependent items last (they need verification work first).
- Fixes that touch the same file/subsystem: do consecutively to keep the
  regate cost shared, but still commit separately.

## Traps

- "Restoring" deleted code whose deletion WAS the fix (check comments +
  git history for why it was removed - the segfault it fixed will come back).
- Patching one flag/site when the root cause is the shared pattern.
- Fixing a PLAUSIBLE finding that a 10-minute trace would have refuted.
- Batching fixes into one commit "to save regate time" - regate time is
  shared by consecutive ordering, not by batching commits.
- Leaving the findings doc checkboxes stale - the doc is only useful as a
  ledger if it reflects actual state at session end (handoff rule: memory
  next-steps points at the doc + which findings remain open).
