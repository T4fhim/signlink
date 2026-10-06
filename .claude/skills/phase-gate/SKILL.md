---
name: phase-gate
description: Run the SignLnk phase-boundary review - confirm the phase's Done-when criteria, community gate, toolkit re-audit, CLAUDE.md pruning, and next-phase setup.
argument-hint: "[phase number being closed]"
disable-model-invocation: true
model: opus
---
# Phase gate: closing Phase $ARGUMENTS

Run this on Opus (it is set in this skill's frontmatter). Each item needs evidence, not assertion.

1. **Done when.** For every step of the phase in `docs/PLAN.md` §7 and the roadmap row in
   `PROJECT_CONTEXT.md`, show the evidence (test, measured number with method, PR). List any gap.
2. **Community gate.** Which gate in PLAN §9 applies (G0–G5)? Is its sign-off in `docs/review/`?
   Nothing shown to Deaf users is called "correct" without it.
3. **Plan review.** Use the `plan-reviewer` subagent on the phase as a whole. Then review PLAN.md
   for the next phase: what assumptions changed, what should be re-scoped. Propose edits; do not
   re-litigate logged decisions without a stated reason.
4. **CLAUDE.md pruning.** For each line ask "would removing this cause a mistake?" Cut what the code
   now makes obvious, move sometimes-relevant procedures into skills, file-area rules into
   `.claude/rules/` (add a rules file for each new area the next phase opens, e.g. `services/api/**`
   in Phase 1), and turn rules Claude keeps breaking into hooks. Update the prompt-router routes and
   the later-phase keywords in `.claude/hooks/prompt-router.mjs` for the new phase. Suggest I run `/doctor` for its proposed cuts and `/context` to check what
   loads.
5. **Toolkit re-audit.** Update `docs/TOOLKIT.md`: promote LATER items for the next phase, demote
   unused ones. Suggest I run `/skill-doctor` and turn off plugin skills that cost context but are
   never used; `/hooks` to confirm hooks; `/status` to confirm settings sources.
6. **Datasets and dependencies.** Ask `license-auditor` to re-check everything the next phase will
   use.
7. **Close.** Update `PROJECT_CONTEXT.md` (phase, roadmap), add a CHANGELOG entry, and print
   `Log to context:` and `Next step:`.
