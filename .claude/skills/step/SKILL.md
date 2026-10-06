---
name: step
description: Run one SignLnk PLAN.md build step end to end - scope check, explore, plan, test-first implementation, verification, adversarial review, commit.
argument-hint: "[phase.step, e.g. 0.6 or 1.3]"
disable-model-invocation: true
---
# Run PLAN step $ARGUMENTS

## Live state
- Branch: !`git branch --show-current`
- Uncommitted: !`git status --short`
- Recent commits: !`git log --oneline -5`

## Procedure (see docs/WORKFLOW.md §3 for the reasoning)

1. **Scope.** Read the `## Current phase` section of `PROJECT_CONTEXT.md` and the step's text in
   `docs/PLAN.md` §7. If the step belongs to a later phase, stop and say so in one line. Quote the
   step's **Done when** line back to me; that is the acceptance test.
2. **Branch.** Work on `feat/phase<P>-<slug>` created from an up-to-date `main`. If uncommitted work
   from another step is present, stop and ask.
3. **Explore, then plan.** If the diff can't be described in one sentence, investigate first
   (use a subagent for wide searches so the main context stays small) and write a short plan:
   files to touch, interfaces affected, tests to add, how "Done when" will be measured. Wait for
   my approval when the step changes an interface in `packages/schemas` or `slk-landmarks-v1`.
4. **Tests first.** Write failing tests or a measurement harness for the Done-when criterion.
   Parity work needs golden fixtures in `tests/fixtures/` used by both TS and Python.
5. **Implement** in small Conventional Commits. Hooks format files and remind you of rules 4–6; act
   on every reminder.
6. **Verify.** Run `/verify`. Fix failures at the root cause, never by skipping or loosening a test.
7. **Review in a fresh context.** Ask the `plan-reviewer` subagent to review step $ARGUMENTS.
   Also ask `privacy-reviewer` if capture, storage, networking or scripts changed, and
   `license-auditor` if dependencies, datasets, configs or the lexicon changed. Fix blockers;
   treat "Optional" items as optional.
8. **Record.** New decision → ADR via `/adr`. New command → `CLAUDE.md` Commands. Measured
   numbers go in the ADR or changelog with how they were measured.
9. **Stop and summarise** with evidence: commands run and their results, measured numbers, and
   what's left. If the step is marked 🧑‍🤝‍🧑 or touches what Deaf users will see, say which review
   gate (PLAN §9) it needs. Do not push or open a PR unless I ask.
