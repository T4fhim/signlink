---
name: adr
description: Write the next SignLnk architecture decision record in docs/adr using the project's ADR format.
argument-hint: "[short title]"
disable-model-invocation: true
---
# New ADR: $ARGUMENTS

Existing ADRs: !`ls docs/adr`

1. Pick the next free number (4 digits). Numbers 0001–0003 are reserved for the PLAN Phase 0
   step 7 ADRs (landmarks not pixels, gloss layer, two data tracks) until they are written.
2. Create `docs/adr/NNNN-<kebab-title>.md` following the style of the newest ADR:

```markdown
# ADR-NNNN: <Title>

Status: proposed | accepted · YYYY-MM-DD · Phase <P> step <S>
<Relation to other ADRs: builds on / supersedes / resolves>

## Context
<Problem, constraints, measured evidence with how it was measured. No invented numbers.>

## Decision
<What we do, stated so it can be checked.>

## Consequences
<What changes, what it costs, what it rules out, follow-ups.>

## Verification
<Test, fixture or measurement that shows the decision holds.>
```

3. If the decision changes an interface, data track or hard rule, also add a one-line entry to the
   Decision log in `PROJECT_CONTEXT.md`.
4. Show me the ADR before marking it `accepted`.
