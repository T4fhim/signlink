---
name: schema-change
description: Make an add-only change to a SignLnk JSON Schema in packages/schemas and propagate it to generated TS + Pydantic types, examples and docs.
argument-hint: "[schema file] [what to add]"
disable-model-invocation: true
---
# Schema change: $ARGUMENTS

Current schemas: !`ls packages/schemas`

Rules (CLAUDE.md rule 5, PLAN §4): schemas are the source of truth; fields may only be **added**;
renaming, removing, retyping or tightening a field is breaking and needs a new `.v2` file plus an
ADR. New fields should be optional unless every producer already emits them.

1. State whether the change is additive. If not, stop and propose the `.v2` + ADR route.
2. Edit the schema in `packages/schemas/<name>.v1.json` (never the generated files; a hook blocks
   that).
3. Update `packages/schemas/examples/<name>.v1.example.json` so the example exercises the new field.
4. Run `pnpm gen:types`, then `pnpm gen:types:check`.
5. Update producers and consumers that should emit or read the field (TS and Python), with tests.
6. If the field changes meaning for the landmark layout or recognition output, update
   PLAN §4 and PROJECT_CONTEXT "Interfaces" in the same change.
7. Run `/verify schemas` and then the full `/verify`.
8. Summarise: field, type, optional/required, producers, consumers, files changed.
