---
paths:
  - "packages/schemas/**"
  - "ml/signlnk_ml/schemas_generated/**"
  - "scripts/gen-types.mjs"
---
<!-- Loads when Claude works on schemas or generated types. -->
# Schema rules (CLAUDE.md rule 5, PLAN §4)

- Use the `/schema-change` procedure for any schema edit.
- `packages/schemas/*.v1.json` are the source of truth. Add optional fields only; renaming, removing,
  retyping or tightening needs a `.v2` file and an ADR.
- Never edit `packages/schemas/generated/` or `ml/signlnk_ml/schemas_generated/` (a hook blocks it).
  Run `pnpm gen:types`, then `pnpm gen:types:check`.
- Keep `packages/schemas/examples/*.example.json` exercising every field; `src/examples.test.ts` and
  `ml/tests/test_schemas.py` validate them.
- If a field's meaning changes for the landmark frame or recognition output, update PLAN §4 and the
  Interfaces section of PROJECT_CONTEXT.md in the same change.
