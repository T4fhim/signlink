# ADR-0002: Gloss layer as the shared intermediate representation

Status: proposed · 2026-10-06 · Phase 0 step 7
Relates to PLAN §4.2 (recognition output), §4.4 (lexicon entry), §2.3 (Direction B).

## Context

Both directions need a common middle form. Direction A (sign → English) recognizes signs; Direction B
(English → sign) must pick signs to show. Sign languages are not word-for-word English, and the project
must run without paid APIs or an LLM (CLAUDE.md rule 1).

## Decision

A **gloss** is the unit between the sign model and the language layer. It is identified by a stable
`gloss_id` of the form `<lang>:<ID-GLOSS>` (for example `ase:THANK-YOU`), defined by a lexicon entry
(PLAN §4.4). The language code comes from config, never hard-coded (rule 3).

- Direction A: recognizer → gloss events → (Phase 4) gloss→English → `CaptionSink`.
- Direction B: English → gloss → `SignOutputRequest` → `SignRenderer`.
- Phase 1–3 need no translation model: captions are the gloss's English translation from the lexicon,
  and English→gloss is a dictionary lookup.
- An LLM may post-process, behind a feature flag, and is never required.

## Consequences

- Recognizer, lexicon, Studio, renderers and translators meet at one small interface; each can be
  replaced independently.
- Output fields `gloss`, `gloss_id`, `model_version`, `lexicon_version` are part of the stable
  `recognition_output` schema: add-only (rule 5).
- Glosses are labels, not a faithful rendering of ASL grammar. Anything shown to Deaf users needs its
  community gate (PLAN §9).
- ID-gloss naming follows Signbank conventions only; no ASL-LEX/Signbank descriptive data is copied
  (rule 4, ADR-0003).

## Verification

- `packages/schemas/recognition_output.v1.json` and `lexicon_entry.v1.json` carry the fields; CI fails
  if the generated TS/Pydantic types are stale (`pnpm gen:types:check`).
- Later phases: gloss WER (Phase 2) and chrF/BLEU (Phase 4) are measured on gloss sequences and translation output.
