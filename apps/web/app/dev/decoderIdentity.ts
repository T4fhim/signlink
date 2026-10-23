/**
 * What the /dev pages tell the decoder about the model they load. One place, so no page carries
 * its own literal (CLAUDE.md rule 3: the language comes from config, not code). The model's
 * `model.json` does not record a language yet; when it (or the app's config) does, read it here.
 *
 * `lexiconVersion` is a placeholder until a lexicon bundle exists (Phase 1 step 7+); `modelVersion`
 * is a dev label, not a release version.
 */
export const DECODER_IDENTITY = {
  language: "ase",
  modelVersion: "isr-dev",
  lexiconVersion: "0000.0.0",
} as const;
