/* AUTO-GENERATED from packages/schemas/lexicon_entry.v1.json. Do not edit; run `pnpm gen:types`. */

/**
 * One sign in the lexicon. Phonology field names follow ASL-LEX / Signbank coding concepts; no descriptive data is copied from them. Add fields only.
 */
export interface LexiconEntryV1 {
  gloss_id: string;
  lang: string;
  id_gloss: string;
  variant: number;
  translations: Translation[];
  phonology?: Phonology;
  lexical_class?: string;
  register_tags?: string[];
  regional_tags?: string[];
  status: "draft" | "in_review" | "approved" | "released" | "rejected";
  recognition?: Recognition;
  clips?: Clip[];
  external_refs?: ExternalRef[];
  license: string;
  revision: number;
  [k: string]: unknown;
}
export interface Translation {
  lang: string;
  text: string;
  sense?: string;
  [k: string]: unknown;
}
export interface Phonology {
  handshape_dom?: string | null;
  handshape_nondom?: string | null;
  sign_type?: string | null;
  location?: string | null;
  movement?: string | null;
  contact?: boolean | null;
  non_manual?: string[];
  [k: string]: unknown;
}
export interface Recognition {
  enabled: boolean;
  n_examples: number;
  n_signers: number;
  [k: string]: unknown;
}
export interface Clip {
  clip_id: string;
  license: string;
  signer_id: string;
  [k: string]: unknown;
}
export interface ExternalRef {
  source: string;
  id: string;
  url: string;
  [k: string]: unknown;
}
