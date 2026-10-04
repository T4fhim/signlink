/* AUTO-GENERATED from packages/schemas/sign_output_request.v1.json. Do not edit; run `pnpm gen:types`. */

/**
 * Request to render a gloss sequence as sign output. Add fields only; never rename or remove.
 */
export interface SignOutputRequestV1 {
  /**
   * Sign language code (ISO 639-3), read from config; never hard-coded.
   */
  lang: string;
  glosses: string[];
  source_text: string;
  [k: string]: unknown;
}
