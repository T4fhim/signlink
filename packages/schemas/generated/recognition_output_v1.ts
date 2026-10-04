/* AUTO-GENERATED from packages/schemas/recognition_output.v1.json. Do not edit; run `pnpm gen:types`. */

/**
 * One recognized sign emitted by the live decoder. Add fields only; never rename or remove.
 */
export interface RecognitionOutputV1 {
  /**
   * ID-gloss, e.g. THANK-YOU.
   */
  gloss: string;
  /**
   * Language-qualified gloss id: <lang>:<ID-GLOSS>.
   */
  gloss_id: string;
  confidence: number;
  /**
   * Sign start, ms.
   */
  t_start: number;
  /**
   * Sign end, ms.
   */
  t_end: number;
  /**
   * Highest-probability candidates, best first.
   */
  top_k: TopKEntry[];
  /**
   * SemVer, e.g. isr-0.3.1.
   */
  model_version: string;
  /**
   * CalVer YYYY.MM.N.
   */
  lexicon_version: string;
  [k: string]: unknown;
}
export interface TopKEntry {
  gloss_id: string;
  p: number;
  [k: string]: unknown;
}
