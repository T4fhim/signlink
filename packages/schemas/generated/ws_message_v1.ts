/* AUTO-GENERATED from packages/schemas/ws_message.v1.json. Do not edit; run `pnpm gen:types`. */

/**
 * WebSocket envelope. Add fields only; never rename or remove.
 */
export interface WsMessageV1 {
  type: "caption" | "gloss" | "sign_seq";
  session: string;
  /**
   * Shape depends on `type`.
   */
  payload: {
    [k: string]: unknown;
  };
  [k: string]: unknown;
}
