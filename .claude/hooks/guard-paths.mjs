// PreToolUse (Edit|Write|NotebookEdit): block edits that break a SignLnk hard rule.
// Returns permissionDecision "deny" with a reason Claude can act on. A deny here holds in every
// permission mode, including auto and bypassPermissions.
import { emit, readInput, repoPath } from "./lib.mjs";

const RULES = [
  {
    test: (p) =>
      p.startsWith("packages/schemas/generated/") ||
      p.startsWith("ml/signlnk_ml/schemas_generated/"),
    why: "Generated file. Edit the schema in packages/schemas/*.json, then run `pnpm gen:types` (CLAUDE.md rule 5).",
  },
  {
    test: (p) => p === "pnpm-lock.yaml" || p === "uv.lock",
    why: "Lockfile. Change dependencies with pnpm/uv (`pnpm add -E`, `uv add`), never by hand.",
  },
  {
    test: (p) => p.startsWith("apps/web/public/mediapipe/"),
    why: "MediaPipe assets are fetched by `pnpm assets` from scripts/mediapipe-assets.json. Edit that manifest instead.",
  },
  {
    test: (p) => /(^|\/)\.env(\..+)?$/.test(p) && !p.endsWith(".env.example"),
    why: "Secrets/env file. Edit .env.example for documented defaults; real values stay on the developer's machine.",
  },
  {
    test: (p) =>
      /\.(webm|mp4|mov|mkv|wav|npy|npz|parquet)$/i.test(p) && !p.startsWith("tests/fixtures/"),
    why: "Recordings and landmark data are personal data and never live in the repo. Write them under $SIGNLNK_DATA_DIR; only synthetic golden fixtures go in tests/fixtures/ (CLAUDE.md rule 2).",
  },
  {
    test: (p) => p.startsWith("lexicon/core/") && /asl-?lex|signbank/i.test(p),
    why: "lexicon/core/ is CC BY 4.0 SignLnk content only. ASL-LEX / Signbank data belongs in lexicon/reference/ (CLAUDE.md rule 4).",
  },
];

const input = readInput();
const p = repoPath(input?.tool_input?.file_path ?? input?.tool_input?.notebook_path);
if (!p) process.exit(0);

const hit = RULES.find((r) => r.test(p));
if (!hit) process.exit(0);

emit({
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: `Blocked edit to ${p}: ${hit.why}`,
  },
});
