// PostToolUse (Edit|Write): format the edited file, then remind Claude of the follow-up a rule
// requires. Never blocks: the edit already happened, so feedback goes in additionalContext.
import { existsSync } from "node:fs";
import { join } from "node:path";
import {
  NORMALIZATION,
  emit,
  isSchema,
  isTrainingConfig,
  projectDir,
  readInput,
  repoPath,
  run,
  venvMatchesPlatform,
} from "./lib.mjs";

const input = readInput();
const p = repoPath(input?.tool_input?.file_path);
if (!p) process.exit(0);

const notes = [];

// 1. Format. Prettier respects .prettierignore (markdown is hand-formatted); ruff respects
//    pyproject's extend-exclude. Failures are reported, not fatal.
const prettierBin = join(projectDir, "node_modules", "prettier", "bin", "prettier.cjs");
if (/\.(ts|tsx|js|mjs|cjs|json|ya?ml|css)$/.test(p) && existsSync(prettierBin)) {
  const r = run(process.execPath, [
    prettierBin,
    "--write",
    "--ignore-unknown",
    "--log-level",
    "warn",
    p,
  ]);
  if (!r.ok) notes.push(`prettier failed on ${p}: ${r.out.split("\n")[0]}`);
} else if (p.endsWith(".py") && venvMatchesPlatform()) {
  const r = run("uv", ["run", "--quiet", "ruff", "format", p]);
  if (!r.ok) notes.push(`ruff format failed on ${p}: ${r.out.split("\n")[0]}`);
}

// 2. Rule reminders.
if (isSchema(p)) {
  notes.push(
    "Schema changed (rule 5): fields may only be added. Run `pnpm gen:types`, update packages/schemas/examples/, then `pnpm gen:types:check`. A breaking change needs a new .v2 file and an ADR.",
  );
}
if (NORMALIZATION.ts.test(p) || NORMALIZATION.py.test(p)) {
  const other = NORMALIZATION.ts.test(p) ? "ml/signlnk_ml/features/" : "packages/landmarks/src/";
  notes.push(
    `Normalization changed (rule 6): mirror the change in ${other} and run both parity suites: \`pnpm --filter @signlnk/landmarks test\` and \`uv run pytest ml/tests/test_normalize.py ml/tests/test_aspect.py\`. Regenerate golden fixtures only if the change is intended.`,
  );
}
if (
  p.startsWith("packages/landmarks/src/layoutSpec") ||
  p.startsWith("packages/schemas/layouts/")
) {
  notes.push(
    "Layout changed: run `pnpm --filter @signlnk/landmarks write-layout` and the geometry/parity tests; ADR-0004 governs the index subset.",
  );
}
if (isTrainingConfig(p)) {
  notes.push(
    "Training config (rule 4): it must declare `track: release` or `track: research`. Release configs may only use release-track datasets from docs/datasets.md.",
  );
}
if (p.endsWith("package.json") || p.endsWith("pyproject.toml")) {
  notes.push(
    "Dependency manifest changed (rule 1): new dependencies must be free/OSS with a known licence and pinned exactly. Ask the license-auditor subagent if unsure.",
  );
}

if (notes.length === 0) process.exit(0);
emit({ hookSpecificOutput: { hookEventName: "PostToolUse", additionalContext: notes.join("\n") } });
