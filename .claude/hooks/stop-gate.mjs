// Stop: deterministic end-of-turn gate for the rules a reviewer can't be trusted to remember.
//  - blocks when a schema changed but generated types are stale (rule 5)
//  - blocks when a training config lacks `track:` (rule 4)
//  - reminds (non-blocking) when only one normalization implementation changed (rule 6)
// Cheap by design: it only runs a check when the relevant files are dirty.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  NORMALIZATION,
  changedFiles,
  emit,
  isSchema,
  isTrainingConfig,
  projectDir,
  readInput,
  run,
  venvMatchesPlatform,
} from "./lib.mjs";

const input = readInput();
// Already continuing because of this hook: let Claude stop (avoids block loops).
if (input?.stop_hook_active === true) process.exit(0);

const files = changedFiles();
if (files.length === 0) process.exit(0);

const blocks = [];
const notes = [];

if (files.some(isSchema) && !venvMatchesPlatform()) {
  notes.push(
    "A schema changed but the stale-types check was skipped: .venv belongs to another OS. Run `pnpm gen:types:check` on the machine that owns .venv.",
  );
} else if (files.some(isSchema)) {
  const r = run(process.execPath, [join(projectDir, "scripts", "gen-types.mjs"), "--check"], {
    timeout: 120_000,
  });
  if (!r.ok) {
    const tail = r.out.split("\n").slice(-5).join("\n");
    blocks.push(
      `Generated types are stale after a schema change. Run \`pnpm gen:types\`.\n${tail}`,
    );
  }
}

for (const cfg of files.filter(isTrainingConfig)) {
  try {
    const text = readFileSync(join(projectDir, cfg), "utf8");
    if (!/^\s*track:\s*(release|research)\s*$/m.test(text)) {
      blocks.push(`${cfg} has no \`track: release|research\` line (CLAUDE.md rule 4).`);
    }
  } catch {
    /* deleted file */
  }
}

const tsChanged = files.some((f) => NORMALIZATION.ts.test(f));
const pyChanged = files.some((f) => NORMALIZATION.py.test(f));
if (tsChanged !== pyChanged) {
  notes.push(
    `Only the ${tsChanged ? "TypeScript" : "Python"} normalization changed. If behaviour changed, mirror it and run both parity suites (rule 6); if not (comments, refactor), say so in the summary.`,
  );
}

if (blocks.length) emit({ decision: "block", reason: blocks.join("\n\n") });
if (notes.length)
  emit({ hookSpecificOutput: { hookEventName: "Stop", additionalContext: notes.join("\n") } });
process.exit(0);
