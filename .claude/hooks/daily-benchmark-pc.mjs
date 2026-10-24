// SessionStart + UserPromptSubmit: once per calendar day, make Claude ask which PC is the benchmark.
// The answer lives in the gitignored CLAUDE.local.md as "Benchmark PC: YYYY-MM-DD — <pc>". Silent
// when that line is dated today, so it only speaks at the first session or prompt of a new day (and
// keeps speaking until the answer is written, e.g. across midnight in a long session).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { emit, projectDir, readInput } from "./lib.mjs";

const input = readInput();
const event = input?.hook_event_name ?? "SessionStart";

const now = new Date();
const pad = (n) => String(n).padStart(2, "0");
const today = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;

let last = null;
try {
  const text = readFileSync(join(projectDir, "CLAUDE.local.md"), "utf8");
  const m = text.match(/^Benchmark PC:\s*(\d{4}-\d{2}-\d{2})/m);
  if (m) last = m[1];
} catch {
  /* file missing: treated as never answered */
}

if (last === today) process.exit(0);

const message =
  `BENCHMARK PC (daily): CLAUDE.local.md has no "Benchmark PC" line dated ${today} ` +
  `(last answer: ${last ?? "never"}). Before anything else today, ask the user which PC is today's ` +
  `benchmark (home PC, AMD Ryzen 7 7840H, or the work PC), then set the line ` +
  `"Benchmark PC: ${today} — <pc>, <cpu>, <OS>" in CLAUDE.local.md, replacing any older one. ` +
  `Never write it from a guess. Name that machine in every benchmark or latency number you report.`;

emit({ hookSpecificOutput: { hookEventName: event, additionalContext: message } });
