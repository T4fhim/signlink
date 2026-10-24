// SessionStart + UserPromptSubmit: twice per calendar day, make Claude ask which PC is the benchmark.
// The day has two slots split at SWITCH_MINUTES (the user moves from the home PC to the work PC at
// 10:15): "am" before it, "pm" from it. The answer lives in the gitignored CLAUDE.local.md as
// "Benchmark PC: YYYY-MM-DD am|pm — <pc>" (a date-only line counts as "am"). Silent when that line
// names today's date and slot, so it speaks at the first session or prompt of each slot (and keeps
// speaking until the answer is written, e.g. across 10:15 in a long session).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { emit, projectDir, readInput } from "./lib.mjs";

const input = readInput();
const event = input?.hook_event_name ?? "SessionStart";

const SWITCH_MINUTES = 10 * 60 + 15;
const now = new Date();
const pad = (n) => String(n).padStart(2, "0");
const today = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
const slot = now.getHours() * 60 + now.getMinutes() >= SWITCH_MINUTES ? "pm" : "am";

let last = null;
try {
  const text = readFileSync(join(projectDir, "CLAUDE.local.md"), "utf8");
  const m = text.match(/^Benchmark PC:\s*(\d{4}-\d{2}-\d{2})(?:\s+(am|pm))?/m);
  if (m) last = `${m[1]} ${m[2] ?? "am"}`;
} catch {
  /* file missing: treated as never answered */
}

if (last === `${today} ${slot}`) process.exit(0);

const message =
  `BENCHMARK PC (${slot === "am" ? "before" : "from"} 10:15): CLAUDE.local.md has no "Benchmark PC" ` +
  `line for ${today} ${slot} (last answer: ${last ?? "never"}). Before anything else, ask the user ` +
  `which PC is the benchmark now (home PC, AMD Ryzen 7 7840H, or the work PC), then set the line ` +
  `"Benchmark PC: ${today} ${slot} — <pc>, <cpu>, <OS>" in CLAUDE.local.md, replacing any older one. ` +
  `Never write it from a guess. Name that machine in every benchmark or latency number you report.`;

emit({ hookSpecificOutput: { hookEventName: event, additionalContext: message } });
