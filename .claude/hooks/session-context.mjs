// SessionStart: give Claude the live state CLAUDE.md can't hold (branch, dirty files, phase).
// After compaction (source "compact") it also re-injects the hard rules, which summaries can drop.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { changedFiles, emit, projectDir, readInput, run } from "./lib.mjs";

const input = readInput();
const source = input?.source ?? "startup";

const branch = run("git", ["branch", "--show-current"]).out || "(detached)";
const dirty = changedFiles();

let phase = "unknown (see PROJECT_CONTEXT.md)";
try {
  const ctx = readFileSync(join(projectDir, "PROJECT_CONTEXT.md"), "utf8");
  const m = ctx.match(/## Current phase\s*\n+([^\n]+)/);
  if (m) phase = m[1].trim();
} catch {
  /* file missing: keep default */
}

const shown = dirty.slice(0, 8).join(", ") + (dirty.length > 8 ? ", ..." : "");
const lines = [
  `SignLnk session (${source}). Branch: ${branch}. Uncommitted files: ${dirty.length}${dirty.length ? ` (${shown})` : ""}.`,
  `Current phase: ${phase}`,
];

if (source === "compact") {
  lines.push(
    "Context was compacted. Hard rules still apply: $0/OSS only; raw video/audio never leave the device; ASL (ase) <-> English (en) read from config; release vs research data tracks; schemas add-only + regenerate types; normalization mirrored TS<->Py with parity tests; never invent numbers, sizes or licences ([VERIFY]). Re-read the current PLAN step before continuing.",
  );
}

emit({
  hookSpecificOutput: { hookEventName: "SessionStart", additionalContext: lines.join("\n") },
});
