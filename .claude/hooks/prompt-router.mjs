// UserPromptSubmit: route the prompt to the SignLnk skill, subagent or rule that the situation
// calls for. Silent unless something matches, so ordinary prompts cost no extra context.
// Slash commands (prompts starting with "/") are skipped: the skill already carries its procedure.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { emit, projectDir, readInput } from "./lib.mjs";

const input = readInput();
const prompt = String(input?.prompt ?? "").trim();
if (!prompt || prompt.startsWith("/")) process.exit(0);
const t = prompt.toLowerCase();

let phase = null;
try {
  const ctx = readFileSync(join(projectDir, "PROJECT_CONTEXT.md"), "utf8");
  const m = ctx.match(/## Current phase\s*\n+[^\n]*?Phase (\d)/);
  if (m) phase = Number(m[1]);
} catch {
  /* no context file: skip phase check */
}

const ROUTES = [
  {
    re: /\b(data ?set|licen[cs]e|wlasl|how2sign|asl citizen|sem-?lex|asl-?lex|signbank|pretrained|weights|corpus)\b/,
    say: "Dataset/licence topic: run `/dataset-license` before relying on it and record it in docs/datasets.md (rule 4, rule 7).",
  },
  {
    re: /\b(schema|landmark_layout|recognition_output|ws_message|lexicon_entry|sign_output_request|new field|add a field)\b/,
    say: "Interface topic: follow `/schema-change` (add-only, regenerate types, update examples).",
  },
  {
    re: /\b(commit|push|pull request|open a pr|merge)\b/,
    say: "Before committing: `/verify` must pass. If a PLAN step is finishing, run the `plan-reviewer` subagent first. Don't push unless asked.",
  },
  {
    re: /\b(next step|start (the )?step|phase \d+,? step \d+|step \d+ of phase|continue (with )?(the )?plan)\b/,
    say: "PLAN step work: use `/step <phase.step>` (scope check, Done-when, tests first, /verify, plan-reviewer).",
  },
  {
    re: /\b(phase (is )?(done|complete|finished)|close (the )?phase|phase gate|move (on )?to phase|start phase)\b/,
    say: "Phase boundary: run `/phase-gate <n>` on Opus before starting the next phase.",
  },
  {
    re: /\b(wrap up|end (of )?(the )?session|done for (the )?(day|now)|log to context|session close|close (the )?session)\b/,
    say: "Session end: run `/session-close`.",
  },
  {
    re: /\b(we decided|decision|trade-?off|adr)\b/,
    say: "If this settles a lasting choice, record it with `/adr` and a Decision-log line in PROJECT_CONTEXT.md.",
  },
  {
    re: /\b(deaf|signers?|advisors?|public demo|release|launch|show (it|this) to)\b/,
    say: "Anything Deaf users will see needs its community gate (PLAN §9) before it is called correct.",
  },
  {
    re: /\b(camera|webcam|microphone|upload|telemetry|analytics|cdn|third[- ]party|send (the )?(video|audio|frames|landmarks))\b/,
    say: "Privacy-sensitive area (rule 2): raw video/audio and landmarks stay on device; ask `privacy-reviewer` to check the diff.",
  },
  {
    re: /\b(add|install|upgrade|bump) (a |the )?(dependency|package|library|lib)\b|\b(pnpm add|uv add|npm install)\b/,
    say: "New dependency (rule 1): free/OSS, known licence, pinned exactly; `license-auditor` if unsure. Dependency adds ask for approval.",
  },
];

const later = [
  [2, /\b(continuous signing|ctc|gloss wer)\b/],
  [3, /\b(whisper|speech[- ]to[- ]text|vad|clip renderer|studio v2)\b/],
  [4, /\b(translation model|seq2seq|chrf|bleu|studio v3)\b/],
  [5, /\b(avatar|pose renderer|three\.js|gltf|blender|studio v4)\b/],
  [6, /\b(tauri|virtual cam(era)?|zoom|teams|webrtc|overlay window)\b/],
];

const notes = ROUTES.filter((r) => r.re.test(t)).map((r) => r.say);
if (phase !== null) {
  const hit = later.find(([p, re]) => p > phase + 1 && re.test(t));
  if (hit)
    notes.push(
      `This touches Phase ${hit[0]} work; current phase is ${phase}. Flag it in one line and offer only the minimum needed now.`,
    );
}

if (notes.length === 0) process.exit(0);
emit({
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: `SignLnk routing:\n- ${notes.slice(0, 4).join("\n- ")}`,
  },
});
