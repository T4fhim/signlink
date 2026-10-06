---
paths:
  - ".claude/**"
  - "CLAUDE.md"
  - "PROJECT_CONTEXT.md"
  - "docs/WORKFLOW.md"
  - "docs/TOOLKIT.md"
---
<!-- Loads when Claude edits the Claude Code framework or the always-loaded context files. -->
# Rules for editing the Claude framework

- CLAUDE.md and PROJECT_CONTEXT.md load every session: keep them short. History goes to
  `docs/CHANGELOG.md`, procedures to skills or `docs/WORKFLOW.md`, file-area rules to `.claude/rules/`.
- A rule Claude keeps breaking becomes a hook; a procedure used sometimes becomes a skill; a noisy or
  independent check becomes a subagent.
- Hooks run as `node <script>` (exec form) so they work on Windows. Test a hook by piping JSON into it
  before relying on it, for example
  `echo '{"tool_input":{"file_path":"uv.lock"}}' | node .claude/hooks/guard-paths.mjs`.
- Keep the `## Current phase` heading in PROJECT_CONTEXT.md with the status on the next line: the
  SessionStart and UserPromptSubmit hooks read it.
- When adding or removing a hook, skill, subagent or rule, update the tables in `docs/WORKFLOW.md` §8
  and `docs/TOOLKIT.md` §2 in the same change.
