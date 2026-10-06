---
name: verify
description: Run SignLnk's full local check suite (the same gates as CI) and report pass/fail with evidence. Use before every commit, after finishing a PLAN step, or when asked to verify changes.
context: fork
agent: test-runner
background: false
allowed-tools: Bash(pnpm lint) Bash(pnpm typecheck) Bash(pnpm test) Bash(pnpm gen:types:check) Bash(uv run ruff check *) Bash(uv run ruff format *) Bash(uv run mypy *) Bash(uv run pytest *) Bash(uv run pytest)
---
Run the SignLnk verification suite from the repository root and report the result.

Scope requested by the caller (empty means everything): $ARGUMENTS

Run, in order, continuing after failures:

1. `pnpm lint`
2. `pnpm typecheck`
3. `pnpm test`
4. `pnpm gen:types:check`
5. `uv run ruff check .`
6. `uv run ruff format --check .`
7. `uv run mypy ml services`
8. `uv run pytest`

If the scope names an area, run only the matching commands (for example "python" → 5–8,
"schemas" → 4, "parity" → `pnpm --filter @signlnk/landmarks test` and
`uv run pytest ml/tests/test_normalize.py ml/tests/test_aspect.py ml/tests/test_geometry_parity.py`).

Report: first line `ALL PASS` or `FAILURES: n`, then one ✓/✗ line per command, then for each
failure the test or file:line, at most 10 lines of error, and the likely root cause. Mention any
real-data tests skipped because `SIGNLNK_DATA_DIR` is unset. Do not fix anything.
