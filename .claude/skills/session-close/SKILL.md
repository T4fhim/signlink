---
name: session-close
description: Close a SignLnk work session - update PROJECT_CONTEXT.md, docs/CHANGELOG.md and CLAUDE.md commands, and print the Log to context block.
disable-model-invocation: true
---
# Close this session

## Live state
- Branch: !`git branch --show-current`
- Uncommitted: !`git status --short`
- Commits not on main: !`git log --oneline main..HEAD`
- Today: !`date +%Y-%m-%d`

## Do this
1. **Decisions and changes.** List what was decided or changed this session, one line each. Use
   only facts from this session and the git state above; mark anything unmeasured `[VERIFY]`.
2. **PROJECT_CONTEXT.md.** Update `Last updated`, the `## Current phase` line (phase, step, branch,
   what's next), Decision log (decisions only), and Open questions / risks. Keep the file lean:
   history goes to the changelog, not here.
3. **docs/CHANGELOG.md.** Add one dated entry at the top: what was built, PR/branch, measured
   results, follow-ups.
4. **CLAUDE.md.** Add or fix any command created or changed this session. Remove lines that are
   now wrong. Do not add narrative. A lesson that applies to one file area (landmarks, schemas, ml,
   web/privacy, lexicon/data) goes in the matching `.claude/rules/*.md` instead.
5. **Drift check.** If PLAN.md or TOOLKIT.md no longer match reality, list the exact edits needed
   and make them if they are factual corrections.
6. **Print** a block titled `Log to context:` with the one-line entries, then
   `Next step:` with one concrete action, and a suggested session name for `/rename`
   (format `p<phase>-s<step>-<slug>`).
7. Remind me that the claude.ai project copies of PROJECT_CONTEXT.md, CLAUDE.md, PLAN.md and
   TOOLKIT.md mirror the repo and need re-uploading if they changed (docs/WORKFLOW.md §12).
