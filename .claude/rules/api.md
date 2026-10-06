---
paths:
  - "services/**"
---
<!-- Loads when Claude works on the FastAPI service / Studio backend (Phase 1 step 8). -->
# API / Studio backend rules

- Security baseline (PLAN §6.5): invite-only accounts, argon2id password hashing, httpOnly session
  cookies, CSRF tokens, login rate limits. Every write goes to `audit_log`.
- Uploads: size, duration and MIME limits; re-encode video server-side with ffmpeg before storage.
- Consent revocation deletes the contributor's examples and clips; the next release rebuilds prototypes.
- Landmarks and clips are personal data: no third-party services, no analytics, no secrets in the repo
  (CLAUDE.md rules 1–2). The public demo must keep working with no server.
- API payloads follow `packages/schemas/*.json`; Pydantic types are generated, never hand-edited (rule 5).
- Postgres in Docker Compose; SQLite allowed in dev. Migrations are reviewed like code.
- mypy strict, type hints everywhere, tests for every endpoint including the unauthorised path.
