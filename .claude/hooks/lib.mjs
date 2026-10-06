// Shared helpers for SignLnk Claude Code hooks.
// Hooks run in exec form (`node <script>`), so they work the same on Windows, macOS and Linux
// and need no jq or shell. Docs: https://code.claude.com/docs/en/hooks-guide
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";

export const projectDir = resolve(process.env.CLAUDE_PROJECT_DIR ?? process.cwd());

/** Read the hook's JSON input from stdin. Returns {} on empty or invalid input. */
export function readInput() {
  try {
    const raw = readFileSync(0, "utf8");
    return raw.trim() ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

/** Repo-relative POSIX path for a tool's file_path, or null when outside the repo. */
export function repoPath(filePath) {
  if (!filePath || typeof filePath !== "string") return null;
  const abs = isAbsolute(filePath) ? filePath : resolve(projectDir, filePath);
  const rel = relative(projectDir, abs).replace(/\\/g, "/");
  if (rel.startsWith("..") || isAbsolute(rel)) return null;
  return rel;
}

/** Print a JSON object to stdout and exit 0 (Claude Code parses it). */
export function emit(obj) {
  process.stdout.write(JSON.stringify(obj));
  process.exit(0);
}

/** Run a command without a shell. Returns {ok, out}. Never throws. */
export function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, {
    cwd: projectDir,
    encoding: "utf8",
    timeout: opts.timeout ?? 60_000,
    env: { ...process.env, GIT_OPTIONAL_LOCKS: "0" },
    windowsHide: true,
  });
  const stdout = r.stdout ?? "";
  return { ok: r.status === 0, stdout, out: `${stdout}${r.stderr ?? ""}`.trim() };
}

/**
 * True when it is safe to call `uv run` here. The repo's .venv is created on one OS; running uv
 * from another OS over the same checkout (e.g. a Linux VM on the Windows folder) would try to
 * rebuild it. A missing .venv is fine: uv creates it as `uv sync` would.
 */
export function venvMatchesPlatform() {
  const venv = join(projectDir, ".venv");
  if (!existsSync(venv)) return true;
  const win = existsSync(join(venv, "Scripts"));
  return process.platform === "win32" ? win : !win;
}

/** Changed + untracked files (repo-relative), using a lock-free git read. */
export function changedFiles() {
  const r = run("git", ["status", "--porcelain", "--untracked-files=all"]);
  if (!r.ok) return [];
  // Use untrimmed stdout: porcelain lines start with a status column that may be a space.
  return r.stdout
    .split(/\r?\n/)
    .filter((l) => /^[ MADRCU?!]{2} /.test(l))
    .map((l) => l.slice(3).replace(/^"|"$/g, ""))
    .map((p) => (p.includes(" -> ") ? p.split(" -> ")[1] : p));
}

// ---- Project rules shared by several hooks -------------------------------------------------

/** Top-level JSON Schemas (source of truth, CLAUDE.md rule 5). */
export const isSchema = (p) => /^packages\/schemas\/[a-z_]+\.v\d+\.json$/.test(p);

/** The two mirrored normalization implementations (CLAUDE.md rule 6). */
export const NORMALIZATION = {
  ts: /^packages\/landmarks\/src\/(normalize|aspect)\.ts$/,
  py: /^ml\/signlnk_ml\/features\/(normalize|aspect)\.py$/,
};

/** Training configs must declare their data track (CLAUDE.md rule 4). */
export const isTrainingConfig = (p) => /^ml\/configs\/.+\.ya?ml$/.test(p);
