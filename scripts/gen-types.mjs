// Generates TS types (json-schema-to-typescript) and Pydantic v2 models (datamodel-code-generator)
// from packages/schemas/*.v<N>.json. Schemas are the source of truth.
//   node scripts/gen-types.mjs           regenerate in place
//   node scripts/gen-types.mjs --check   generate to a temp dir and fail if committed output differs
import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemasDir = join(root, "packages", "schemas");
const committed = {
  ts: join(schemasDir, "generated"),
  py: join(root, "ml", "signlnk_ml", "schemas_generated"),
};
const check = process.argv.includes("--check");

const schemaFiles = readdirSync(schemasDir)
  .filter((f) => /^[a-z_]+\.v\d+\.json$/.test(f))
  .sort();
if (schemaFiles.length === 0) {
  console.error("gen-types: no schemas found in packages/schemas");
  process.exit(1);
}

const moduleName = (file) => file.replace(/\.json$/, "").replace(/\./g, "_");
const lf = (text) => text.replace(/\r\n/g, "\n");

const tmpRoot = check ? mkdtempSync(join(tmpdir(), "signlnk-gen-")) : null;
const out = {
  ts: check ? join(tmpRoot, "ts") : committed.ts,
  py: check ? join(tmpRoot, "py") : committed.py,
};
for (const dir of Object.values(out)) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
}

for (const file of schemaFiles) {
  const name = moduleName(file);
  const schema = JSON.parse(readFileSync(join(schemasDir, file), "utf8"));
  const banner = `/* AUTO-GENERATED from packages/schemas/${file}. Do not edit; run \`pnpm gen:types\`. */`;
  const ts = await compile(schema, schema.title, { bannerComment: banner, cwd: schemasDir });
  writeFileSync(join(out.ts, `${name}.ts`), lf(ts));

  const py = spawnSync(
    "uv",
    [
      "run",
      "datamodel-codegen",
      "--input",
      join(schemasDir, file),
      "--input-file-type",
      "jsonschema",
      "--output",
      join(out.py, `${name}.py`),
      "--output-model-type",
      "pydantic_v2.BaseModel",
      "--target-python-version",
      "3.11",
      "--disable-timestamp",
      "--field-constraints",
      "--extra-fields",
      "allow",
      "--custom-file-header",
      `# AUTO-GENERATED from packages/schemas/${file}. Do not edit; run \`pnpm gen:types\`.`,
    ],
    { cwd: root, encoding: "utf8" },
  );
  if (py.status !== 0) {
    console.error(`gen-types: datamodel-codegen failed for ${file}\n${py.stderr}${py.stdout}`);
    process.exit(1);
  }
}
writeFileSync(
  join(out.py, "__init__.py"),
  '"""Pydantic models generated from packages/schemas. Do not edit; run `pnpm gen:types`."""\n',
);

if (!check) {
  console.log(`gen-types: wrote ${schemaFiles.length} schema(s) to TS and Python`);
  process.exit(0);
}

const problems = [];
for (const kind of ["ts", "py"]) {
  const list = (dir) =>
    readdirSync(dir)
      .filter((f) => /.(ts|py)$/.test(f))
      .sort();
  const fresh = list(out[kind]);
  const have = existsSync(committed[kind]) ? list(committed[kind]) : [];
  for (const f of fresh.filter((f) => !have.includes(f))) problems.push(`missing: ${kind}/${f}`);
  for (const f of have.filter((f) => !fresh.includes(f))) problems.push(`stale: ${kind}/${f}`);
  for (const f of fresh.filter((f) => have.includes(f))) {
    const a = lf(readFileSync(join(out[kind], f), "utf8"));
    const b = lf(readFileSync(join(committed[kind], f), "utf8"));
    if (a !== b) problems.push(`differs: ${kind}/${f}`);
  }
}
rmSync(tmpRoot, { recursive: true, force: true });
if (problems.length > 0) {
  console.error(`gen-types: generated types are stale. Run \`pnpm gen:types\` and commit.`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("gen-types: generated types are current");
