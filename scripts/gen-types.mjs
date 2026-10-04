// Regenerates TS types (json-schema-to-typescript) and Pydantic models (datamodel-code-generator)
// from packages/schemas/*.json. Real generation lands in Phase 0 step 2.
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const schemasDir = join(dirname(fileURLToPath(import.meta.url)), "..", "packages", "schemas");
const schemas = readdirSync(schemasDir).filter((f) => f.endsWith(".json") && f !== "package.json");

if (schemas.length === 0) {
  console.log("gen-types: no schemas yet, nothing to generate");
  process.exit(0);
}

console.error(`gen-types: found ${schemas.length} schema(s) but generation is not implemented yet`);
process.exit(1);
