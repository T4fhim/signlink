import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { describe, expect, it } from "vitest";

const schemasDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const readJson = (path: string): Record<string, unknown> =>
  JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;

const schemaFiles = readdirSync(schemasDir)
  .filter((f) => /^[a-z_]+\.v\d+\.json$/.test(f))
  .sort();

const makeAjv = () => {
  const ajv = new Ajv2020({ strict: true });
  addFormats(ajv);
  return ajv;
};

describe("schema contracts", () => {
  it("finds the five schemas", () => {
    expect(schemaFiles).toEqual([
      "landmark_layout.v1.json",
      "lexicon_entry.v1.json",
      "recognition_output.v1.json",
      "sign_output_request.v1.json",
      "ws_message.v1.json",
    ]);
  });

  describe.each(schemaFiles)("%s", (file) => {
    const schema = readJson(join(schemasDir, file));
    const example = readJson(join(schemasDir, "examples", file.replace(".json", ".example.json")));

    it("accepts its example", () => {
      const validate = makeAjv().compile(schema);
      expect(validate(example), JSON.stringify(validate.errors)).toBe(true);
    });

    it("rejects an example missing a required field", () => {
      const validate = makeAjv().compile(schema);
      for (const key of schema["required"] as string[]) {
        const without = Object.fromEntries(Object.entries(example).filter(([k]) => k !== key));
        expect(validate(without), `without ${key}`).toBe(false);
      }
    });

    it("tolerates added fields (add-only evolution)", () => {
      const validate = makeAjv().compile(schema);
      expect(validate({ ...example, future_field: 1 })).toBe(true);
    });
  });
});
