/**
 * Minimal NumPy `.npy` (format 1.0) support for float32 arrays: little-endian, C order.
 * Recordings are saved with this so `np.load` and ml/signlnk_ml/data/recordings.py can read them.
 */

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
const PREFIX_BYTES = 10; // magic (6) + version (2) + header length (2)
const ALIGNMENT = 64;

/** Encodes `data` (row-major, shape `shape`) as a `.npy` file. */
export function encodeNpy(data: Float32Array, shape: readonly number[]): Uint8Array {
  const count = shape.reduce((product, size) => product * size, 1);
  if (count !== data.length) {
    throw new RangeError(`shape [${shape.join(", ")}] needs ${count} values, got ${data.length}`);
  }
  const shapeText = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeText}, }`;
  // Pad with spaces so the data starts on a 64-byte boundary; the header ends with a newline.
  const padding = (ALIGNMENT - ((PREFIX_BYTES + dict.length + 1) % ALIGNMENT)) % ALIGNMENT;
  const header = `${dict}${" ".repeat(padding)}\n`;

  const out = new Uint8Array(PREFIX_BYTES + header.length + data.length * 4);
  out.set([...MAGIC, 1, 0]);
  const view = new DataView(out.buffer);
  view.setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[PREFIX_BYTES + i] = header.charCodeAt(i);
  let offset = PREFIX_BYTES + header.length;
  for (const value of data) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  return out;
}

/** Decodes a float32 `.npy` written by `encodeNpy` (or NumPy with the same dtype and order). */
export function decodeNpy(bytes: Uint8Array): { shape: number[]; data: Float32Array } {
  if (MAGIC.some((byte, i) => bytes[i] !== byte)) throw new Error("not an .npy file");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes[6] !== 1) throw new Error(`unsupported .npy version ${bytes[6]}.${bytes[7]}`);
  const headerLength = view.getUint16(8, true);
  const header = new TextDecoder().decode(
    bytes.subarray(PREFIX_BYTES, PREFIX_BYTES + headerLength),
  );
  if (!header.includes("'<f4'") || !header.includes("'fortran_order': False")) {
    throw new Error(`unsupported .npy header: ${header.trim()}`);
  }
  const shapeMatch = /'shape': \(([^)]*)\)/.exec(header);
  if (!shapeMatch) throw new Error("no shape in .npy header");
  const shape = (shapeMatch[1] ?? "")
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .map(Number);

  const start = PREFIX_BYTES + headerLength;
  const count = (bytes.length - start) / 4;
  const data = new Float32Array(count);
  for (let i = 0; i < count; i++) data[i] = view.getFloat32(start + i * 4, true);
  return { shape, data };
}
