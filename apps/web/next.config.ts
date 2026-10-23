import type { NextConfig } from "next";

// SIGNLNK_ISOLATE=1 serves /dev/* cross-origin isolated, which lets onnxruntime-web use threads.
// Opt-in until a measurement (docs/reports/phase1-step6-decoder.md) shows it is worth keeping.
const isolate = process.env["SIGNLNK_ISOLATE"] === "1";

const config: NextConfig = {
  reactStrictMode: true,
  headers: () =>
    Promise.resolve(
      isolate
        ? [
            {
              // Worker scripts (/_next, and ORT's thread workers in /ort) must carry the page's policy.
              source: "/:path(dev/.*|_next/.*|ort/.*)",
              headers: [
                { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
                { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
              ],
            },
          ]
        : [],
    ),
  transpilePackages: ["@signlnk/landmarks", "@signlnk/recognizer", "@signlnk/schemas"],
};

export default config;
