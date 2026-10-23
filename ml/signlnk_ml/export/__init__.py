"""Export of trained models (Phase 1 step 5). Needs torch and the `export` extra."""

import os

# Python onnxruntime sends Microsoft telemetry on Windows unless told not to (its Privacy.md).
# The project is "no telemetry" (CLAUDE.md rule 2); this runs before onnxruntime is imported.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
