# Frozen RoboMemArena compatibility inputs

This release integrates selected RoboMemArena task definitions and ordered-stage logic into the LIBERO-Agent framework.

Source: `OpenHelix-Team/RoboMemArena` at
`cc156e519990ae43cf3b64281a548724f428fbbd`.

Contents:

- `core/`: only files added or changed by RoboMemArena relative to this
  LIBERO checkout, plus a compatibility namespace shim;
- `bddl/`: official Task 1--26 BDDL files;
- `stage/`: official ordered-stage and shared physical pour-counter logic,
  with the broad evaluation-runner import replaced by a three-helper local
  adapter.

These files are evaluator-private. The integration's `bootstrap.py` overlays
them on a temporary view of the installed LIBERO package. It uses lightweight
links when supported and falls back to real-file copies, so the release
artifact itself contains no symbolic links and does not require link support.
The temporary compatibility view must not be copied to an Agent workspace.
