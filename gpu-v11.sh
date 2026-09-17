#!/usr/bin/env bash
# Run from an existing v9 GPU workspace. Does not modify v9 artifacts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-.venv/bin/python}"
STAGE="${1:-benchmark}"
OUT="${OUT:-outputs/v11}"
LIMIT="${LIMIT:-2000}"
BUNDLE_BATCH="${BUNDLE_BATCH:-2}"
COMMON=(--output "$OUT" --bundle-batch "$BUNDLE_BATCH")
case "$STAGE" in
  benchmark) "$PY" -u -m legalir2.v11 benchmark "${COMMON[@]}" --limit 10 ;;
  collect)
    "$PY" -u -m legalir2.v11 collect "${COMMON[@]}" --limit "$LIMIT"
    "$PY" -u -m legalir2.v11 collect "${COMMON[@]}" --split validation --questions artifacts/splits/validation.json
    ;;
  fit) "$PY" -u -m legalir2.v11 fit --output "$OUT" ;;
  evaluate) "$PY" -u -m legalir2.v11 evaluate --output "$OUT" ;;
  submit) "$PY" -u -m legalir2.v11 submit "${COMMON[@]}" ;;
  all)
    bash "$0" benchmark
    bash "$0" collect
    bash "$0" fit
    bash "$0" evaluate
    bash "$0" submit
    ;;
  *) echo 'Usage: bash gpu-v11.sh {benchmark|collect|fit|evaluate|submit|all}' >&2; exit 2 ;;
esac
