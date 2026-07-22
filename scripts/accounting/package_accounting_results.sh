#!/usr/bin/env bash
set -euo pipefail

TYPE="${1:-}"
SOURCE="${2:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="../outputs/manual_upload"
mkdir -p "$OUT"

case "$TYPE" in
  ordinary)
    if [ -z "$SOURCE" ] && [ -f ../outputs/latest_routecat_ordinary_accounting_dir.txt ]; then
      SOURCE="$(cat ../outputs/latest_routecat_ordinary_accounting_dir.txt)"
    fi
    [ -d "$SOURCE" ] || { echo "[fatal] ordinary accounting source not found: $SOURCE" >&2; exit 1; }
    FULL="$OUT/routecat_ordinary_accounting_full_${STAMP}.tar.gz"
    tar -C "$(dirname "$SOURCE")" -czf "$FULL" "$(basename "$SOURCE")"
    sha256sum "$FULL" > "$FULL.sha256"
    python scripts/utilities/make_pruning_only_accounting.py "$SOURCE" \
      --output-dir "$OUT/accounting_result_pruning_only_${STAMP}" \
      --archive "$OUT/routecat_ordinary_accounting_pruning_only_${STAMP}.tar.gz"
    sha256sum "$OUT/routecat_ordinary_accounting_pruning_only_${STAMP}.tar.gz" \
      > "$OUT/routecat_ordinary_accounting_pruning_only_${STAMP}.tar.gz.sha256"
    echo "[manual upload required] download both archives from $OUT; the slim archive is sufficient for Experiments 1–3."
    ;;
  advanced)
    if [ -z "$SOURCE" ] && [ -f ../outputs/latest_routecat_advanced_accounting_dir.txt ]; then
      SOURCE="$(cat ../outputs/latest_routecat_advanced_accounting_dir.txt)"
    fi
    [ -d "$SOURCE" ] || { echo "[fatal] advanced accounting source not found: $SOURCE" >&2; exit 1; }
    ARCHIVE="$OUT/routecat_advanced_accounting_${STAMP}.tar.gz"
    tar -C "$(dirname "$SOURCE")" -czf "$ARCHIVE" "$(basename "$SOURCE")"
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    echo "[manual upload required] download $ARCHIVE and its checksum before the node is released."
    ;;
  *)
    echo "usage: $0 ordinary|advanced [source_directory]" >&2
    exit 2
    ;;
esac
