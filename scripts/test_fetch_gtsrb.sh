#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip is required for tests." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BASE_DIR="$TMP_DIR/base"
WORK_DIR="$TMP_DIR/work"
RAW_DIR="$TMP_DIR/raw"
CHECKSUMS_FILE="$TMP_DIR/gtsrb.sha256"

mkdir -p "$BASE_DIR" "$WORK_DIR" "$RAW_DIR"

FILES=(
  "GTSRB_Final_Training_Images.zip"
  "GTSRB_Final_Training_HueHist.zip"
  "GTSRB_Final_Training_HOG.zip"
  "GTSRB_Final_Training_Haar.zip"
)

for file in "${FILES[@]}"; do
  name="${file%.zip}"
  mkdir -p "$WORK_DIR/$name"
  printf 'fixture:%s\n' "$file" >"$WORK_DIR/$name/README.txt"
  (cd "$WORK_DIR" && zip -qr "$BASE_DIR/$file" "$name")
done

BASE_URL="file://$BASE_DIR"

"$REPO_ROOT/scripts/fetch_gtsrb.sh" \
  --base-url "$BASE_URL" \
  --target-dir "$RAW_DIR" \
  --checksums-file "$CHECKSUMS_FILE" \
  --skip-verify \
  --write-checksums >/dev/null

first_run_output="$TMP_DIR/first_run.txt"
second_run_output="$TMP_DIR/second_run.txt"
third_run_output="$TMP_DIR/third_run.txt"

"$REPO_ROOT/scripts/fetch_gtsrb.sh" \
  --base-url "$BASE_URL" \
  --target-dir "$RAW_DIR" \
  --checksums-file "$CHECKSUMS_FILE" \
  --extract >"$first_run_output"

"$REPO_ROOT/scripts/fetch_gtsrb.sh" \
  --base-url "$BASE_URL" \
  --target-dir "$RAW_DIR" \
  --checksums-file "$CHECKSUMS_FILE" \
  --extract >"$second_run_output"

grep -q "Skipping existing verified file" "$second_run_output"
grep -q "Skipping extraction (cache hit)" "$second_run_output"

# Simulate "ZIP moved to trash" and ensure rerun does not re-download when extracted cache exists.
rm -f "$RAW_DIR/GTSRB_Final_Training_HOG.zip"
"$REPO_ROOT/scripts/fetch_gtsrb.sh" \
  --base-url "$BASE_URL" \
  --target-dir "$RAW_DIR" \
  --checksums-file "$CHECKSUMS_FILE" \
  --extract >"$third_run_output"
grep -q "Skipping download (already extracted with cache marker): GTSRB_Final_Training_HOG.zip" "$third_run_output"
if [[ -f "$RAW_DIR/GTSRB_Final_Training_HOG.zip" ]]; then
  echo "Error: HOG archive should not be re-downloaded when extraction cache exists." >&2
  exit 1
fi

# Tamper with a remaining archive and verify integrity check fails.
printf 'tamper\n' >>"$RAW_DIR/GTSRB_Final_Training_Images.zip"
if "$REPO_ROOT/scripts/fetch_gtsrb.sh" \
  --target-dir "$RAW_DIR" \
  --checksums-file "$CHECKSUMS_FILE" \
  --verify-only >/dev/null 2>&1; then
  echo "Error: verify-only should fail after tampering." >&2
  exit 1
fi

echo "test_fetch_gtsrb.sh: PASS"
