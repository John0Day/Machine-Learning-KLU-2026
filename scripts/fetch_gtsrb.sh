#!/usr/bin/env bash
set -euo pipefail

# Download official GTSRB training archives into data/raw using fixed ERDA URLs.
# Safe for repeated runs: existing verified files are reused and extraction is cached.

TARGET_DIR=""
CHECKSUMS_FILE=""
EXTRACT=0
FORCE=0
DRY_RUN=0
VERIFY_SHA256=1
VERIFY_ONLY=0
WRITE_CHECKSUMS=0
LOCK_DIR=""
EXTRACT_STATE_DIR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/data/raw"
CHECKSUMS_FILE="${REPO_ROOT}/checksums/gtsrb.sha256"

FILES=(
  "GTSRB_Final_Training_Images.zip"
  "GTSRB_Final_Training_HueHist.zip"
  "GTSRB_Final_Training_HOG.zip"
  "GTSRB_Final_Training_Haar.zip"
)

BASE_URL_DEFAULT="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
BASE_URL="$BASE_URL_DEFAULT"

print_help() {
  cat <<'USAGE'
Usage: fetch_gtsrb.sh [options]

Options:
  --base-url <url>            Optional override for archive base URL
  --target-dir <path>         Download directory (default: <repo>/data/raw)
  --checksums-file <path>     SHA-256 file (default: checksums/gtsrb.sha256)
  --skip-verify               Disable SHA-256 verification
  --verify-only               Only verify local archives; no download/extract
  --write-checksums           Write checksums for local archives to checksums file
  --extract                   Extract each archive after download
  --force                     Re-download archives and re-extract
  --dry-run                   Print actions without executing
  -h, --help                  Show this help

Examples:
  ./scripts/fetch_gtsrb.sh --extract
  ./scripts/fetch_gtsrb.sh --verify-only
  ./scripts/fetch_gtsrb.sh --skip-verify --write-checksums
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --checksums-file)
      CHECKSUMS_FILE="$2"
      shift 2
      ;;
    --skip-verify)
      VERIFY_SHA256=0
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --write-checksums)
      WRITE_CHECKSUMS=1
      shift
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ "$VERIFY_ONLY" -eq 1 && "$WRITE_CHECKSUMS" -eq 1 ]]; then
  echo "Error: --verify-only cannot be combined with --write-checksums" >&2
  exit 1
fi

if [[ "$VERIFY_ONLY" -eq 1 && "$EXTRACT" -eq 1 ]]; then
  echo "Error: --verify-only cannot be combined with --extract" >&2
  exit 1
fi

sha256_of() {
  local file_path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print $1}'
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    LC_ALL=C LANG=C shasum -a 256 "$file_path" | awk '{print $1}'
    return
  fi

  echo "Error: neither sha256sum nor shasum is available." >&2
  exit 1
}

checksums_has_entries() {
  local file_path="$1"
  [[ -f "$file_path" ]] && grep -Eq '^[0-9a-fA-F]{64}[[:space:]]+' "$file_path"
}

lookup_expected_hash() {
  local checksums_file="$1"
  local archive_name="$2"
  awk -v f="$archive_name" '$1 ~ /^[0-9a-fA-F]{64}$/ && $2 == f {print tolower($1)}' "$checksums_file"
}

verify_hash_match() {
  local archive_path="$1"
  local expected_hash="$2"

  local actual_hash
  actual_hash="$(sha256_of "$archive_path")"
  [[ "$actual_hash" == "$expected_hash" ]]
}

verify_archive_hash() {
  local archive_path="$1"
  local archive_name="$2"

  if [[ "$VERIFY_SHA256" -ne 1 ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] verify sha256 '$archive_name' using '$CHECKSUMS_FILE'"
    return 0
  fi

  local expected
  expected="$(lookup_expected_hash "$CHECKSUMS_FILE" "$archive_name")"
  if [[ -z "$expected" ]]; then
    echo "Error: Missing checksum entry for '$archive_name' in '$CHECKSUMS_FILE'." >&2
    exit 1
  fi

  if ! verify_hash_match "$archive_path" "$expected"; then
    local actual
    actual="$(sha256_of "$archive_path")"
    echo "Error: SHA-256 mismatch for '$archive_name'." >&2
    echo "Expected: $expected" >&2
    echo "Actual:   $actual" >&2
    exit 1
  fi

  echo "Verified SHA-256: $archive_name"
}

write_checksums_file() {
  local out_file="$1"
  local target_dir="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write checksums to '$out_file'"
    return 0
  fi

  mkdir -p "$(dirname "$out_file")"
  {
    echo "# SHA-256 for official GTSRB training archives"
    echo "# Format: <sha256> <filename>"
    for file in "${FILES[@]}"; do
      local archive_path="${target_dir%/}/${file}"
      if [[ ! -f "$archive_path" ]]; then
        echo "Error: cannot write checksums, file missing: $archive_path" >&2
        exit 1
      fi
      local hash
      hash="$(sha256_of "$archive_path")"
      echo "${hash} ${file}"
    done
  } >"$out_file"

  echo "Wrote checksums: $out_file"
}

download_file_atomic() {
  local url="$1"
  local out="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] curl -fL --retry 3 --retry-delay 2 -o '$out' '$url'"
    return 0
  fi

  local tmp_out
  tmp_out="${out}.part.$$"
  rm -f "$tmp_out"
  curl -fL --retry 3 --retry-delay 2 -o "$tmp_out" "$url"
  mv -f "$tmp_out" "$out"
}

maybe_extract_file() {
  local zip_path="$1"
  local extract_dir="$2"
  local archive_name="$3"
  local out_dir="${extract_dir%/}/${archive_name%.zip}"
  local marker_file="${EXTRACT_STATE_DIR%/}/${archive_name}.sha256"

  if [[ "$EXTRACT" -ne 1 ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] unzip -o '$zip_path' -d '$extract_dir'"
    return 0
  fi

  local archive_hash
  archive_hash="$(sha256_of "$zip_path")"

  if [[ "$FORCE" -ne 1 && -d "$out_dir" && -f "$marker_file" ]]; then
    local cached_hash
    cached_hash="$(cat "$marker_file")"
    if [[ "$cached_hash" == "$archive_hash" ]]; then
      echo "Skipping extraction (cache hit): $archive_name"
      return 0
    fi
  fi

  echo "Extracting: $archive_name"
  unzip -o "$zip_path" -d "$extract_dir" >/dev/null
  printf '%s\n' "$archive_hash" >"$marker_file"
}

acquire_lock() {
  LOCK_DIR="${TARGET_DIR%/}/.fetch.lock"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    return 0
  fi

  echo "Error: another fetch process seems to be running (lock: $LOCK_DIR)." >&2
  echo "If not, remove the lock directory and retry." >&2
  exit 1
}

release_lock() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  if [[ -n "$LOCK_DIR" && -d "$LOCK_DIR" ]]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

mkdir -p "$TARGET_DIR"
EXTRACT_STATE_DIR="${TARGET_DIR%/}/.extract-state"
mkdir -p "$EXTRACT_STATE_DIR"

acquire_lock
trap release_lock EXIT

if [[ "$VERIFY_SHA256" -eq 1 ]]; then
  if ! checksums_has_entries "$CHECKSUMS_FILE"; then
    if [[ "$VERIFY_ONLY" -eq 1 ]]; then
      echo "Error: --verify-only requires valid entries in '$CHECKSUMS_FILE'." >&2
      exit 1
    fi
    echo "Warning: No usable checksums found at '$CHECKSUMS_FILE'. SHA-256 verification disabled." >&2
    VERIFY_SHA256=0
  fi
fi

echo "Base URL: $BASE_URL"
echo "Target directory: $TARGET_DIR"
echo "Checksums: $CHECKSUMS_FILE"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  for file in "${FILES[@]}"; do
    local_path="${TARGET_DIR%/}/${file}"
    if [[ ! -f "$local_path" ]]; then
      echo "Error: missing local archive for verification: $local_path" >&2
      exit 1
    fi
    verify_archive_hash "$local_path" "$file"
  done
  echo "Verification complete."
  exit 0
fi

for file in "${FILES[@]}"; do
  url="${BASE_URL%/}/${file}"
  out="${TARGET_DIR%/}/${file}"
  should_download=1

  if [[ -f "$out" && "$FORCE" -ne 1 ]]; then
    if [[ "$VERIFY_SHA256" -eq 1 ]]; then
      expected="$(lookup_expected_hash "$CHECKSUMS_FILE" "$file")"
      if [[ -z "$expected" ]]; then
        echo "Error: Missing checksum entry for '$file' in '$CHECKSUMS_FILE'." >&2
        exit 1
      fi
      if verify_hash_match "$out" "$expected"; then
        should_download=0
        echo "Skipping existing verified file: $out"
      else
        echo "Existing file failed checksum, re-downloading: $out"
        should_download=1
      fi
    else
      should_download=0
      echo "Skipping existing file: $out"
    fi
  fi

  if [[ "$should_download" -eq 1 ]]; then
    echo "Downloading: $file"
    download_file_atomic "$url" "$out"
  fi

  verify_archive_hash "$out" "$file"
  maybe_extract_file "$out" "$TARGET_DIR" "$file"
done

if [[ "$WRITE_CHECKSUMS" -eq 1 ]]; then
  write_checksums_file "$CHECKSUMS_FILE" "$TARGET_DIR"
fi

echo "Done."
