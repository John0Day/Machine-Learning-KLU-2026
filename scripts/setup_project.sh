#!/usr/bin/env bash
set -euo pipefail

# One-command project bootstrap:
# 1) create virtual environment
# 2) install Python dependencies
# 3) download + extract GTSRB archives
# 4) run dataset inspection script

PYTHON_BIN=""
PYTHON_EXPLICIT=0
SKIP_DATA=0
SKIP_INSPECTION=0
FORCE_DATA=0
FORCE_INSTALL=0
SKIP_VERIFY=0
SKIP_PIP_UPGRADE=0
TRASH_ZIPS=1
BASE_URL=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

print_help() {
  cat <<'USAGE'
Usage: setup_project.sh [options]

Options:
  --python <bin>        Python executable to use (default auto: 3.12 -> 3.11 -> 3.10 -> 3)
  --skip-data           Skip dataset download/extraction
  --skip-inspection     Skip running src/dataset.py
  --force-data          Force re-download of dataset archives
  --force-install       Force dependency installation even if unchanged
  --no-pip-upgrade      Skip pip self-upgrade step
  --skip-verify         Disable checksum verification during dataset pull
  --keep-zips           Keep downloaded ZIP archives (default moves ZIPs to trash after extraction)
  --base-url <url>      Override dataset base URL for fetch_gtsrb.sh
  -h, --help            Show this help

Default behavior:
  Creates .venv, installs requirements, fetches/extracts GTSRB, moves ZIPs to trash, and runs dataset inspection.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      PYTHON_EXPLICIT=1
      shift 2
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-inspection)
      SKIP_INSPECTION=1
      shift
      ;;
    --force-data)
      FORCE_DATA=1
      shift
      ;;
    --force-install)
      FORCE_INSTALL=1
      shift
      ;;
    --no-pip-upgrade)
      SKIP_PIP_UPGRADE=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --keep-zips)
      TRASH_ZIPS=0
      shift
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
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

pick_default_python() {
  local candidates=("python3.12" "python3.11" "python3.10" "python3")
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

sha256_of_file() {
  local file_path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print $1}'
    return
  fi
  LC_ALL=C LANG=C shasum -a 256 "$file_path" | awk '{print $1}'
}

if [[ "$PYTHON_EXPLICIT" -eq 0 ]]; then
  if ! PYTHON_BIN="$(pick_default_python)"; then
    echo "Error: no Python interpreter found (tried python3.12, python3.11, python3.10, python3)." >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

echo "[1/4] Setting up virtual environment at $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ ! -x "$VENV_PYTHON" || ! -x "$VENV_PIP" ]]; then
  echo "Error: virtual environment appears broken at $VENV_DIR" >&2
  exit 1
fi

echo "[2/4] Installing dependencies"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"
CURRENT_REQ_HASH="$(sha256_of_file "$REPO_ROOT/requirements.txt")"
CACHED_REQ_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  CACHED_REQ_HASH="$(cat "$REQ_HASH_FILE")"
fi

if [[ "$FORCE_INSTALL" -eq 1 || "$CURRENT_REQ_HASH" != "$CACHED_REQ_HASH" ]]; then
  if [[ "$SKIP_PIP_UPGRADE" -eq 0 ]]; then
    "$VENV_PYTHON" -m pip install --upgrade pip
  fi
  "$VENV_PIP" install -r "$REPO_ROOT/requirements.txt"
  printf '%s\n' "$CURRENT_REQ_HASH" >"$REQ_HASH_FILE"
else
  echo "Dependencies unchanged, skipping pip install (use --force-install to override)."
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  echo "[3/4] Fetching dataset archives"
  fetch_cmd=("$REPO_ROOT/scripts/fetch_gtsrb.sh" "--extract")

  if [[ "$FORCE_DATA" -eq 1 ]]; then
    fetch_cmd+=("--force")
  fi
  if [[ "$SKIP_VERIFY" -eq 1 ]]; then
    fetch_cmd+=("--skip-verify")
  fi
  if [[ "$TRASH_ZIPS" -eq 1 ]]; then
    fetch_cmd+=("--trash-zips")
  fi
  if [[ -n "$BASE_URL" ]]; then
    fetch_cmd+=("--base-url" "$BASE_URL")
  fi

  "${fetch_cmd[@]}"
else
  echo "[3/4] Skipped dataset fetch (--skip-data)"
fi

if [[ "$SKIP_INSPECTION" -eq 0 ]]; then
  echo "[4/4] Running dataset inspection"
  "$VENV_PYTHON" "$REPO_ROOT/src/dataset.py"
else
  echo "[4/4] Skipped dataset inspection (--skip-inspection)"
fi

echo "Bootstrap complete."
