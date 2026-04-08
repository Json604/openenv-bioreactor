#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: bash scripts/validate-submission.sh <hf_space_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

echo "[1/3] Checking Space reset endpoint"
HTTP_CODE="$(curl -s -o /tmp/openenv-reset.out -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || true)"
if [ "$HTTP_CODE" != "200" ]; then
  echo "FAILED: $PING_URL/reset returned HTTP $HTTP_CODE"
  cat /tmp/openenv-reset.out || true
  exit 1
fi
echo "PASSED: /reset returned 200"

echo "[2/3] Checking Docker build"
docker build "$REPO_DIR" >/tmp/openenv-docker-build.log
echo "PASSED: Docker build succeeded"

echo "[3/3] Checking openenv validate"
if ! command -v openenv >/dev/null 2>&1; then
  echo "FAILED: openenv command not found. Install with: python3 -m pip install openenv-core"
  exit 1
fi
(cd "$REPO_DIR" && openenv validate --verbose)
echo "PASSED: openenv validate succeeded"

echo "All 3/3 checks passed"
