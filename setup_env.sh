#!/usr/bin/env bash
# Thin wrapper — all logic lives in setup_env.py so this and setup_env.bat
# can't drift. Usage: ./setup_env.sh [cpu|cu121]   (default: cu121)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/setup_env.py" "$@"
