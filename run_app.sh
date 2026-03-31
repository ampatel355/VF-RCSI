#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  exec "$PROJECT_ROOT/.venv/bin/python" -m streamlit run app.py "$@"
fi

if command -v streamlit >/dev/null 2>&1; then
  exec streamlit run app.py "$@"
fi

printf '%s\n' "Could not find Streamlit."
printf '%s\n' "Install dependencies first with: pip install -r requirements.txt"
exit 1
