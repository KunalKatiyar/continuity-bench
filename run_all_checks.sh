#!/usr/bin/env bash
# Every check in one command: self-checks, a deterministic rebuild, corpus
# validation, the free-heuristic leakage attack, and the leaderboard.
#
# Prefers the project venv when present. On this machine a bare `python3` resolves
# to a sibling project's .venv, which has no pip and no SDKs.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
echo "using $("$PY" --version) at $PY"

for suite in test_*.py; do
    "$PY" "$suite"
done
"$PY" bench.py build >/dev/null
"$PY" bench.py validate
"$PY" evaluate.py attack
"$PY" build_site.py
