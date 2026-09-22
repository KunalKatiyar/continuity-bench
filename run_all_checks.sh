#!/usr/bin/env bash
# Every check in one command: self-checks, a deterministic rebuild, corpus
# validation, the free-heuristic leakage attack, and the leaderboard.
set -euo pipefail
cd "$(dirname "$0")"

for suite in test_*.py; do
    python3 "$suite"
done
python3 bench.py build >/dev/null
python3 bench.py validate
python3 evaluate.py attack
python3 build_site.py
