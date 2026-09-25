#!/usr/bin/env bash
# Render the shareable leaderboard into the sibling repo and copy the raw results.
# The public repo holds only generated output and data, so it can be served by
# GitHub Pages with no build step and no Python.
set -euo pipefail
cd "$(dirname "$0")"

LB="${1:-../continuity-bench-leaderboard}"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

mkdir -p "$LB/data/results"
rm -f "$LB"/data/results/*.json
cp results/*.json "$LB/data/results/"
"$PY" build_site.py --public --out "$LB/index.html"
touch "$LB/.nojekyll"
echo "published to $LB"
