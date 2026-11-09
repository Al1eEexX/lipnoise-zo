#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
python3 -m pytest -q
python3 experiments/e1_bias.py
python3 experiments/e2_phase.py
python3 experiments/e3_rounding.py
