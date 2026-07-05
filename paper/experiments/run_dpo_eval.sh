#!/usr/bin/env bash
# Experiment 2: DPO evaluation
set -euo pipefail

OUTPUT_DIR="./results/dpo"
mkdir -p "$OUTPUT_DIR"

echo "=== DPO custom loop with SysPlug ==="
python examples/dpo_custom_loop.py | tee "$OUTPUT_DIR/sysplug_run.log"

echo "Done. Results in $OUTPUT_DIR"
