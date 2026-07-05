#!/usr/bin/env bash
# Experiment 4: Memory and throughput prediction accuracy
set -euo pipefail

OUTPUT_DIR="./results/prediction"
mkdir -p "$OUTPUT_DIR"

echo "=== Memory prediction benchmark (mock) ==="
python tests/benchmarks/bench_memory_prediction.py \
    --mock --samples 50 --output "$OUTPUT_DIR/memory_bench.csv"

echo ""
echo "=== Throughput prediction benchmark (mock) ==="
python tests/benchmarks/bench_throughput_prediction.py \
    --mock --samples 20

echo "Done. Results in $OUTPUT_DIR"
