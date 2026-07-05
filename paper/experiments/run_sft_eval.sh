#!/usr/bin/env bash
# Experiment 1: SFT evaluation with LLaMA-3-8B
# Requires: 4×A100-40GB, sysplug[hf] installed

set -euo pipefail

MODEL="meta-llama/Meta-Llama-3-8B"
DATASET="tatsu-lab/alpaca"
OUTPUT_DIR="./results/sft"
mkdir -p "$OUTPUT_DIR"

echo "=== Baseline run (no SysPlug) ==="
python -c "
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
model = AutoModelForCausalLM.from_pretrained('${MODEL}')
args = TrainingArguments(
    output_dir='${OUTPUT_DIR}/baseline',
    per_device_train_batch_size=1,
    num_train_epochs=1,
    report_to=[],
)
# TODO: full run
print('Baseline args:', args.per_device_train_batch_size)
"

echo ""
echo "=== SysPlug run ==="
python examples/sft_huggingface.py

echo ""
echo "Results saved to $OUTPUT_DIR"
