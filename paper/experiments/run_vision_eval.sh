#!/usr/bin/env bash
# Experiment 3: Vision classification (ResNet-50 / ViT)
set -euo pipefail

OUTPUT_DIR="./results/vision"
mkdir -p "$OUTPUT_DIR"

python -c "
import sysplug, torch, torch.nn as nn

# ResNet-50 equivalent
class ResBlock(nn.Module):
    def __init__(self, c): super().__init__(); self.conv = nn.Conv2d(c, c, 3, padding=1)
    def forward(self, x): return x + self.conv(x)

model = nn.Sequential(*[ResBlock(64) for _ in range(8)], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1000))
param_count = sum(p.numel() for p in model.parameters())
print(f'ResNet-like model: {param_count/1e6:.1f}M params')

advisor = sysplug.Advisor(model=model, training_type='supervised', verbose=True)
cfg = advisor.suggest_config({'batch_size': 32, 'learning_rate': 1e-3, 'precision': 'fp32'})
print(cfg.summary(verbose=False))
" | tee "$OUTPUT_DIR/vision_run.log"

echo "Done."
