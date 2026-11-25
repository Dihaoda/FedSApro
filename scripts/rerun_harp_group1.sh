#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src

config="configs/reproduction/natural_domain_attack_study.yaml"
model="qwen2.5-0.5b"
seed="2026"

for method in indexguard flame fltrust fedgraph; do
  for attack in none label_flip textual_backdoor; do
    python3 -m fedsapro.cli run \
      --config "$config" \
      --model "$model" \
      --attack "$attack" \
      --method "$method" \
      --seed "$seed" \
      --force
  done
done

for attack in label_flip textual_backdoor; do
  python3 -m fedsapro.cli run \
    --config "$config" \
    --model "$model" \
    --attack "$attack" \
    --method harp_fedllm \
    --seed "$seed" \
    --force
done

