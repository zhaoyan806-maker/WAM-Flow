#!/bin/bash
set -euo pipefail

CONFIG=${CONFIG:-config/grpo_navsim.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-output/train/grpo_navsim}
BETA=${BETA:-0.03}
CLIP_EPSILON=${CLIP_EPSILON:-0.2}

python train_grpo.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  grpo.kl.beta="${BETA}" \
  grpo.clip_epsilon="${CLIP_EPSILON}"
