#!/usr/bin/env bash
set -euo pipefail

cd /data/robot_project/inference_server

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export HF_HOME="${HF_HOME:-/data/robot_project/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/robot_project/cache/hf/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-/data/robot_project/tmp}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PI_POLICY_TYPE="${PI_POLICY_TYPE:-pi0}"
export PI_MODEL_PATH="${PI_MODEL_PATH:-/data/robot_project/models/pi0_yellow_cube_green_scoop_full_20260708_091450/pretrained_model}"
export PI_DEVICE="${PI_DEVICE:-cuda}"
export PI_DEFAULT_TASK="${PI_DEFAULT_TASK:-Pick up the yellow cube and place it into the green scoop.}"
export PI_N_ACTION_STEPS="${PI_N_ACTION_STEPS:-10}"
export PI_COMPILE_MODEL="${PI_COMPILE_MODEL:-0}"
export PI_GRADIENT_CHECKPOINTING="${PI_GRADIENT_CHECKPOINTING:-0}"
export PI_LOAD_ON_STARTUP="${PI_LOAD_ON_STARTUP:-1}"

mkdir -p /data/robot_project/cache/hf/hub /data/robot_project/tmp /data/robot_project/logs

exec /home/rootx90/miniforge3/envs/lerobot/bin/python -m uvicorn server_policy:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1
