#!/usr/bin/env bash
set -euo pipefail

: "${HF_REPO_ID:?Set HF_REPO_ID, for example cyrusmoazami/tabular-state-transformer}"

hf auth whoami >/dev/null
hf upload "$HF_REPO_ID" . --exclude ".git/*" --exclude "__pycache__/*" --exclude "*.pyc"
