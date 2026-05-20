#!/bin/bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="172.16.20.230"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

/usr/bin/ssh "${REMOTE_USER}@${REMOTE_HOST}" BASE_MODEL_PATH="${BASE_MODEL_PATH}" /bin/bash <<'EOF'
set -euo pipefail

REPO_DIR="/ptpool/users/qzy/code/seethrough3d"
PYTHON_BIN="/ptpool/users/czz/miniconda3/envs/st3d/bin/python"
SCRIPT_PATH="/ptpool/users/qzy/code/seethrough3d/infer20.py"
SCENE_PKL="/ptpool/users/qzy/code/seethrough3d/inference/saved_scenes/test4.6/example_test4.6_1_fixed.pkl"
PLACEHOLDER_PROMPT="a photo of PLACEHOLDER, a realistic urban traffic scene, soft morning light, slightly cool air, a clean natural composition, clear depth, and a documentary photography style."

cd "${REPO_DIR}"

if [ -n "${BASE_MODEL_PATH:-}" ]; then
  export SEETHROUGH3D_BASE_MODEL_PATH="${BASE_MODEL_PATH}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

exec "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --scene-pkls "${SCENE_PKL}" \
  --placeholder-prompt "${PLACEHOLDER_PROMPT}"
EOF
