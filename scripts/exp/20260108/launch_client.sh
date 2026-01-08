#!/bin/bash
# 启动客户端 - 运行在 conda_env_a

set -e

ENV_NAME="3d_diffuser_actor"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

echo "激活环境 ${ENV_NAME} 并启动客户端..."

# 初始化 conda
eval "$(conda shell.bash hook)"
cd "${PROJECT_ROOT}"

# 激活环境并运行
conda activate ${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
python unit_tests/test_client_server/client.py

