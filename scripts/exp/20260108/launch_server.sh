#!/bin/bash
# 启动服务器 - 运行在 conda_env_b

set -e

ENV_NAME="conda_env_b"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

echo "激活环境 ${ENV_NAME} 并启动服务器..."

# 初始化 conda
eval "$(conda shell.bash hook)"
cd "${PROJECT_ROOT}"

# 激活环境并运行
conda activate ${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
python unit_tests/test_client_server/server.py

