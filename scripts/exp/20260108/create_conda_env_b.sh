#!/bin/bash
# 创建 conda_env_b (Python 3.10) - 用于运行服务器

set -e

ENV_NAME="conda_env_b"
PYTHON_VERSION="3.10"

echo "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# 激活环境并安装依赖
echo "安装依赖..."
conda run -n ${ENV_NAME} pip install "websockets>=11.0" msgpack numpy typing_extensions

echo "✅ 环境 ${ENV_NAME} 创建完成"

