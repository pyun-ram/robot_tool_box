#!/bin/bash
# 创建 conda_env_a (Python 3.8) - 用于运行客户端

set -e

ENV_NAME="3d_diffuser_actor"

# 激活环境并安装依赖
echo "安装依赖..."
conda run -n ${ENV_NAME} pip install "websockets>=11.0" msgpack typing_extensions

echo "✅ 环境 ${ENV_NAME} 创建完成"

