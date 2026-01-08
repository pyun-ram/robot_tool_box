#!/bin/bash
# 检测并终止占用 8765 端口的进程

set -e

PORT=8765

echo "检测端口 ${PORT} 的占用情况..."

# 方法1: 使用 lsof (如果可用)
if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti :${PORT} 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "找到占用端口 ${PORT} 的进程: $PIDS"
        for PID in $PIDS; do
            echo "终止进程 PID: $PID"
            kill -9 $PID 2>/dev/null || true
        done
        sleep 1
        echo "✅ 端口 ${PORT} 已释放"
        exit 0
    fi
fi

# 方法2: 使用 fuser (如果可用)
if command -v fuser >/dev/null 2>&1; then
    if fuser ${PORT}/tcp >/dev/null 2>&1; then
        echo "找到占用端口 ${PORT} 的进程"
        fuser -k ${PORT}/tcp 2>/dev/null || true
        sleep 1
        echo "✅ 端口 ${PORT} 已释放"
        exit 0
    fi
fi

# 方法3: 使用 ss (如果可用)
if command -v ss >/dev/null 2>&1; then
    PIDS=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u || true)
    if [ -n "$PIDS" ]; then
        echo "找到占用端口 ${PORT} 的进程: $PIDS"
        for PID in $PIDS; do
            echo "终止进程 PID: $PID"
            kill -9 $PID 2>/dev/null || true
        done
        sleep 1
        echo "✅ 端口 ${PORT} 已释放"
        exit 0
    fi
fi

# 方法4: 使用 netstat (如果可用)
if command -v netstat >/dev/null 2>&1; then
    PIDS=$(netstat -tlnp 2>/dev/null | grep ":${PORT}" | awk '{print $7}' | cut -d'/' -f1 | grep -E '^[0-9]+$' | sort -u || true)
    if [ -n "$PIDS" ]; then
        echo "找到占用端口 ${PORT} 的进程: $PIDS"
        for PID in $PIDS; do
            echo "终止进程 PID: $PID"
            kill -9 $PID 2>/dev/null || true
        done
        sleep 1
        echo "✅ 端口 ${PORT} 已释放"
        exit 0
    fi
fi

# 方法5: 通过 /proc/net/tcp 查找 (Linux 原生方法，无需额外工具)
# 将端口号转换为十六进制 (8765 = 0x2239)
PORT_HEX=$(printf "%04X" ${PORT})
# 在 /proc/net/tcp 中查找监听该端口的进程
if [ -f /proc/net/tcp ]; then
    # 查找本地地址包含该端口号的条目 (格式: 00000000:2239 表示 0.0.0.0:8765)
    TCP_LINE=$(grep -E "00000000:${PORT_HEX}|[0-9A-F]{8}:${PORT_HEX}" /proc/net/tcp 2>/dev/null | head -1 || true)
    if [ -n "$TCP_LINE" ]; then
        # 从 /proc/net/tcp 获取 inode
        INODE=$(echo "$TCP_LINE" | awk '{print $10}')
        # 通过 inode 查找进程
        for PID_DIR in /proc/[0-9]*/fd/*; do
            if [ -L "$PID_DIR" ] && [ "$(readlink "$PID_DIR" 2>/dev/null)" = "socket:[$INODE]" ]; then
                PID=$(echo "$PID_DIR" | cut -d'/' -f3)
                if [ -n "$PID" ] && [ "$PID" != "$$" ]; then
                    echo "找到占用端口 ${PORT} 的进程 PID: $PID (通过 /proc/net/tcp)"
                    kill -9 $PID 2>/dev/null || true
                    sleep 1
                    echo "✅ 端口 ${PORT} 已释放"
                    exit 0
                fi
            fi
        done
    fi
fi

# 方法6: 通过进程名查找 (最后的手段)
PIDS=$(ps aux | grep -E 'unit_tests\.(server|client)' | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "找到可能的 unit_tests 进程: $PIDS"
    for PID in $PIDS; do
        echo "终止进程 PID: $PID"
        kill -9 $PID 2>/dev/null || true
    done
    sleep 1
    echo "✅ 已清理 unit_tests 相关进程"
    exit 0
fi

echo "ℹ️  未找到占用端口 ${PORT} 的进程"

