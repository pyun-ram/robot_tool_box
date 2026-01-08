#!/bin/bash

# 设置基础路径
BASE_EPISODE_DIR="data/20260102_s1_data/"
BASE_SAVE_DIR="data/20260102_s1_data_converted/train/pick_moving_target_from_belt/all_variations/episodes"

# 检查基础目录是否存在
if [ ! -d "$BASE_EPISODE_DIR" ]; then
    echo "错误: 目录 $BASE_EPISODE_DIR 不存在"
    exit 1
fi

# 获取所有子目录
echo "正在查找 $BASE_EPISODE_DIR 下的所有子目录..."
subdirs=$(find "$BASE_EPISODE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ -z "$subdirs" ]; then
    echo "警告: 在 $BASE_EPISODE_DIR 下未找到任何子目录"
    exit 1
fi

export PYTHONPATH=/usr/app/Code/3d_diffuser_actor
# 遍历每个子目录并运行转换命令
for subdir in $subdirs; do
    # if subdir/calibrations.json 不存在，则跳过
    if [ ! -f "$subdir/calibrations.json" ]; then
        echo "跳过子目录: $subdir，因为 calibrations.json 不存在"
        continue
    fi
    
    # 获取子目录名称（相对于BASE_EPISODE_DIR）
    subdir_name=$(basename "$subdir")
    
    # 构建保存路径（使用子目录名称作为episode名称）
    save_dir="$BASE_SAVE_DIR/$subdir_name"
    # if save_dir 存在，则已经处理完，则跳过
    if [ -d "$save_dir" ]; then
        echo "跳过子目录: $subdir，因为已经处理过"
        continue
    fi
    
    echo "=========================================="
    echo "处理子目录: $subdir"
    echo "保存目录: $save_dir"
    echo "=========================================="
    
    # 运行转换命令
    python tools/run.py convert_data_to_standard_format \
        --episode_dir "$subdir" \
        --save_dir "$save_dir"
    
    # 检查命令执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功处理: $subdir"
    else
        echo "✗ 处理失败: $subdir"
        exit 1
    fi
    echo ""
done

echo "所有子目录处理完成！"