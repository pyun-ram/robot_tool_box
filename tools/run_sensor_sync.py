#!/usr/bin/env python
"""启动sensor_sync_node的脚本"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.sensor_sync_node import main

if __name__ == '__main__':
    main()

