#!/usr/bin/env python
"""启动policy_model_node的脚本"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.policy_model_node import main

if __name__ == '__main__':
    main()

