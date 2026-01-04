# ROS客户端框架架构实现

本文档描述了已实现的ROS客户端框架架构（简化版）。

## 文件结构

```
robot_tool_box/
├── config/
│   └── config.yaml              # 统一配置文件（server + client）
├── client/                      # 客户端节点
│   ├── __init__.py
│   ├── sensor_sync_node.py     # 数据收集同步节点（ROS，20Hz发布）
│   ├── policy_model_node.py    # Policy模型节点（ROS，20Hz调用Server）
│   └── execution_node.py       # 执行节点（ROS，20Hz执行Action Buffer）
├── server/                      # 服务器端
│   ├── __init__.py
│   └── policy_server.py        # Policy Server（统一处理Policy和World Model）
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── action_buffer.py        # Action Buffer类（Time Ensemble）
│   ├── ros_utils.py            # ROS相关工具函数
│   ├── websocket_utils.py      # WebSocket客户端工具函数
│   ├── robot_sdk.py            # 机器人SDK封装
│   ├── ros_message_helper.py   # ROS消息辅助函数
│   └── ros_message_types.md    # ROS消息类型说明文档
├── tools/                       # 工具脚本
│   ├── run_sensor_sync.py      # 启动sensor_sync_node
│   ├── run_policy_model.py     # 启动policy_model_node
│   ├── run_execution.py        # 启动execution_node
│   └── run_server.py           # 启动policy_server
├── robot_tool_box/             # 包初始化
│   └── config_loader.py        # 配置加载器
├── INTEGRATION.md              # 集成说明文档
└── FRAMEWORK_README.md         # 本文件
```

## 核心组件

### 1. 配置文件系统

- **config/config.yaml**: 统一的YAML配置文件，包含server和client的所有配置
- **robot_tool_box/config_loader.py**: 配置加载器，提供`get_server_config()`和`get_client_config()`方法

### 2. 客户端节点

#### sensor_sync_node
- 订阅ROS原始话题（4个相机的RGB/Depth和gripper pose）
- 使用`message_filters.ApproximateTimeSynchronizer`同步消息
- 以20Hz频率发布同步后的数据到`/sensor_sync/data`话题

#### policy_model_node
- 订阅`/sensor_sync/data`话题
- 以20Hz频率处理消息
- 数据预处理（RGB转点云等，框架阶段为占位函数）
- 通过WebSocket调用server
- 根据server返回的类型处理（update_world_model或action）
- 如果收到action，更新Action Buffer

#### execution_node
- 从Action Buffer读取joint angles
- 以20Hz频率执行
- 通过机器人SDK发送joint commands到机器人

### 3. 服务器端

#### policy_server
- 统一WebSocket服务器
- 根据模型类型（default或3dafdp）处理请求
- 返回不同类型的响应（update_world_model或action）

### 4. 工具模块

#### Action Buffer (utils/action_buffer.py)
- 线程安全的action历史队列
- 支持时间平滑（Time Ensemble）
- 支持不同的平滑方法（加权平均等）

#### WebSocket客户端 (utils/websocket_utils.py)
- 封装WebSocket连接
- 连接管理和重连机制
- 数据序列化（msgpack_numpy）

#### ROS工具 (utils/ros_utils.py)
- ROS消息转换函数
- 图像格式转换
- 点云格式转换

#### 机器人SDK (utils/robot_sdk.py)
- 机器人控制接口封装
- 发送joint commands

## 使用说明

### 配置

编辑 `config/config.yaml` 文件，设置：
- 服务器地址和端口
- ROS话题名称
- 相机参数路径
- Action Buffer参数
- 机器人配置

### 启动系统

1. **启动服务器**：
```bash
python tools/run_server.py
```

2. **启动客户端节点**（在不同终端）：
```bash
# 终端1: 传感器同步节点
python tools/run_sensor_sync.py

# 终端2: Policy模型节点
python tools/run_policy_model.py

# 终端3: 执行节点
python tools/run_execution.py
```

## 框架阶段说明

当前实现为**框架阶段**，主要特点：

1. **基本架构已实现**：所有节点、服务器和工具模块的框架结构已创建
2. **核心功能已实现**：
   - 配置加载系统
   - Action Buffer（时间平滑）
   - WebSocket通信框架
   - ROS节点框架

3. **需要后续实现的功能**：
   - 数据预处理（RGB转点云、图像resize等）
   - 模型加载和推理（server端）
   - World Model更新逻辑（server端）
   - 实际的机器人SDK集成
   - ROS自定义消息类型（当前使用标准消息类型或字典）

## 依赖

参见 `requirements.txt`，主要包括：
- numpy
- msgpack
- websockets
- pyyaml
- rospy (ROS环境)
- message_filters (ROS环境)

## 更多信息

- 详细的集成说明请参考 `INTEGRATION.md`
- ROS消息类型说明请参考 `utils/ros_message_types.md`

