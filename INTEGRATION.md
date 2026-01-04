# 系统集成说明

## 节点间的集成方式

### 1. ROS话题通信

- **sensor_sync_node** → **policy_model_node**: 通过ROS话题 `/sensor_sync/data` 通信
  - sensor_sync_node以20Hz频率发布同步后的传感器数据
  - policy_model_node订阅该话题，以20Hz频率处理

### 2. Action Buffer共享

- **policy_model_node** → **execution_node**: 通过Action Buffer共享数据
  - policy_model_node将接收到的action（joint angles）添加到Action Buffer
  - execution_node从Action Buffer读取平滑后的action并执行
  - Action Buffer是线程安全的数据结构，支持多线程读写

### 3. WebSocket通信

- **policy_model_node** ↔ **policy_server**: 通过WebSocket通信
  - policy_model_node发送observation到server
  - server返回响应（类型：update_world_model或action）
  - policy_model_node根据响应类型处理

## 执行流程

### 完整数据流

1. **ROS原始Topics** → **sensor_sync_node**
   - 4个相机的RGB/Depth图像
   - Gripper pose

2. **sensor_sync_node** → **/sensor_sync/data Topic** (20Hz)
   - 同步后的传感器数据

3. **/sensor_sync/data** → **policy_model_node** (20Hz)
   - 数据预处理（RGB转点云等）
   - 格式化Observation

4. **policy_model_node** → **WebSocket Server**
   - 发送Observation

5. **WebSocket Server** → **policy_model_node**
   - 返回响应（update_world_model或action）

6. **policy_model_node** → **Action Buffer**
   - 如果收到action类型，添加joint angles到buffer

7. **Action Buffer** → **execution_node** (20Hz)
   - 读取平滑后的joint angles

8. **execution_node** → **机器人执行** (20Hz)
   - 通过ROS话题或机器人SDK发送joint commands

## 部署说明

### 框架阶段

当前实现为框架阶段，主要特点：
- 使用标准ROS消息类型或字典传递数据
- 数据预处理和模型推理为占位函数
- Action Buffer和WebSocket通信已实现基本功能

### 实际部署时需要的改进

1. **ROS消息类型**：创建自定义消息类型用于 `/sensor_sync/data` 话题
2. **数据预处理**：实现完整的RGB/Depth转点云、图像resize等逻辑
3. **模型加载**：在server端加载实际的policy model和world model
4. **机器人SDK**：根据实际机器人实现SDK接口
5. **错误处理**：完善错误处理和日志记录
6. **配置验证**：添加配置文件的验证逻辑

## 运行方式

### 启动服务器

```bash
python tools/run_server.py
```

### 启动客户端节点（在不同终端）

```bash
# 终端1: 传感器同步节点
python tools/run_sensor_sync.py

# 终端2: Policy模型节点
python tools/run_policy_model.py

# 终端3: 执行节点
python tools/run_execution.py
```

## 配置文件

所有配置在 `config/config.yaml` 中，包括：
- 服务器配置（host, port, model_type等）
- 客户端配置（ROS话题、WebSocket连接、Action Buffer参数等）

## 注意事项

1. 确保ROS环境已正确设置
2. 确保所有依赖包已安装
3. 根据实际机器人调整配置文件
4. 框架阶段的某些功能需要根据实际需求实现

