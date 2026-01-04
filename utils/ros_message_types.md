# ROS自定义消息类型说明

## /sensor_sync/data 话题消息格式

为了简化实现，我们可以使用标准的ROS消息类型组合，或者创建自定义消息类型。

### 方案1：使用标准消息类型（推荐用于框架阶段）

使用`sensor_msgs/Image`和`geometry_msgs/Pose`等标准消息类型组合：

```python
# 消息结构（伪代码）
{
    "cam1_rgb": sensor_msgs.msg.Image,
    "cam1_depth": sensor_msgs.msg.Image,
    "cam2_rgb": sensor_msgs.msg.Image,
    "cam2_depth": sensor_msgs.msg.Image,
    "cam3_rgb": sensor_msgs.msg.Image,
    "cam3_depth": sensor_msgs.msg.Image,
    "cam4_rgb": sensor_msgs.msg.Image,
    "cam4_depth": sensor_msgs.msg.Image,
    "gripper_pose": geometry_msgs.msg.Pose  # 或自定义消息，包含[x,y,z,rx,ry,rz,rw,openness]
}
```

### 方案2：创建自定义消息类型（实际部署时推荐）

需要创建一个ROS package来定义自定义消息类型：

1. 创建package：
```bash
cd ~/catkin_ws/src
catkin_create_pkg robot_tool_box_msgs std_msgs sensor_msgs geometry_msgs
cd robot_tool_box_msgs
mkdir msg
```

2. 创建消息定义文件 `msg/SensorSyncData.msg`：
```
# 4个相机的RGB图像
sensor_msgs/Image cam1_rgb
sensor_msgs/Image cam1_depth
sensor_msgs/Image cam2_rgb
sensor_msgs/Image cam2_depth
sensor_msgs/Image cam3_rgb
sensor_msgs/Image cam3_depth
sensor_msgs/Image cam4_rgb
sensor_msgs/Image cam4_depth

# Gripper位姿 [x,y,z,rx,ry,rz,rw,openness]
float64[8] gripper_pose

# 时间戳
std_msgs/Header header
```

3. 在`CMakeLists.txt`和`package.xml`中配置消息生成

4. 编译：
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 框架实现建议

在框架实现阶段，建议使用方案1（标准消息类型），或者使用Python字典在节点内部传递数据。实际部署时再使用方案2（自定义消息类型）。

