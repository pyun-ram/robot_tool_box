"""执行节点

从Action Buffer读取joint angles，以20Hz频率执行，
通过机器人SDK发送joint commands到机器人。
"""

import rospy
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_client_config
from utils.action_buffer import ActionBuffer
from utils.robot_sdk import RobotController


class ExecutionNode:
    """执行节点类"""
    
    def __init__(self, config_path: str = None, action_buffer: Optional[ActionBuffer] = None):
        """
        初始化执行节点
        
        Args:
            config_path: 配置文件路径
            action_buffer: Action Buffer实例（可选，如果不提供则创建新实例）
        """
        # 加载配置
        client_config = get_client_config(config_path)
        self.ros_config = client_config.get("ros", {})
        self.execution_config = client_config.get("execution_node", {})
        self.robot_config = client_config.get("robot", {})
        self.action_buffer_config = client_config.get("action_buffer", {})
        
        self.execution_rate = self.execution_config.get("execution_rate", 20.0)
        
        # 初始化ROS节点
        rospy.init_node('execution_node', anonymous=True)
        
        # 初始化Action Buffer（如果未提供）
        if action_buffer is None:
            buffer_size = self.action_buffer_config.get("buffer_size", 10)
            smoothing_method = self.action_buffer_config.get("smoothing_method", "weighted_average")
            weights = self.action_buffer_config.get("weights", "exponential")
            self.action_buffer = ActionBuffer(buffer_size, smoothing_method, weights)
        else:
            self.action_buffer = action_buffer
        
        # 初始化机器人控制器
        robot_config_dict = {
            "use_ros": True,
            "joint_command_topic": self.robot_config.get("joint_command_topic", "/joint_command"),
        }
        self.robot_controller = RobotController(robot_config_dict)
        self.robot_controller.initialize()
        
        # 定时器（20Hz执行）
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.execution_rate), self._execution_callback)
        
        rospy.loginfo(f"Execution node initialized, executing at {self.execution_rate} Hz")
    
    def _execution_callback(self, event):
        """执行回调函数（20Hz）"""
        # 从Action Buffer读取平滑后的joint angles
        joint_angles = self.action_buffer.get_smoothed_action()
        
        if joint_angles is None:
            # Buffer为空，跳过本次执行
            return
        
        # 通过机器人控制器发送joint commands
        success = self.robot_controller.send_joint_command(joint_angles)
        
        if not success:
            rospy.logwarn("Failed to send joint command")
    
    def run(self):
        """运行节点"""
        rospy.loginfo("Execution node started")
        rospy.spin()
    
    def shutdown(self):
        """关闭节点"""
        if self.robot_controller is not None:
            self.robot_controller.close()


def main():
    """主函数"""
    try:
        node = ExecutionNode()
        rospy.on_shutdown(node.shutdown)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

