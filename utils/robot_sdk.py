"""机器人SDK工具模块

封装机器人控制接口，用于发送joint commands到机器人。
"""

import logging
from typing import Optional
import numpy as np


class RobotController:
    """机器人控制器类，用于发送joint commands"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化机器人控制器
        
        Args:
            config: 机器人配置字典（可选）
        """
        self.config = config or {}
        self._initialized = False
        self._joint_command_publisher = None
        
        # ROS相关（如果使用ROS）
        self._use_ros = self.config.get("use_ros", True)
        self._joint_command_topic = self.config.get("joint_command_topic", "/joint_command")
    
    def initialize(self) -> None:
        """初始化机器人连接"""
        if self._initialized:
            return
        
        if self._use_ros:
            try:
                import rospy
                from std_msgs.msg import Float64MultiArray
                
                if not rospy.get_node_uri():
                    rospy.init_node('robot_controller', anonymous=True)
                
                self._joint_command_publisher = rospy.Publisher(
                    self._joint_command_topic,
                    Float64MultiArray,
                    queue_size=1
                )
                
                # 等待publisher连接
                rospy.sleep(0.1)
                
                logging.info(f"Robot controller initialized with ROS topic: {self._joint_command_topic}")
            except ImportError:
                logging.warning("rospy not available, robot controller will use direct SDK")
                self._use_ros = False
        
        self._initialized = True
    
    def send_joint_command(self, joint_angles: np.ndarray) -> bool:
        """
        发送joint angles到机器人
        
        Args:
            joint_angles: joint angles数组，shape为(N,)，N为关节数量
            
        Returns:
            是否发送成功
        """
        if not self._initialized:
            self.initialize()
        
        try:
            joint_angles = np.array(joint_angles, dtype=np.float64)
            
            if self._use_ros and self._joint_command_publisher is not None:
                # 使用ROS发布
                from std_msgs.msg import Float64MultiArray
                
                msg = Float64MultiArray()
                msg.data = joint_angles.tolist()
                self._joint_command_publisher.publish(msg)
                return True
            else:
                # 使用直接SDK（需要根据实际机器人SDK实现）
                logging.warning("Direct SDK not implemented, please use ROS or implement SDK")
                return False
        except Exception as e:
            logging.error(f"Error sending joint command: {e}")
            return False
    
    def get_joint_states(self) -> Optional[np.ndarray]:
        """
        获取当前关节状态（可选）
        
        Returns:
            当前关节角度数组，如果失败则返回None
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if self._use_ros:
                # 使用ROS订阅（需要实现订阅逻辑）
                # 这里只是框架，实际实现需要根据具体需求
                logging.warning("get_joint_states not fully implemented")
                return None
            else:
                # 使用直接SDK
                logging.warning("Direct SDK get_joint_states not implemented")
                return None
        except Exception as e:
            logging.error(f"Error getting joint states: {e}")
            return None
    
    def close(self) -> None:
        """关闭连接"""
        if self._joint_command_publisher is not None:
            self._joint_command_publisher.unregister()
            self._joint_command_publisher = None
        
        self._initialized = False
        logging.info("Robot controller closed")

