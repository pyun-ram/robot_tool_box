"""Policy模型节点

订阅/sensor_sync/data话题，以20Hz频率处理消息，
通过WebSocket调用server，根据响应类型处理（update_world_model或action）。
"""

import rospy
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_client_config
from utils.websocket_utils import WebSocketClient
from utils.action_buffer import ActionBuffer
from utils.ros_message_helper import parse_sensor_sync_message_dict
from std_msgs.msg import Header


class PolicyModelNode:
    """Policy模型节点类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化Policy模型节点
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        client_config = get_client_config(config_path)
        self.ros_config = client_config.get("ros", {})
        self.websocket_config = client_config.get("websocket", {})
        self.camera_config = client_config.get("camera", {})
        self.action_buffer_config = client_config.get("action_buffer", {})
        self.policy_config = client_config.get("policy_model_node", {})
        
        self.sync_topic = self.ros_config.get("sync_topic", "/sensor_sync/data")
        self.processing_rate = self.policy_config.get("processing_rate", 20.0)
        
        # 初始化ROS节点
        rospy.init_node('policy_model_node', anonymous=True)
        
        # 初始化WebSocket客户端
        server_uri = self.websocket_config.get("server_uri", "ws://10.11.5.2:8000")
        api_key = self.websocket_config.get("api_key")
        reconnect_interval = self.websocket_config.get("reconnect_interval", 5.0)
        self.ws_client = WebSocketClient(server_uri, api_key, reconnect_interval)
        
        # 初始化Action Buffer
        buffer_size = self.action_buffer_config.get("buffer_size", 10)
        smoothing_method = self.action_buffer_config.get("smoothing_method", "weighted_average")
        weights = self.action_buffer_config.get("weights", "exponential")
        self.action_buffer = ActionBuffer(buffer_size, smoothing_method, weights)
        
        # 订阅/sensor_sync/data话题
        # 注意：框架阶段使用Header消息作为占位符，实际部署时使用自定义消息类型
        self.sub = rospy.Subscriber(self.sync_topic, Header, self._sync_callback)
        
        # 存储最新接收到的数据（框架阶段，实际应该从消息中获取）
        self.latest_sync_data = None
        self.lock = rospy.Lock()
        
        # Gripper历史状态（用于curr_gripper_history）
        self.gripper_history = []
        self.max_gripper_history = 3
        
        # 定时器（20Hz处理）
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.processing_rate), self._process_callback)
        
        rospy.loginfo(f"Policy model node initialized, processing at {self.processing_rate} Hz")
    
    def _sync_callback(self, msg: Header):
        """
        同步数据回调函数
        
        注意：框架阶段，实际数据需要通过其他机制获取（如参数服务器、共享内存等）
        实际部署时应该从自定义消息类型中获取
        """
        # 框架阶段：存储时间戳，实际数据需要通过其他机制获取
        with self.lock:
            self.latest_sync_data = {
                "timestamp": msg.stamp,
                "header": msg
            }
    
    def _process_callback(self, event):
        """处理回调函数（20Hz）"""
        with self.lock:
            if self.latest_sync_data is None:
                return
            
            # 框架阶段：这里应该从latest_sync_data中获取实际的传感器数据
            # 实际部署时，应该从自定义消息类型中解析数据
            sync_data_dict = self._get_sync_data()
            
            if sync_data_dict is None:
                return
            
            # 数据预处理：格式化Observation
            obs = self._preprocess_data(sync_data_dict)
            
            if obs is None:
                return
            
            # 通过WebSocket调用server
            try:
                response = self.ws_client.send_and_receive(obs)
                
                # 根据响应类型处理
                response_type = response.get("type")
                
                if response_type == "update_world_model":
                    # 仅接收通知，不更新Action Buffer
                    rospy.logdebug("Received update_world_model response")
                elif response_type == "action":
                    # 接收joint angles，更新Action Buffer
                    joint_angles = response.get("joint_angles")
                    if joint_angles is not None:
                        joint_angles = np.array(joint_angles, dtype=np.float32)
                        self.action_buffer.add_action(joint_angles)
                        rospy.logdebug(f"Added action to buffer: {joint_angles.shape}")
                else:
                    rospy.logwarn(f"Unknown response type: {response_type}")
            except Exception as e:
                rospy.logerr(f"Error calling server: {e}")
    
    def _get_sync_data(self) -> Optional[Dict]:
        """
        获取同步数据（框架阶段）
        
        实际部署时应该从自定义消息类型中解析
        """
        # 框架阶段：返回None，实际实现时需要从消息中解析
        # 这里只是框架结构
        return None
    
    def _preprocess_data(self, sync_data_dict: Dict) -> Optional[Dict]:
        """
        数据预处理：格式化Observation
        
        Args:
            sync_data_dict: 同步数据字典
            
        Returns:
            Observation字典
        """
        # 框架阶段：这里应该实现实际的数据预处理逻辑
        # 包括：RGB图像处理、深度图转点云、格式化Observation等
        
        # 框架结构：
        obs = {
            "rgbs": None,  # (4, 3, H, W)
            "pcds": None,  # (4, 3, H, W)
            "curr_gripper": None,  # (1, 8)
            "curr_gripper_history": None,  # (3, 8)
            "instr": None,  # (53, 512) - 可选
        }
        
        # 实际实现时需要：
        # 1. 从sync_data_dict中提取RGB和Depth图像
        # 2. 使用相机参数将深度图转换为点云
        # 3. 处理gripper pose
        # 4. 格式化数据
        
        return None  # 框架阶段返回None
    
    def _update_gripper_history(self, gripper_pose: np.ndarray):
        """
        更新gripper历史状态
        
        Args:
            gripper_pose: 当前gripper pose，shape为(8,)
        """
        self.gripper_history.append(gripper_pose.copy())
        if len(self.gripper_history) > self.max_gripper_history:
            self.gripper_history.pop(0)
    
    def _get_gripper_history(self) -> np.ndarray:
        """
        获取gripper历史状态
        
        Returns:
            gripper历史状态数组，shape为(3, 8)
        """
        if len(self.gripper_history) == 0:
            return np.zeros((3, 8), dtype=np.float32)
        
        # 填充到3个
        history = np.array(self.gripper_history)
        if len(history) < 3:
            # 用第一个状态填充
            padding = np.tile(history[0:1], (3 - len(history), 1))
            history = np.vstack([padding, history])
        
        return history.astype(np.float32)
    
    def run(self):
        """运行节点"""
        rospy.loginfo("Policy model node started")
        rospy.spin()
    
    def shutdown(self):
        """关闭节点"""
        if self.ws_client is not None:
            self.ws_client.close()


def main():
    """主函数"""
    try:
        node = PolicyModelNode()
        rospy.on_shutdown(node.shutdown)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

