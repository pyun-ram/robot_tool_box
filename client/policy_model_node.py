"""Policy模型节点

订阅/sensor_sync_package话题，以20Hz频率处理消息，
通过WebSocket调用server，根据响应类型处理（update_world_model或action）。
"""

import cv2
import logging
import rospy
import threading
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加astribot_sdk路径
astribot_sdk_path = Path("/home/astribot/Desktop/base_ws/astribot_sdk")
if astribot_sdk_path.exists():
    sys.path.insert(0, str(astribot_sdk_path))
    from core.astribot_api.astribot_client import Astribot
else:
    logging.warning(f"astribot_sdk path not found: {astribot_sdk_path}")
    Astribot = None

from robot_tool_box.config_loader import get_client_config
from utils.websocket_utils import WebSocketClient
from utils.action_buffer import ActionBuffer
from utils.robot_sdk import RobotController
from utils.ros_message_helper import parse_sensor_sync_message_dict
from astribot_msgs.msg import GripperState, SyncPackage
from std_msgs.msg import Header

DEBUG = False

# 全局变量：是否保存观测数据
SAVE = False

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
        self.execution_config = client_config.get("execution_node", {}) 
        
        self.sync_topic = self.ros_config.get("sync_topic", "/sensor_sync_package")
        self.processing_rate = self.policy_config.get("processing_rate", 20.0)
        self.execution_rate = self.execution_config.get("execution_rate", 20.0)
        
        # 初始化锁（必须在订阅话题之前）
        self.lock = threading.Lock()
        
        # 存储最新接收到的数据
        self.latest_sync_data = None
        
        # Gripper历史状态
        self.gripper_history = []
        self.max_gripper_history = 3
        
        # 初始化ROS节点
        # rospy.init_node('policy_model_node', anonymous=True)

        if DEBUG:
            rospy.init_node('policy_model_node_debug', anonymous=True)
        else:
            self.astribot_client = Astribot()
            # self.astribot_client.move_to_home()
            body_name_list = ['astribot_torso',
                            'astribot_arm_left', 'astribot_gripper_left',
                            'astribot_arm_right', 'astribot_gripper_right']

            body_command_list = [[ 0.00236367, -1.25767e-06, 1.20031, -2.35307e-06, 0.00455332, -0.000916464, 0.999989],
                                 [ 0.255233,  0.27364, 0.806031,  0.00434999, 0.00392949,  0.709954, 0.704224],
                                 [ -0.132694],
                                 [ 0.4203003, -0.30904002, 0.92467913, 0.0052456, 0.00635652,  0.705191, 0.708969],
                                 [ 0.642704]]

            self.astribot_client.move_cartesian_pose(body_name_list, body_command_list, duration=10.0, use_wbc=False)
            # 等待用户按 Enter 开始
            input("Press Enter to start the robot...")
            rospy.loginfo("Starting robot control...")
            self.astribot_names = [self.astribot_client.arm_right_name, self.astribot_client.effector_right_name]
            self.end_effector_pose = self.astribot_client.get_current_cartesian_pose(names=self.astribot_names)
            # 定时器（20Hz执行action）
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.execution_rate), self._execution_callback)
        
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
        self.curr_gripper_buffer = ActionBuffer(3, smoothing_method, weights)
        
        # 订阅/sensor_sync_package话题
        # 设置queue_size=1，只保留最新消息，自动丢弃旧消息
        self.sub = rospy.Subscriber(self.sync_topic, SyncPackage, self._sync_callback, queue_size=1) ######

        self.sub_curr_gripper = rospy.Subscriber('/curr_gripper', GripperState, self._curr_gripper_callback)
        
        # self.astribot_client = Astribot(freq=self.execution_rate)


        rospy.loginfo(f"Policy model node initialized, processing at {self.processing_rate} Hz, executing at {self.execution_rate} Hz")
    
    def _sync_callback(self, msg: SyncPackage):
        """
        同步数据回调函数
        
        在这里处理接收到的同步数据
        """

        # 快速提取第一个RGB图像用于保存
        if len(msg.rgb_list) > 0:
            np_rgb = np.frombuffer(msg.rgb_list[0].data, dtype=np.uint8)
            rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
            if rgb_img is not None:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"rgbs_before_lock_{time.time():.6f}.png", rgb_img)

        
        with self.lock:
            
            # 数据预处理：格式化Observation
            obs = self._preprocess_data(msg)
            
            if obs is None:
                return
            
            # 通过WebSocket调用server
            try:
                response = self.ws_client.send_and_receive(obs)
                # print(response)
                
                # 根据响应类型处理
                response_type = response.get("type")
                
                if response_type == "update_world_model":
                    # 仅接收通知，不更新Action Buffer
                    print("Received update_world_model response")
                elif response_type == "action":
                    # 接收arm+gripper，更新Action Buffer
                    arm_gripper_pos = response.get("arm_gripper_pos")
                    if arm_gripper_pos is not None:
                        arm_gripper_pos = np.array(arm_gripper_pos, dtype=np.float32)
                        
                        # 如果SAVE为True，保存obs和arm_gripper_pos
                        if SAVE:
                            # 将arm_gripper_pos添加到obs字典中
                            obs_with_action = obs.copy()
                            obs_with_action["arm_gripper_pos"] = arm_gripper_pos
                            
                            # 创建保存目录
                            save_dir = project_root / "tools" / "output"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 生成文件名（包含时间戳）
                            timestamp = time.time()
                            filename = f"obs_{timestamp:.6f}.npy"
                            save_path = save_dir / filename
                            
                            # 保存为.npy文件
                            np.save(str(save_path), obs_with_action)
                            rospy.loginfo(f"Saved observation to {save_path}")
                        
                        self.action_buffer.add_action(arm_gripper_pos)
                        # print(f"Added action to buffer: {arm_gripper_pos.shape}")
                else:
                    rospy.logwarn(f"Unknown response type: {response_type}")
            except Exception as e:
                rospy.logerr(f"Error calling server: {e}")
    
    def _execution_callback(self, event):
        """
        执行回调函数（20Hz）
        从Action Buffer读取平滑后的action，发送到机器人
        """
        # 从Action Buffer读取平滑后的action
        # arm_gripper_pos = self.action_buffer.get_smoothed_action()
        arm_gripper_pos = self.action_buffer.get_latest_action()
        
        if arm_gripper_pos is None:
            # Buffer为空，跳过本次执行
            return
        
        # 通过机器人控制器发送joint commands
        # arm_gripper_pos 形状为 (8,)，前7个是位置+四元数，最后1个是gripper状态
        # self.end_effector_pose[0] 需要是列表（7个元素：x, y, z, qx, qy, qz, qw）
        # self.end_effector_pose[1] 需要是列表（1个元素：gripper状态）
        self.end_effector_pose[0] = arm_gripper_pos[:-1].tolist()  # 转换为列表
        # z方向补偿
        # self.end_effector_pose[0][2] -= 0.005
        # self.end_effector_pose[1] = [arm_gripper_pos[-1]]
        self.end_effector_pose[1] = [50] if arm_gripper_pos[-1] > 0.8 else [0] # 包装成列表
        # self.astribot_client.set_cartesian_pose(self.astribot_names, self.end_effector_pose)
        print(f"end_effector_pose: {self.end_effector_pose}")
        self.astribot_client.move_cartesian_pose(self.astribot_names, self.end_effector_pose, duration=3.0, use_wbc=False)
    
    def _preprocess_data(self, sync_data_msg: SyncPackage) -> Optional[Dict]:
        """
        数据预处理：格式化Observation
        
        Args:
            sync_data_msg: 同步数据字典
            
        Returns:
            Observation字典
        """
        # 实际实现时需要：
        # 1. 从sync_data_msg中提取RGB和Depth图像
        # 2. 使用相机参数将深度图转换为点云
        # 3. 格式化数据
        
        rgb_images = []
        for rgb_msg in sync_data_msg.rgb_list:
            # 解压压缩的RGB图像
            np_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
            # OpenCV默认使用BGR格式，需要转换为RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_img)

        # 提取深度图像并转换为点云
        point_clouds = []
        for depth_msg in sync_data_msg.depth_list:
            # 假设深度图为16位单通道图像
            depth_img = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)

            # 深度图转点云
            # 使用相机的内参进行转换（如焦距、光心），假设有一个depth_to_point_cloud函数
            # point_cloud = depth_to_point_cloud(depth_img, camera_intrinsics=self.camera_config)
            # point_clouds.append(point_cloud)
            point_clouds.append(depth_img)

        # 处理夹爪状态
        gripper_hist = np.array(sync_data_msg.gripper_hist.data, dtype=np.float32)  # 假设夹爪状态为一个浮点数组
        assert gripper_hist.shape == (24,), f"gripper_hist shape should be (24,), but got {gripper_hist.shape}"  

        # 获取夹爪历史记录（你可以在这里调用相应的历史方法）
        # gripper_history = self._get_gripper_history(gripper_state)
        # gripper_history = self.curr_gripper_buffer._buffer.copy()
        # 将deque转换为numpy数组
        gripper_history_array = gripper_hist.reshape(3, 8)
        # print(f"gripper_history shape: {gripper_history_array.shape}")
        # print(f"gripper_history: {gripper_history_array}")

        # 格式化Observation
        observation = {
            "rgbs": np.stack(rgb_images, axis=0),  # (4, 3, H, W)?
            "pcds": np.stack(point_clouds, axis=0),  # (4, 1, H, W)?
            "curr_gripper": gripper_history_array[-1],  # (8,)
            "curr_gripper_history": gripper_history_array,  # (3, 8)
            "instr": None,  # (53, 512) - 可选，根据需要填充
        }

        print(observation['rgbs'].shape)
        n, h, w, c = observation['rgbs'].shape
        debug_rgbs = observation['rgbs'].reshape((n * h, w, 3))
        cv2.imwrite(f"debug_rgbs_{time.time()}.png", debug_rgbs)
        
        # for key, value in observation.items():
        #     if value is not None:
        #         print(f"{key}: {value.shape}")
        #     else:
        #         print(f"{key}: None")
    
        return observation

    def _curr_gripper_callback(self, msg: GripperState):
        """
        当前夹爪状态回调函数
        """
        self.curr_gripper_buffer.add_action(np.array(msg.data, dtype=np.float32))

    
    def _get_gripper_history(self, gripper_pose: np.ndarray) -> np.ndarray: 
        """ 更新gripper历史状态 Args: gripper_pose: 当前gripper pose，shape为(8,) """ 
        self.gripper_history.append(gripper_pose.copy()) 
        if len(self.gripper_history) > self.max_gripper_history: 
            self.gripper_history.pop(0) 
        
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

