"""Policy模型节点（验证版本）

从本地文件系统读取数据，以指定间隔处理消息，
通过WebSocket调用server，根据响应类型处理（update_world_model或action）。
"""

import cv2
import logging
import sys
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
    try:
        from core.astribot_api.astribot_client import Astribot
        HAS_ASTRIBOT = True
    except ImportError:
        HAS_ASTRIBOT = False
        Astribot = None
        logging.warning(f"Failed to import Astribot from {astribot_sdk_path}")
else:
    HAS_ASTRIBOT = False
    Astribot = None
    logging.warning(f"astribot_sdk path not found: {astribot_sdk_path}")

from robot_tool_box.config_loader import get_client_config
from utils.websocket_utils import WebSocketClient
from utils.action_buffer import ActionBuffer


class PolicyModelNode:
    """Policy模型节点类（验证版本 - 从本地文件读取数据）"""
    
    def __init__(self, data_path: str, config_path: str = None, enable_robot: bool = False):
        """
        初始化Policy模型节点
        
        Args:
            data_path: 数据文件路径（包含cam1_color, cam1_depth等文件夹的根目录）
            config_path: 配置文件路径
            enable_robot: 是否启用机器人控制（True则实际发送指令到机器人，False则仅模拟）
        """
        # 加载配置
        client_config = get_client_config(config_path)
        self.websocket_config = client_config.get("websocket", {})
        self.camera_config = client_config.get("camera", {})
        self.action_buffer_config = client_config.get("action_buffer", {})
        self.execution_config = client_config.get("execution_node", {})
        
        # 数据路径配置
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # 相机文件夹名称
        self.camera_folders = {
            'cam1': {'color': 'cam1_color', 'depth': 'cam1_depth'},
            'cam2': {'color': 'cam2_color', 'depth': 'cam2_depth'},
            'cam3': {'color': 'cam3_color', 'depth': 'cam3_depth'},
            'cam4': {'color': 'cam4_color', 'depth': 'cam4_depth'},
        }
        
        # 读取间隔和起始索引
        self.read_interval = 10  # 以10为间隔
        self.current_index = 0
        self.enable_robot = True #enable_robot
        
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
        
        # Gripper历史状态
        self.gripper_history = []
        self.max_gripper_history = 3
        
        # 检查数据文件是否存在（在初始化机器人之前检查，因为需要读取robot_state/0.npy）
        self._check_data_files()
        
        # 初始化Astribot客户端（如果启用机器人控制）
        self.astribot_client = None
        self.astribot_names = None
        self.end_effector_pose = None
        
        if self.enable_robot:
            if not HAS_ASTRIBOT:
                raise ValueError("Astribot SDK not available. Cannot enable robot control.")
            
            self.astribot_client = Astribot()
            
            # 设置初始位置
            # 从robot_state/0.npy读取arm_right和gripper_right的初始数据
            robot_state_file = self.data_path / "robot_state" / "0.npy"
            robot_state_0 = np.load(robot_state_file)
            robot_state_0 = np.array(robot_state_0, dtype=np.float32)
            
            # 构造body_command_list
            # 'astribot_torso', 'astribot_arm_left', 'astribot_gripper_left' 使用固定值
            # 'astribot_arm_right' 使用 robot_state/0.npy[:7]（位置+四元数，7个元素）
            # 'astribot_gripper_right' 使用 robot_state/0.npy[7]（gripper状态，1个元素）
            body_name_list = ['astribot_torso',
                            'astribot_arm_left', 'astribot_gripper_left',
                            'astribot_arm_right', 'astribot_gripper_right']
            
            body_command_list = [
                [0.00236367, -1.25767e-06, 1.20031, -2.35307e-06, 0.00455332, -0.000916464, 0.999989],  # astribot_torso
                [0.255233, 0.27364, 0.806031, 0.00434999, 0.00392949, 0.709954, 0.704224],  # astribot_arm_left
                [-0.132694],  # astribot_gripper_left
                robot_state_0[:7].tolist(),  # astribot_arm_right: 使用robot_state/0.npy的前7个元素
                [robot_state_0[7]],  # astribot_gripper_right: 使用robot_state/0.npy的第8个元素
            ]
            
            logging.info(f"Moving robot to initial position from robot_state/0.npy...")
            logging.info(f"Arm right initial pose: {body_command_list[3]}")
            logging.info(f"Gripper right initial state: {body_command_list[4]}")
            self.astribot_client.move_cartesian_pose(body_name_list, body_command_list, duration=10.0, use_wbc=False)
            
            # 等待用户确认
            input("Press Enter to start robot control...")
            logging.info("Starting robot control...")
            
            self.astribot_names = [self.astribot_client.arm_right_name, self.astribot_client.effector_right_name]
            self.end_effector_pose = self.astribot_client.get_current_cartesian_pose(names=self.astribot_names)
            logging.info(f"Initial end effector pose: {self.end_effector_pose}")
        
        logging.info(f"Policy model node initialized, data path: {self.data_path}")
        logging.info(f"Reading data with interval: {self.read_interval}")
        logging.info(f"Robot control enabled: {self.enable_robot}")
    
    def _check_data_files(self):
        """检查数据文件是否存在"""
        # 检查相机文件夹
        for cam_name, folders in self.camera_folders.items():
            color_path = self.data_path / folders['color']
            depth_path = self.data_path / folders['depth']
            if not color_path.exists():
                raise ValueError(f"Camera color folder not found: {color_path}")
            if not depth_path.exists():
                raise ValueError(f"Camera depth folder not found: {depth_path}")
        
        # 检查robot_state文件夹
        robot_state_path = self.data_path / "robot_state"
        if not robot_state_path.exists():
            raise ValueError(f"Robot state folder not found: {robot_state_path}")
        
        # 检查第一个文件是否存在（索引0）
        test_file = robot_state_path / "0.npy"
        if not test_file.exists():
            raise ValueError(f"Test file not found: {test_file}")
        
        logging.info("Data files check passed")
    
    def _load_observation_from_files(self, index: int) -> Optional[Dict]:
        """
        从本地文件加载observation数据
        
        Args:
            index: 文件索引（如0, 10, 20, ...）
            
        Returns:
            observation字典，如果文件不存在则返回None
        """
        rgb_images = []
        depth_images = []
        
        # 读取4个相机的RGB和Depth图像
        for cam_name, folders in self.camera_folders.items():
            # 读取RGB图像
            color_file = self.data_path / folders['color'] / f"{index}.png"
            if not color_file.exists():
                logging.warning(f"Color file not found: {color_file}")
                return None
            
            rgb_img = cv2.imread(str(color_file), cv2.IMREAD_COLOR)
            if rgb_img is None:
                logging.warning(f"Failed to load color image: {color_file}")
                return None
            # 转换为RGB格式（OpenCV默认是BGR）
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_img)
            
            # 读取Depth图像
            depth_file = self.data_path / folders['depth'] / f"{index}.png"
            if not depth_file.exists():
                logging.warning(f"Depth file not found: {depth_file}")
                return None
            
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                logging.warning(f"Failed to load depth image: {depth_file}")
                return None
            # 如果是16位深度图，保持原样；如果是8位，可能需要转换
            depth_images.append(depth_img)
        
        # 读取robot_state
        robot_state_file = self.data_path / "robot_state" / f"{index}.npy"
        if not robot_state_file.exists():
            logging.warning(f"Robot state file not found: {robot_state_file}")
            return None
        
        robot_state = np.load(robot_state_file)
        gripper_state = np.array(robot_state, dtype=np.float32)
        
        # 获取夹爪历史记录
        gripper_history = self._get_gripper_history(gripper_state)
        
        # 格式化Observation
        observation = {
            "rgbs": np.stack(rgb_images, axis=0),  # (4, H, W, 3)
            "pcds": np.stack(depth_images, axis=0),  # (4, H, W)
            "curr_gripper": gripper_state,  # (8,)
            "curr_gripper_history": gripper_history,  # (3, 8)
            "instr": None,  # (53, 512) - 可选，根据需要填充
        }
        
        return observation
    
    def _preprocess_data(self, obs: Dict) -> Optional[Dict]:
        """
        数据预处理：格式化Observation（如果需要额外处理）
        
        Args:
            obs: observation字典（已经从文件加载）
            
        Returns:
            Observation字典
        """
        # 这里可以进行额外的预处理，比如：
        # 1. 调整图像尺寸
        # 2. 归一化等
        # 目前直接返回原始数据
        return obs

    
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
    
    def process_single_observation(self, index: int) -> bool:
        """
        处理单个observation
        
        Args:
            index: 文件索引
            
        Returns:
            是否成功处理
        """
        # 从文件加载observation
        obs = self._load_observation_from_files(index)
        if obs is None:
            logging.warning(f"Failed to load observation at index {index}")
            return False
        
        # 预处理
        obs = self._preprocess_data(obs)
        if obs is None:
            return False
        
        # 通过WebSocket调用server
        try:
            response = self.ws_client.send_and_receive(obs)
            print(f"Index {index} - Response: {response}")
            
            # 根据响应类型处理
            response_type = response.get("type")
            
            if response_type == "update_world_model":
                # 仅接收通知，不更新Action Buffer
                print(f"Index {index} - Received update_world_model response")
            elif response_type == "action":
                # 接收arm+gripper，更新Action Buffer
                arm_gripper_pos = response.get("arm_gripper_pos")
                if arm_gripper_pos is not None:
                    arm_gripper_pos = np.array(arm_gripper_pos, dtype=np.float32)
                    self.action_buffer.add_action(arm_gripper_pos)
                    print(f"Index {index} - Added action to buffer: {arm_gripper_pos.shape}")
                    
                    # 如果启用机器人控制，执行action
                    if self.enable_robot:
                        self._execute_action(arm_gripper_pos)
            else:
                logging.warning(f"Index {index} - Unknown response type: {response_type}")
            
            return True
        except Exception as e:
            logging.error(f"Index {index} - Error calling server: {e}")
            return False
    
    def _execute_action(self, arm_gripper_pos: np.ndarray):
        """
        执行action到机器人
        
        Args:
            arm_gripper_pos: action数组，shape为(8,)，前7个是位置+四元数，最后1个是gripper状态
        """
        if self.astribot_client is None or self.astribot_names is None or self.end_effector_pose is None:
            logging.warning("Robot client not initialized, skipping action execution")
            return
        
        try:
            # arm_gripper_pos 形状为 (8,)，前7个是位置+四元数，最后1个是gripper状态
            # self.end_effector_pose[0] 需要是列表（7个元素：x, y, z, qx, qy, qz, qw）
            # self.end_effector_pose[1] 需要是列表（1个元素：gripper状态）
            self.end_effector_pose[0] = arm_gripper_pos[:-1].tolist()  # 转换为列表
            # 根据gripper状态决定gripper命令：大于0.8则打开(50)，否则关闭(0)
            self.end_effector_pose[1] = [50] if arm_gripper_pos[-1] > 0.8 else [0]
            
            # 发送到机器人
            self.astribot_client.move_cartesian_pose(
                self.astribot_names, 
                self.end_effector_pose, 
                duration=2.0, 
                use_wbc=False
            )
            logging.debug(f"Executed action: pose={self.end_effector_pose[0]}, gripper={self.end_effector_pose[1]}")
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def run(self):
        """运行节点，循环读取文件并处理"""
        logging.info("Policy model node started, reading from local files...")
        
        # 确定文件范围
        robot_state_path = self.data_path / "robot_state"
        max_index = self._find_max_index(robot_state_path)
        
        if max_index is None:
            logging.error("No valid data files found")
            return
        
        logging.info(f"Found data files from index 0 to {max_index}")
        logging.info(f"Processing with interval {self.read_interval}...")
        
        # 循环处理所有文件（以10为间隔）
        current_index = 0
        while current_index <= max_index:
            logging.info(f"Processing index {current_index}...")
            
            success = self.process_single_observation(current_index)
            if not success:
                logging.warning(f"Skipping index {current_index} due to error")
            
            # 暂停，等待用户按Enter继续
            if current_index < max_index:
                try:
                    input(f"Processed index {current_index}. Press Enter to continue to index {current_index + self.read_interval}...")
                except (EOFError, KeyboardInterrupt):
                    logging.info("Interrupted by user, stopping...")
                    break
            
            # 移动到下一个索引
            current_index += self.read_interval
        
        logging.info("All data processed")
    
    def _find_max_index(self, robot_state_path: Path) -> Optional[int]:
        """
        查找最大的文件索引
        
        Args:
            robot_state_path: robot_state文件夹路径
            
        Returns:
            最大索引，如果没有找到则返回None
        """
        max_index = None
        for npy_file in robot_state_path.glob("*.npy"):
            try:
                # 从文件名提取索引（如"0.npy" -> 0）
                index = int(npy_file.stem)
                if max_index is None or index > max_index:
                    max_index = index
            except ValueError:
                continue
        
        return max_index
    
    def shutdown(self):
        """关闭节点"""
        if self.ws_client is not None:
            self.ws_client.close()
        
        # 关闭机器人连接（如果需要）
        if self.astribot_client is not None:
            # 如果Astribot有close方法，调用它
            # 目前只是记录日志
            logging.info("Closing robot connection...")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Policy Model Node (Validation - reads from local files)")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory containing cam1_color, cam1_depth, etc. folders"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--enable_robot",
        action="store_true",
        help="Enable robot control (send actual commands to robot)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        node = PolicyModelNode(
            data_path=args.data_path, 
            config_path=args.config_path,
            enable_robot=args.enable_robot
        )
        node.run()
        node.shutdown()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        if 'node' in locals():
            node.shutdown()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        if 'node' in locals():
            node.shutdown()


if __name__ == '__main__':
    main()

