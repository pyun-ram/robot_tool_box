"""Policy Server

统一WebSocket服务器，接收policy_model_node的请求，
根据模型类型和内部逻辑返回不同类型的响应（update_world_model或action）。
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_server_config

# 尝试导入websocket server（如果可用）
try:
    from openpi.serving import websocket_policy_server
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    logging.warning("openpi.serving not available, using basic WebSocket server")


# 辅助函数（占位实现，待后续实现）
def get_t(obs: Dict) -> float:
    """获取时间戳"""
    pass


def get_target_pose(obs: Dict) -> np.ndarray:
    """
    获取目标位姿 (7,) (x,y,z,rx,ry,rz,rw)
    
    通过点云聚类和颜色检测找到黄色目标物体的中心位置
    """
    # 假设从 obs 获取点云和RGB（已实现）
    # pcd: (N, 3) - 点云坐标
    # rgb: (N, 3) - RGB颜色值，范围 [0, 1] 或 [0, 255]
    pcd = obs.get("pcd")  # (N, 3)
    rgb = obs.get("rgb")  # (N, 3)
    
    if pcd is None or rgb is None or len(pcd) == 0:
        # 如果数据无效，返回默认位姿
        logging.warning("Invalid point cloud or RGB data in observation")
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    pcd = np.asarray(pcd)
    rgb = np.asarray(rgb)
    
    # 确保RGB值在 [0, 1] 范围内（如果输入是 [0, 255] 范围）
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # 1. 根据颜色筛选黄色点
    # 黄色在RGB空间中：高R、高G、低B
    # 使用HSV空间可能更准确，但这里用RGB简化
    yellow_mask = (
        (rgb[:, 0] > 0.7) &  # R > 0.7
        (rgb[:, 1] > 0.7) &  # G > 0.7
        (rgb[:, 2] < 0.3)    # B < 0.3
    )
    
    yellow_pcd = pcd[yellow_mask]
    
    if len(yellow_pcd) == 0:
        # 如果没有找到黄色点，返回默认位姿
        logging.warning("No yellow points found in point cloud")
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # 2. 对黄色点进行基于距离的聚类
    # 使用简单的阈值聚类方法
    cluster_eps = 0.05  # 5cm 聚类阈值
    yellow_cluster_pcd = _cluster_points_simple(yellow_pcd, eps=cluster_eps)
    
    if yellow_cluster_pcd is None or len(yellow_cluster_pcd) == 0:
        logging.warning("No valid yellow cluster found")
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # 3. 计算聚类中心（质心）
    center = np.mean(yellow_cluster_pcd, axis=0)
    
    # 4. 旋转设置为平行于基坐标系（单位四元数：无旋转）
    # 四元数格式 (rx, ry, rz, rw) = (0, 0, 0, 1) 表示无旋转
    rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # 5. 组合位置和旋转，返回 7 维位姿
    pose = np.concatenate([center, rotation])
    return pose.astype(np.float32)


def _cluster_points_simple(points: np.ndarray, eps: float = 0.05) -> Optional[np.ndarray]:
    """
    简单的基于距离的点云聚类（类似DBSCAN的简化版本）
    
    Args:
        points: 点云数组 (N, 3)
        eps: 聚类距离阈值
        
    Returns:
        最大聚类的点云 (M, 3)，如果没有找到有效聚类则返回None
    """
    if len(points) == 0:
        return None
    
    if len(points) == 1:
        return points
    
    # 使用简单的连通性聚类
    # 对于每个点，找到距离小于eps的所有点
    N = len(points)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    
    for i in range(N):
        if visited[i]:
            continue
        
        # 开始新的聚类
        cluster_indices = [i]
        visited[i] = True
        queue = [i]
        
        # BFS扩展聚类
        while queue:
            current_idx = queue.pop(0)
            current_point = points[current_idx]
            
            # 找到所有未访问且距离小于eps的点
            distances = np.linalg.norm(points - current_point, axis=1)
            neighbors = np.where((distances < eps) & (~visited))[0]
            
            for neighbor_idx in neighbors:
                cluster_indices.append(neighbor_idx)
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)
        
        if len(cluster_indices) > 0:
            clusters.append(points[cluster_indices])
    
    if len(clusters) == 0:
        return None
    
    # 返回最大的聚类（假设目标物体是最大的黄色聚类）
    largest_cluster = max(clusters, key=len)
    return largest_cluster


def get_gripper_pose(obs: Dict) -> np.ndarray:
    """获取夹爪位姿 (8,) [x,y,z,rx,ry,rz,rw,openness]"""
    pass


def get_gripper_openness(obs: Dict) -> float:
    """获取夹爪开合度"""
    pass


def get_finish_pose(obs: Dict) -> np.ndarray:
    """获取完成位姿 (7,) (x,y,z,rx,ry,rz,rw)"""
    pass


def get_target_velocity(target_state_list: list) -> np.ndarray:
    """
    通过一阶差值计算目标速度 (6,) (vx,vy,vz,wx,wy,wz)
    
    Args:
        target_state_list: 包含 {"t": float, "target_pose": np.ndarray} 的列表
        
    Returns:
        速度数组 (6,)
    """
    if len(target_state_list) < 2:
        # 如果数据点不足，返回零速度
        return np.zeros(6, dtype=np.float32)
    
    # 获取最后两个状态
    state_prev = target_state_list[-2]
    state_curr = target_state_list[-1]
    
    dt = state_curr["t"] - state_prev["t"]
    if dt <= 0:
        return np.zeros(6, dtype=np.float32)
    
    pose_prev = state_prev["target_pose"]  # (7,)
    pose_curr = state_curr["target_pose"]  # (7,)
    
    # 计算线速度 (vx, vy, vz)
    linear_velocity = (pose_curr[:3] - pose_prev[:3]) / dt
    
    # 计算角速度 (wx, wy, wz)
    # 这里简化处理：直接从四元数的差值估算角速度
    # 实际应用中可能需要更精确的四元数差值计算方法
    angular_velocity = (pose_curr[3:7] - pose_prev[3:7]) / dt
    
    velocity = np.concatenate([linear_velocity, angular_velocity[:3]])
    return velocity.astype(np.float32)


class PolicyServer:
    """Policy Server类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化Policy Server
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        server_config = get_server_config(config_path)
        self.host = server_config.get("host", "0.0.0.0")
        self.port = server_config.get("port", 8000)
        self.api_key = server_config.get("api_key")
        self.policy_config = server_config.get("policy", {})
        self.model_type = self.policy_config.get("model_type", "default")
        
        # 初始化policy model和world model（框架阶段）
        self.policy_model = None
        self.world_model = None
        
        # 框架阶段：计数器，用于决定是否返回action（仅用于演示）
        self._request_count = 0
        
        # 状态机相关变量
        self.target_state_list = []
        self.current_state = "reach"  # 初始状态为 "reach"
        
        logging.info(f"Policy server initialized (model_type={self.model_type}, host={self.host}, port={self.port})")
    
    def handle_request(self, obs: Dict) -> Dict:
        """
        处理请求，返回响应
        
        Args:
            obs: observation字典
            
        Returns:
            响应字典，包含type字段（"update_world_model"或"action"）
        """
        if self.model_type == "3dafdp":
            # 3dafdp模型：先更新world model，然后决定是否返回action
            self._update_world_model(obs)
            
            # 根据内部逻辑决定是否返回action（框架阶段使用简单逻辑）
            if self._should_return_action():
                action = self._policy_model_infer(obs)
                return {
                    "type": "action",
                    "joint_angles": action
                }
            else:
                return {
                    "type": "update_world_model",
                    "status": "ok"
                }
        else:
            # 标准模型：直接返回action
            action = self._policy_model_infer(obs)
            return {
                "type": "action",
                "joint_angles": action
            }
    
    def _update_world_model(self, obs: Dict) -> None:
        """
        更新World Model（框架阶段）
        
        Args:
            obs: observation字典
        """
        # 框架阶段：占位函数
        # 实际实现时需要调用world_model.update()
        logging.debug("Updating world model")
        pass
    
    def _policy_model_infer(self, obs: Dict) -> np.ndarray:
        """
        Policy模型推理（框架阶段）
        
        Args:
            obs: observation字典
            
        Returns:
            action数组 (8,) [x,y,z,rx,ry,rz,rw,openness]
        """
        # 获取观测数据
        t = get_t(obs)
        target_pose = get_target_pose(obs)  # (7,) (x,y,z,rx,ry,rz,rw) in base frame
        self.target_state_list.append({"t": t, "target_pose": target_pose})
        target_velocity = get_target_velocity(self.target_state_list)  # (6,) (vx,vy,vz,wx,wy,wz) in base frame
        gripper_pose = get_gripper_pose(obs)  # (8,) [x,y,z,rx,ry,rz,rw,openness] in base frame
        gripper_openness = get_gripper_openness(obs)  # float: 0=open, 1=close
        finish_pose = get_finish_pose(obs)  # (7,) (x,y,z,rx,ry,rz,rw) in base frame
        
        # 常量定义
        OPEN = 0.0  # 打开
        CLOSE = 1.0  # 关闭
        DISTANCE_THRESHOLD = 0.1  # 10 cm = 0.1 m
        
        # 计算夹爪到目标的距离（仅考虑位置，不考虑旋转）
        gripper_pos = gripper_pose[:3]
        target_pos = target_pose[:3]
        d_gripper_target = np.linalg.norm(gripper_pos - target_pos)
        
        # 判断夹爪状态（open = 0, close = 1）
        is_gripper_open = (gripper_openness < 0.5)
        is_gripper_closed = (gripper_openness >= 0.5)
        
        # 状态转换逻辑
        # state: "reach" -> "grasp" -> "pick" -> "place" -> "reach"
        if self.current_state == "reach":
            if d_gripper_target < DISTANCE_THRESHOLD and is_gripper_open:
                self.current_state = "grasp"
        elif self.current_state == "grasp":
            if is_gripper_closed:
                self.current_state = "pick"
        elif self.current_state == "pick":
            if is_gripper_closed:  # 保持closed状态，执行pick动作
                self.current_state = "place"
        elif self.current_state == "place":
            if is_gripper_open:
                self.current_state = "reach"
        
        # 根据当前状态生成动作
        action = gripper_pose.copy()  # 复制当前夹爪位姿作为基础
        
        if self.current_state == "reach":
            # 移动到目标位置，保持夹爪打开
            action[:3] = target_pose[:3]  # 设置位置
            action[3:7] = target_pose[3:7]  # 设置旋转（可选，也可以保持当前旋转）
            action[-1] = OPEN
        elif self.current_state == "grasp":
            # 移动到目标位置，关闭夹爪
            action[:3] = target_pose[:3]  # 设置位置
            action[3:7] = target_pose[3:7]  # 设置旋转
            action[-1] = CLOSE
        elif self.current_state == "pick":
            # 提升夹爪（z轴方向增加0.2m），保持关闭
            action[2] += 0.2  # z轴提升0.2m
            action[-1] = CLOSE
        elif self.current_state == "place":
            # 移动到完成位置，打开夹爪
            action[:3] = finish_pose[:3]  # 设置位置
            action[3:7] = finish_pose[3:7]  # 设置旋转
            action[-1] = OPEN
        
        return action
    
    def _should_return_action(self) -> bool:
        """
        判断是否应该返回action（框架阶段）
        
        Returns:
            是否返回action
        """
        # 框架阶段：简单逻辑，每3次请求返回一次action
        # 实际实现时需要根据world model的状态决定
        self._request_count += 1
        return self._request_count % 3 == 0
    
    def serve(self):
        """启动服务器"""
        if HAS_OPENPI:
            # 使用openpi的WebSocket服务器（需要实现Policy接口）
            # 框架阶段使用基本实现
            logging.info("Using openpi WebSocket server (framework mode)")
            # 这里需要实现BasePolicy接口
            # 暂时使用基本实现
            self._serve_basic()
        else:
            # 使用基本WebSocket服务器
            self._serve_basic()
    
    def _serve_basic(self):
        """
        基本WebSocket服务器实现（框架阶段）
        
        注意：实际部署时应该使用openpi的WebSocket服务器
        """
        try:
            import websockets
            import asyncio
            
            async def handler(websocket, path):
                logging.info(f"Client connected: {websocket.remote_address}")
                try:
                    async for message in websocket:
                        # 解析消息（框架阶段，实际需要使用msgpack_numpy）
                        try:
                            import json
                            obs = json.loads(message)
                        except Exception:
                            # 如果是二进制消息，需要msgpack解析
                            logging.error("Binary message parsing not implemented in framework")
                            continue
                        
                        # 处理请求
                        response = self.handle_request(obs)
                        
                        # 发送响应（框架阶段使用JSON）
                        try:
                            response_json = json.dumps(response, default=str)
                            await websocket.send(response_json)
                        except Exception as e:
                            logging.error(f"Error sending response: {e}")
                except websockets.exceptions.ConnectionClosed:
                    logging.info("Client disconnected")
                except Exception as e:
                    logging.error(f"Error in handler: {e}")
            
            async def main():
                async with websockets.serve(handler, self.host, self.port):
                    logging.info(f"Policy server started on ws://{self.host}:{self.port}")
                    await asyncio.Future()  # 永远运行
            
            asyncio.run(main())
        except ImportError:
            logging.error("websockets not available. Please install: pip install websockets")
            raise
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            raise


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, force=True)
    
    try:
        server = PolicyServer()
        server.serve()
    except KeyboardInterrupt:
        logging.info("Server stopped")
    except Exception as e:
        logging.error(f"Server error: {e}")
        raise


if __name__ == '__main__':
    main()