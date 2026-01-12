import time
import logging
import sys
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple
from openpi_client import base_policy as _base_policy
import open3d as o3d

VISUALZATION = False

# 配置日志：同时输出到终端和文件
def setup_logging(log_file: str = "log.txt", log_level=logging.INFO):
    """
    配置logging同时输出到终端和文件
    同时重定向print输出到文件
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
    """
    # 创建formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件handler（追加模式）
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 移除已有的handlers（避免重复）
    root_logger.handlers.clear()
    
    # 添加handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 重定向print到文件（让print也保存到文件）
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                if hasattr(f, 'flush'):
                    f.flush()
    
    # 保存原始的stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 打开日志文件用于print输出（追加模式）
    log_file_obj = open(log_file, 'a', encoding='utf-8')
    
    # 将stdout和stderr同时输出到终端和文件
    sys.stdout = TeeOutput(original_stdout, log_file_obj)
    sys.stderr = TeeOutput(original_stderr, log_file_obj)
    
    logging.info(f"Logging initialized. Logs will be saved to {log_file}")

# 辅助函数（占位实现，待后续实现）
def get_t(obs: Dict) -> float:
    """获取时间戳"""
    pass


def get_target_pose(obs: Dict) -> np.ndarray:
    """
    获取目标位姿 (7,) (x,y,z,rx,ry,rz,rw)
    
    通过点云聚类和颜色检测找到黄色目标物体的中心位置
    """
    VISUALZATION = True
    suffix = int(time.time() * 1000)
    # 假设从 obs 获取点云和RGB（已实现）
    # pcd: (N, 3) - 点云坐标
    # rgb: (N, 3) - RGB颜色值，范围 [0, 1] 或 [0, 255]
    pcd = obs.get("pcds")  # TODO: -> (N, 3)
    rgb = obs.get("rgbs")  # TODO: -> (N, 3)
    
    if pcd is None or rgb is None or len(pcd) == 0:
        # 如果数据无效，返回默认位姿
        logging.warning("Invalid point cloud or RGB data in observation")
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    pcd = np.asarray(pcd)
    rgb = np.asarray(rgb)
    logging.info(f"pcd{pcd.shape}, rgb{rgb.shape}")
    # save_pcd_with_rgb_to_ply(pcd.reshape(-1, 3), rgb.reshape(-1, 3), f"pcd_{suffix}.ply")
    # 确保RGB值在 [0, 1] 范围内（如果输入是 [0, 255] 范围）
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # 1. 根据颜色筛选黄色点
    # 黄色在RGB空间中：高R、高G、低B
    # 使用HSV空间可能更准确，但这里用RGB简化
    # yellow_mask = (
    #     (rgb[:, 0] > 0.7) &  # R > 0.7
    #     (rgb[:, 1] > 0.7) &  # G > 0.7
    #     (rgb[:, 2] < 0.3)    # B < 0.3
    # )
    yellow_mask = (
        (rgb[..., 0] > 0.7) &
        (rgb[..., 1] > 0.7) &
        (rgb[..., 2] < 0.3)
    )

    
    # yellow_pcd = pcd[yellow_mask]
    valid = np.isfinite(pcd).all(axis=-1) & (np.linalg.norm(pcd, axis=-1) > 1e-6)
    yellow_mask &= valid

    yellow_pcd = pcd[yellow_mask]
    yellow_rgb = rgb[yellow_mask]
    print(yellow_pcd.shape)
    logging.info(yellow_pcd.shape, yellow_rgb.shape)
    # visualize yellow_pcd and save into ply
    if VISUALZATION and len(yellow_pcd) > 0:
        save_pcd_with_rgb_to_ply(yellow_pcd, yellow_rgb, f"yellow_pcd_{suffix}.ply")
    if len(yellow_pcd) == 0:
        # 如果没有找到黄色点，返回默认位姿
        logging.warning("No yellow points found in point cloud")
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # 2. 对黄色点进行基于距离的聚类
    # 使用简单的阈值聚类方法
    cluster_eps = 0.05  # 5cm 聚类阈值
    yellow_cluster_pcd, cluster_indices = _cluster_points_simple(yellow_pcd, eps=cluster_eps)
    print(yellow_cluster_pcd.shape, cluster_indices.shape)
    # visualize yellow_cluster_pcd and save into ply
    if VISUALZATION and yellow_cluster_pcd is not None and len(yellow_cluster_pcd) > 0:
        yellow_cluster_rgb = yellow_rgb[cluster_indices]
        save_pcd_with_rgb_to_ply(yellow_cluster_pcd, yellow_cluster_rgb, f"yellow_cluster_{suffix}.ply")

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

    # 可视化pose为三个坐标轴
    if VISUALZATION:
        visualize_pose_as_axes(pose, axis_length=0.05, filename=f"pose_axes_{suffix}.ply")
    
    return pose.astype(np.float32)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    Args:
        quaternion: 四元数 (rx, ry, rz, rw) 或 (w, x, y, z)
        
    Returns:
        3x3旋转矩阵
    """
    # 假设输入是 (rx, ry, rz, rw) 格式
    if len(quaternion) == 4:
        x, y, z, w = quaternion
    else:
        raise ValueError(f"Invalid quaternion length: {len(quaternion)}")
    
    # 归一化四元数
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-8:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def visualize_pose_as_axes(pose: np.ndarray, axis_length: float = 0.1, filename: str = None) -> str:
    """
    将pose可视化为三个坐标轴（X轴红色，Y轴绿色，Z轴蓝色）
    
    Args:
        pose: 位姿 (7,) (x, y, z, rx, ry, rz, rw)
        axis_length: 坐标轴长度（米）
        filename: 保存的文件名，如果为None则自动生成
        
    Returns:
        保存的文件路径
    """
    if len(pose) != 7:
        logging.warning(f"Invalid pose length: {len(pose)}, expected 7")
        return ""
    
    # 提取位置和旋转
    position = pose[:3]  # (x, y, z)
    quaternion = pose[3:7]  # (rx, ry, rz, rw)
    
    # 转换为旋转矩阵
    R = quaternion_to_rotation_matrix(quaternion)
    
    # 创建三个坐标轴的方向向量（在局部坐标系中）
    # X轴：红色，Y轴：绿色，Z轴：蓝色
    axes_local = np.array([
        [axis_length, 0, 0],  # X轴
        [0, axis_length, 0],  # Y轴
        [0, 0, axis_length]   # Z轴
    ])
    
    # 应用旋转矩阵
    axes_world = (R @ axes_local.T).T
    
    # 平移到pose位置
    axes_points = axes_world + position
    
    # 创建点云：原点 + 三个轴端点
    points = np.vstack([position, axes_points])
    
    # 创建颜色：原点白色，X轴红色，Y轴绿色，Z轴蓝色
    colors = np.array([
        [1.0, 1.0, 1.0],  # 原点：白色
        [1.0, 0.0, 0.0],  # X轴：红色
        [0.0, 1.0, 0.0],  # Y轴：绿色
        [0.0, 0.0, 1.0]   # Z轴：蓝色
    ])
    
    # 为了更好的可视化，在轴上添加更多点
    num_points_per_axis = 20
    all_points = [position]  # 原点
    all_colors = [[1.0, 1.0, 1.0]]  # 原点颜色
    
    for i, (axis_dir, color) in enumerate(zip(axes_world, colors[1:])):
        # 在轴上均匀采样点
        t_values = np.linspace(0, 1, num_points_per_axis)
        axis_points = position[None, :] + axis_dir[None, :] * t_values[:, None]
        all_points.append(axis_points)
        all_colors.append(np.tile(color, (num_points_per_axis, 1)))
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    # 确保数据类型正确
    points = np.ascontiguousarray(points.astype(np.float64))
    colors = np.ascontiguousarray(colors.astype(np.float64))
    
    # 创建open3d点云对象
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    
    # 生成文件名
    if filename is None:
        filename = f"pose_axes_{int(time.time() * 1000)}.ply"
    
    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, pcd_o3d)
    logging.info(f"Saved pose axes visualization to {filename}")
    return filename


def _cluster_points_simple(points: np.ndarray, eps: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    简单的基于距离的点云聚类（类似DBSCAN的简化版本）
    
    Args:
        points: 点云数组 (N, 3)
        eps: 聚类距离阈值
        
    Returns:
        (最大聚类的点云 (M, 3), 对应的索引 (M,))，如果没有找到有效聚类则返回 (None, None)
    """
    if len(points) == 0:
        return None, None
    
    if len(points) == 1:
        return points, np.array([0])
    
    # 使用简单的连通性聚类
    # 对于每个点，找到距离小于eps的所有点
    N = len(points)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    cluster_indices_list = []
    
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
            cluster_indices_list.append(np.array(cluster_indices))
    
    if len(clusters) == 0:
        return None, None
    
    # 返回最大的聚类（假设目标物体是最大的黄色聚类）
    largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
    largest_cluster = clusters[largest_idx]
    largest_indices = cluster_indices_list[largest_idx]
    return largest_cluster, largest_indices


def save_pcd_with_rgb_to_ply(pcd: np.ndarray, rgb: np.ndarray, filename: str = None) -> str:
    """
    将点云和颜色信息保存为PLY文件
    
    Args:
        pcd: 点云数组 (N, 3)
        rgb: RGB颜色数组 (N, 3)，范围 [0, 1]
        filename: 保存的文件名，如果为None则自动生成
        
    Returns:
        保存的文件路径
    """
    if pcd is None or len(pcd) == 0:
        logging.warning("Empty point cloud, cannot save")
        return ""
    
    if len(pcd) != len(rgb):
        logging.warning(f"Point cloud and RGB size mismatch: {len(pcd)} vs {len(rgb)}")
        # 如果大小不匹配，只使用较小的长度
        min_len = min(len(pcd), len(rgb))
        pcd = pcd[:min_len]
        rgb = rgb[:min_len]
    
    # 确保RGB值在 [0, 1] 范围内
    rgb = np.asarray(rgb)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # 创建open3d点云对象
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb)
    
    # 生成文件名
    if filename is None:
        filename = f"pointcloud_{int(time.time() * 1000)}.ply"
    
    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, pcd_o3d)
    logging.info(f"Saved point cloud with RGB to {filename}")
    return filename


def get_gripper_pose(obs: Dict) -> np.ndarray:
    """获取夹爪位姿 (8,) [x,y,z,rx,ry,rz,rw]"""
    # TODO:
    return obs['curr_gripper_history'][-1][:7]


def get_gripper_openness(obs: Dict) -> float:
    """获取夹爪开合度"""
        # TODO:
    return obs['curr_gripper_history'][-1][-1]


def get_finish_pose(obs: Dict) -> np.ndarray:
    """获取完成位姿 (7,) (x,y,z,rx,ry,rz,rw)"""
    return np.array([ 0.3903003, -0.30904002, 0.86467913, 0.0052456, 0.00635652,  0.705191, 0.708969])


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

def to_numpy_tree(x):
    """递归：torch -> numpy；容器保持结构；其余类型原样保留（int/float/str/bool/None等）"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        return {k: to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [to_numpy_tree(v) for v in x]
        return type(x)(y)
    return x

class ExpertPolicy(_base_policy.BasePolicy):
    
    def __init__(self, camera_config: Optional[Dict] = None, log_file: str = "log.txt", enable_logging: bool = True):
        """
        初始化ExpertPolicy
        
        Args:
            camera_config: 相机配置
            log_file: 日志文件路径，如果为None则不保存到文件
            enable_logging: 是否启用日志保存到文件
        """
        self._camera_config = camera_config
        self.target_state_list = []
        self.current_state = "reach"  # 初始状态为 "reach"
        
        # 如果启用日志，设置同时输出到终端和文件
        if enable_logging and log_file:
            setup_logging(log_file, logging.INFO)
        
        logging.info("ExpertPolicy initialized")

    @staticmethod
    def obs_to_tensor(
        obs: Dict[str, Any],
        device: str = "cuda",
        non_blocking: bool = True,
    ) -> Dict[str, Any]:
        dev = torch.device(device)
        out: Dict[str, Any] = {}

        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                vv = v.copy()
                if k == "rgbs":
                    # uint8 -> float32 (可选归一化)
                    vv = v.astype(np.float32, copy=False)
                    t = torch.from_numpy(vv).to(dev, non_blocking=non_blocking)
                    out[k] = t

                elif k == "pcds":
                    # 关键：uint16 不能直接 from_numpy -> 先转 float32
                    vv = v.astype(np.float32)  # 这里会 copy，但不丢精度
                    out[k] = torch.from_numpy(vv).to(dev, non_blocking=non_blocking)

                else:
                    out[k] = torch.from_numpy(v).to(dev, non_blocking=non_blocking)

            elif torch.is_tensor(v):
                out[k] = v.to(dev, non_blocking=non_blocking)
            else:
                out[k] = v

        return out


    # ---------- 2) torch 版 cache（一次算好放 GPU）----------
    def _prepare_cam_cache_torch(self, H: int, W: int, device: torch.device):
        # 像素网格 (1,H,W)
        # 注意 meshgrid 的 indexing，xy 对应 (W,H) 的习惯
        uu, vv = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing="xy",
        )
        self._uu = uu[None, ...]  # (1,H,W)
        self._vv = vv[None, ...]  # (1,H,W)

        K = torch.as_tensor(self._camera_config["intrinsics"], device=device, dtype=torch.float32)  # (C,3,3)
        self._fx = K[:, 0, 0][:, None, None]
        self._fy = K[:, 1, 1][:, None, None]
        self._cx = K[:, 0, 2][:, None, None]
        self._cy = K[:, 1, 2][:, None, None]
        self._inv_fx = 1.0 / self._fx
        self._inv_fy = 1.0 / self._fy

        T = torch.as_tensor(self._camera_config["extrinsics"], device=device, dtype=torch.float32)  # (C,4,4)
        self._R = T[:, :3, :3]  # (C,3,3)
        self._t = T[:, :3, 3]   # (C,3)

    # ---------- 3) torch 版 depth(mm)->点云(world) ----------
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        期望：obs['pcds'] 是 depth，shape=(C,H,W)，单位 mm，且已经是 CUDA tensor。
        输出：obs['pcds'] 变为 world 点云，shape=(C,H,W,3)，float32 CUDA tensor
        """

        obs = self.obs_to_tensor(obs)

        Z = obs["pcds"]

        if not torch.is_tensor(Z):
            raise TypeError("obs['pcds'] must be a torch.Tensor. Call obs_to_tensor first.")
        if Z.ndim != 3:
            raise ValueError(f"Expected depth shape (C,H,W), got {tuple(Z.shape)}")

        device = Z.device

        # depth: mm -> m
        Z = Z.to(dtype=torch.float32).mul_(0.001)

        C, H, W = Z.shape

        # 第一次 or 分辨率变化：建 cache（在同一个 device 上）
        if (not hasattr(self, "_uu")) or (self._uu.shape[-2:] != (H, W)) or (self._uu.device != device):
            self._prepare_cam_cache_torch(H, W, device)

        valid = Z > 0.0

        X = (self._uu - self._cx) * Z * self._inv_fx   # (C,H,W)
        Y = (self._vv - self._cy) * Z * self._inv_fy   # (C,H,W)

        # (C,HW,3)
        P = torch.stack([X, Y, Z], dim=-1).reshape(C, -1, 3)

        # Pw = P @ R^T + t
        Pw = torch.bmm(P, self._R.transpose(1, 2)) + self._t[:, None, :]
        points_world = Pw.reshape(C, H, W, 3)

        # 清零无效点（更快的写法：乘 mask）
        points_world = points_world * valid[..., None].to(points_world.dtype)

        obs["pcds"] = points_world
        return obs

    
    
    def infer(self, obs: Dict) -> Dict:
        """
        假的推理方法，返回零数组作为占位符
        
        Args:
            obs: observation字典
            
        Returns:
            包含joint_angles的字典
        """

        obs = self.preprocess_obs(obs)
        obs = to_numpy_tree(obs)

        # 保存obs到本地npy文件
        if VISUALZATION:
            import os
            os.makedirs("./src/temp", exist_ok=True)
            np.save("./src/temp/obs.npy", obs)
        
        # 获取观测数据
        # t = get_t(obs)
        t = time.time()
        target_pose = get_target_pose(obs)  # (7,) (x,y,z,rx,ry,rz,rw) in base frame
        self.target_state_list.append({"t": t, "target_pose": target_pose})
        target_velocity = get_target_velocity(self.target_state_list)  # (6,) (vx,vy,vz,wx,wy,wz) in base frame
        gripper_pose = get_gripper_pose(obs)  # (8,) [x,y,z,rx,ry,rz,rw,openness] in base frame
        gripper_openness = get_gripper_openness(obs)  # float: 0=open, 1=close
        # finish_pose = get_finish_pose(obs)  # (7,) (x,y,z,rx,ry,rz,rw) in base frame
        # finish_pose = gripper_pose
        finish_pose = get_finish_pose(obs)
        
        # 常量定义
        OPEN = 0.0  # 打开
        CLOSE = 1.0  # 关闭
        DISTANCE_THRESHOLD = 0.20  # 10 cm = 0.1 m
        
        # 计算夹爪到目标的距离（仅考虑位置，不考虑旋转）
        gripper_pos = gripper_pose[:3]
        target_pos = target_pose[:3]
        d_gripper_target = np.linalg.norm(gripper_pos - target_pos)
        
        # 判断夹爪状态（open = 0, close = 1）
        is_gripper_open = (gripper_openness < 25)
        is_gripper_closed = (gripper_openness >= 25)
        print("curr_state", self.current_state)
        print("d_gripper_target", d_gripper_target)
        print("is_gripper_open", is_gripper_open)
        print("is_gripper_closed", is_gripper_closed)
        print("DISTANCE_THRESHOLD", DISTANCE_THRESHOLD)
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
        print("after")
        print(self.current_state)
        
        # 根据当前状态生成动作
        # action = gripper_pose.copy()  # 复制当前夹爪位姿作为基础

        pose7 = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)

        action = np.zeros(8, dtype=np.float32)
        action[:7] = pose7          # x y z qx qy qz qw
        action[7]  = OPEN           # gripper
        
        if self.current_state == "reach":
            # 移动到目标位置，保持夹爪打开
            action[:3] = target_pose[:3]  # 设置位置
            action[2] += 0.15
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

        action[3:7] = [0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 0.41908594192841403]
        logging.info(action)
        time.sleep(2)
        
        # return action
        # 返回假的action（7个关节的零数组）
        # num_joints = 7
        # action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 0.41908594192841403, 43.30252697496553])
        return {
            # "joint_angles": np.zeros(num_joints, dtype=np.float32)
            "action": action
        }
    
    def reset(self) -> None:
        """重置policy状态"""
        pass