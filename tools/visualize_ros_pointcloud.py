"""从ROS话题订阅数据并保存为.npy文件

订阅 /sensor_sync_package 话题，获取RGB和Depth图像，
转换为点云图像并保存为.npy文件。
"""

import json
import cv2
import numpy as np
from pathlib import Path
import rospy
import sys
import threading
from typing import Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from astribot_msgs.msg import SyncPackage, GripperState
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# =========================
# 工具函数
# =========================

def get_rot_mat(pose):
    """从位姿（x,y,z,qx,qy,qz,qw）获取旋转矩阵"""
    from scipy.spatial.transform import Rotation as R
    x, y, z, qx, qy, qz, qw = pose
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def to_cam_hwc3(x):
    """
    把 x 统一转成 (C, H, W, 3) 的 numpy
    支持:
      - torch.Size([1, C, 3, H, W])  (channel-first)
      - torch.Size([1, C, H, W, 3])  (channel-last)
      - torch.Size([C, 3, H, W])     (channel-first)
      - torch.Size([C, H, W, 3])     (channel-last)
    """
    # try:
    #     import torch
    #     if torch.is_tensor(x):
    #         x = x.detach().cpu()
    # except Exception:
    #     pass

    x = np.asarray(x)

    if x.ndim == 5:
        x = x[0]  # squeeze batch

    if x.ndim != 4:
        raise ValueError(f"Expected 4D after squeeze batch, got {x.shape}")

    # channel-first -> channel-last
    if x.shape[1] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (0, 2, 3, 1))
    elif x.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Cannot infer channel dim from shape {x.shape}")

    return x

def load_camera_json(intrinsics_json_path, extrinsics_json_path):
    """
    读取:
      - camera_intrinsics.json
      - camera_to_board_transforms.json (含 T_camera_to_base)

    返回：
        cameras[cam_name] = {
            "K": (3,3),
            "dist": (...),
            "T_camera_to_base": (4,4)
        }
    """
    with open(intrinsics_json_path, "r") as f:
        intrinsics_cfg = json.load(f)

    with open(extrinsics_json_path, "r") as f:
        extrinsics_cfg = json.load(f)

    cameras = {}
    for cam_name, intr_cfg in intrinsics_cfg.items():
        if cam_name not in extrinsics_cfg:
            print(f"⚠️ {cam_name} not found in extrinsics json, skip")
            continue

        ext_cfg = extrinsics_cfg[cam_name]
        K = np.array(intr_cfg["camera_matrix"], dtype=np.float64)
        dist = np.array(intr_cfg["distortion_coefficients"], dtype=np.float64)
        T_camera_to_base = np.array(ext_cfg["T_camera_to_base"], dtype=np.float64)

        cameras[cam_name] = {
            "K": K,
            "dist": dist,
            "T_camera_to_base": T_camera_to_base,
        }
    return cameras

def depth_to_pointcloud_image(
    depth_img: np.ndarray,
    K: np.ndarray,
    T_c2base: np.ndarray,
    max_depth: float = 2.0,
):
    """
    将深度图转换为点云图像（保持图像尺寸）
    
    输入:
      - depth_img: (H, W) 深度图，单位：米（float32）
      - K: (3, 3) 相机内参矩阵
      - T_c2base: (4, 4) 相机到base的变换矩阵
      - max_depth: 最大深度（米）
    
    输出:
      - pcd_img: (H, W, 3) base frame下的点云图像，每个像素对应一个3D点坐标
    """
    H, W = depth_img.shape
    
    # 深度已经是米为单位
    z = depth_img.astype(np.float32)
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    
    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    # 转换为齐次坐标并变换到base frame
    xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=2)  # (H, W, 3)
    xyz_cam_flat = xyz_cam.reshape(-1, 3)  # (H*W, 3)
    xyz_hom = np.hstack([xyz_cam_flat, np.ones((xyz_cam_flat.shape[0], 1))])
    xyz_base_flat = (T_c2base @ xyz_hom.T).T[:, :3]  # (H*W, 3)
    xyz_base = xyz_base_flat.reshape(H, W, 3)  # (H, W, 3)
    
    # 无效深度点设置为0
    mask = (z > 0) & (z < max_depth) & np.isfinite(z)
    xyz_base[~mask] = 0.0
    
    return xyz_base

def depth_to_pointcloud_base(
    depth_img: np.ndarray,
    rgb_img: np.ndarray,
    K: np.ndarray,
    T_c2base: np.ndarray,
    max_depth: float = 2.0,
):
    """
    将深度图转换为base frame下的点云（稀疏点云，用于可视化）
    
    输入:
      - depth_img: (H, W) 深度图，单位：米（float32）
      - rgb_img: (H, W, 3) RGB图像
      - K: (3, 3) 相机内参矩阵
      - T_c2base: (4, 4) 相机到base的变换矩阵
      - max_depth: 最大深度（米）
    
    输出:
      - xyz: (N, 3) base frame下的点云
      - colors: (N, 3) [0,1] 归一化的颜色
    """
    if rgb_img.shape[:2] != depth_img.shape[:2]:
        rgb_img = cv2.resize(
            rgb_img,
            (depth_img.shape[1], depth_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    H, W = depth_img.shape
    
    # 深度已经是米为单位
    z = depth_img.astype(np.float32)
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = z.flatten()
    
    # 过滤有效深度
    mask = (z > 0) & (z < max_depth) & np.isfinite(z)
    u = u[mask]
    v = v[mask]
    z = z[mask]
    
    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    # 转换为齐次坐标并变换到base frame
    xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    xyz_hom = np.hstack([xyz_cam, np.ones((xyz_cam.shape[0], 1))])
    xyz_base = (T_c2base @ xyz_hom.T).T[:, :3]
    
    # 提取颜色
    rgb_flat = rgb_img.reshape(-1, 3)[mask]
    colors = rgb_flat.astype(np.float32)
    
    # 颜色归一化到 [0,1]
    if colors.max() > 1.5:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    
    return xyz_base, colors



class ROSPointCloudVisualizer:
    """从ROS话题订阅数据并保存为.npy文件"""
    
    def __init__(
        self,
        sync_topic: str = "/sensor_sync_package",
        intrinsics_json: Optional[Path] = None,
        extrinsics_json: Optional[Path] = None,
        data_root: Optional[Path] = None,
        cam_names: Optional[list] = None,
        depth_scale: float = 1000.0,
        max_depth: float = 2.0,
    ):
        """
        初始化ROS点云可视化器
        
        Args:
            sync_topic: 同步数据话题
            intrinsics_json: 相机内参JSON文件路径
            extrinsics_json: 相机外参JSON文件路径
            data_root: 数据根目录（如果提供了，会在data_root下查找相机标定文件）
            cam_names: 相机名称列表，顺序对应SyncPackage中的相机顺序
            depth_scale: 深度缩放因子
            max_depth: 最大深度（米）
        """
        # ROS初始化
        rospy.init_node('ros_pointcloud_visualizer', anonymous=True)
        self.bridge = CvBridge()
        self.sync_topic = sync_topic
        
        # 相机标定文件路径
        if data_root is not None:
            data_root = Path(data_root)
            if intrinsics_json is None:
                intrinsics_json = data_root / "camera_intrinsics.json"
            if extrinsics_json is None:
                extrinsics_json = data_root / "camera_to_board_transforms.json"
        
        # 加载相机标定参数
        if intrinsics_json is not None and extrinsics_json is not None:
            intrinsics_json = Path(intrinsics_json)
            extrinsics_json = Path(extrinsics_json)
            if intrinsics_json.exists() and extrinsics_json.exists():
                self.cameras = load_camera_json(intrinsics_json, extrinsics_json)
                print(f"✓ 加载相机标定参数: {len(self.cameras)} 个相机")
            else:
                print(f"⚠️ 相机标定文件不存在，将使用默认参数")
                self.cameras = {}
        else:
            self.cameras = {}
        
        # 相机名称（默认顺序）
        self.cam_names = cam_names or ["camera_0", "camera_1", "camera_2", "camera_3"]
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        
        # 数据存储
        self.latest_sync_data = None
        self.lock = threading.Lock()
        
        # 订阅话题
        self.sub = rospy.Subscriber(self.sync_topic, SyncPackage, self._sync_callback)
        
        print(f"✓ 订阅话题: {self.sync_topic}")
        print(f"✓ 相机名称顺序: {self.cam_names}")
        
        # 等待订阅者建立连接（给ROS时间处理连接）
        rospy.sleep(0.1)
    
    def _sync_callback(self, msg: SyncPackage):
        """同步数据回调函数"""
        print("✓ 回调函数被调用，接收到同步数据")
        with self.lock:
            self.latest_sync_data = msg
    
    def _process_sync_data(self, msg: SyncPackage, return_image_format: bool = False) -> Optional[tuple]:
        """
        处理同步数据，转换为点云
        
        Args:
            msg: SyncPackage消息
            return_image_format: 如果True，返回图像格式的点云（用于保存）；如果False，返回稀疏点云（用于可视化）
        
        返回:
            如果return_image_format=False: (all_xyz, all_rgb, eff) 或 None
            如果return_image_format=True: (rgb_obs, pcd_obs, curr_gripper) 或 None
        """
        try:
            print("  提取RGB图像...")
            # 提取RGB图像
            rgb_images = []
            for rgb_msg in msg.rgb_list:
                if isinstance(rgb_msg, CompressedImage):
                    # 解压压缩图像
                    np_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8)
                    rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                else:
                    # 普通Image消息
                    rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
                rgb_images.append(rgb_img)
            
            print(f"  提取了 {len(rgb_images)} 个RGB图像")
            # 提取Depth图像（统一转换为米为单位的float32）
            depth_images = []
            for depth_msg in msg.depth_list:
                if depth_msg.encoding == "16UC1":
                    # 16位深度图（单位：毫米），转换为米
                    depth_img = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                        depth_msg.height, depth_msg.width
                    ).astype(np.float32) / self.depth_scale
                elif depth_msg.encoding == "32FC1":
                    # 32位浮点深度图（单位：米）
                    depth_img = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
                        depth_msg.height, depth_msg.width
                    )
                else:
                    # 使用cv_bridge转换，假设是16UC1
                    depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                    depth_img = depth_img.astype(np.float32) / self.depth_scale
                depth_images.append(depth_img)
            
            if len(rgb_images) != len(depth_images):
                print(f"⚠️ RGB图像数量({len(rgb_images)}) != Depth图像数量({len(depth_images)})")
                return None
            
            # 提取gripper数据
            curr_gripper = None
            eff = None
            if hasattr(msg, 'gripper_hist') and msg.gripper_hist is not None:
                gripper_hist = np.array(msg.gripper_hist.data, dtype=np.float32)
                # gripper_hist形状是(24,)，包含3帧×8个元素
                # 格式：[frame_20_ago, frame_10_ago, current_frame]
                # 每帧8个元素：[x, y, z, qx, qy, qz, qw, gripper_state]
                if gripper_hist.size >= 24:
                    curr_gripper = gripper_hist.reshape(3, 8)  # (3, 8) 3帧×8个元素
                    # 取当前gripper位置（最后一帧的前3个元素）
                    current_gripper = gripper_hist[-8:]  # 最后一帧的8个元素
                    eff = current_gripper[:3]  # 位置（x, y, z）
            
            print(f"  提取了 {len(depth_images)} 个深度图像")
            if return_image_format:
                # 返回图像格式（用于保存）
                print("  开始转换为点云图像格式...")
                rgb_obs_list = []
                pcd_obs_list = []
                
                for i, cam_name in enumerate(self.cam_names):
                    print(f"    处理相机 {cam_name} ({i+1}/{len(self.cam_names)})...")
                    if i >= len(rgb_images):
                        break
                    
                    if cam_name in self.cameras:
                        K = self.cameras[cam_name]["K"]
                        T_c2base = self.cameras[cam_name]["T_camera_to_base"]
                    else:
                        print(f"⚠️ {cam_name} 的标定参数不存在，跳过")
                        continue
                    
                    # RGB图像
                    rgb_img = rgb_images[i].astype(np.float32)
                    rgb_obs_list.append(rgb_img)
                    
                    # 点云图像
                    print(f"      转换点云图像...")
                    pcd_img = depth_to_pointcloud_image(
                        depth_img=depth_images[i],
                        K=K,
                        T_c2base=T_c2base,
                        max_depth=self.max_depth,
                    )
                    print(f"      点云图像完成，shape: {pcd_img.shape}")
                    pcd_obs_list.append(pcd_img)
                
                if len(rgb_obs_list) == 0:
                    print("⚠️ 没有成功处理任何相机")
                    return None
                
                print("  堆叠图像...")
                # 堆叠为 (C, H, W, 3) 格式
                rgb_obs = np.stack(rgb_obs_list, axis=0)  # (C, H, W, 3)
                pcd_obs = np.stack(pcd_obs_list, axis=0)  # (C, H, W, 3)
                print("  堆叠完成")
                
                return rgb_obs, pcd_obs, curr_gripper
            
            else:
                # 返回稀疏点云（用于可视化）
                all_xyz, all_rgb = [], []
                
                for i, cam_name in enumerate(self.cam_names):
                    if i >= len(rgb_images):
                        break
                    
                    if cam_name in self.cameras:
                        K = self.cameras[cam_name]["K"]
                        T_c2base = self.cameras[cam_name]["T_camera_to_base"]
                    else:
                        print(f"⚠️ {cam_name} 的标定参数不存在，跳过")
                        continue
                    
                    xyz, rgb = depth_to_pointcloud_base(
                        depth_img=depth_images[i],
                        rgb_img=rgb_images[i],
                        K=K,
                        T_c2base=T_c2base,
                        max_depth=self.max_depth,
                    )
                    
                    all_xyz.append(xyz)
                    all_rgb.append(rgb)
                
                if len(all_xyz) == 0:
                    print("⚠️ 没有成功生成任何点云")
                    return None
                
                # 合并点云
                all_xyz = np.vstack(all_xyz)
                all_rgb = np.vstack(all_rgb)
                
                return all_xyz, all_rgb, eff
            
        except Exception as e:
            print(f"⚠️ 处理同步数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_observation(self, output_path: str) -> bool:
        """
        保存观察数据为.npy文件
        
        Args:
            output_path: 输出文件路径
        
        返回:
            是否成功保存
        """
        # 复制数据引用，避免长时间持有锁
        with self.lock:
            if self.latest_sync_data is None:
                print("⚠️ 还没有接收到同步数据")
                return False
            # 复制消息引用（不需要深度复制，因为我们在处理时不会修改）
            msg = self.latest_sync_data
        
        # 在锁外处理数据（处理可能很慢）
        print("  开始处理同步数据...")
        result = self._process_sync_data(msg, return_image_format=True)
        if result is None:
            print("  ⚠️ 处理数据失败")
            return False
        
        print("  数据处理完成，准备保存...")
        rgb_obs, pcd_obs, curr_gripper = result
        
        # 构建输出字典
        input_dict = {
            "rgb_obs": rgb_obs,
            "pcd_obs": pcd_obs,
            "curr_gripper": curr_gripper,
        }
        
        # 保存为.npy文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), input_dict)
        
        print(f"✓ 保存观察数据到: {output_path}")
        print(f"  - rgb_obs shape: {rgb_obs.shape}")
        print(f"  - pcd_obs shape: {pcd_obs.shape}")
        if curr_gripper is not None:
            print(f"  - curr_gripper shape: {curr_gripper.shape}")
        
        return True
    


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从ROS话题订阅数据并保存为.npy文件")
    parser.add_argument(
        "--sync_topic",
        type=str,
        default="/sensor_sync_package",
        help="同步数据话题（默认: /sensor_sync_package）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据根目录（包含camera_intrinsics.json和camera_to_board_transforms.json）",
    )
    parser.add_argument(
        "--intrinsics_json",
        type=str,
        default=None,
        help="相机内参JSON文件路径",
    )
    parser.add_argument(
        "--extrinsics_json",
        type=str,
        default=None,
        help="相机外参JSON文件路径",
    )
    parser.add_argument(
        "--cam_names",
        type=str,
        nargs="+",
        default=None,
        help="相机名称列表（默认: camera_0 camera_1 camera_2 camera_3）",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="深度缩放因子（默认: 1000.0，将毫米转为米）",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=2.0,
        help="最大深度（米，默认: 2.0）",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="保存观察数据为.npy文件的路径",
    )
    
    args = parser.parse_args()
    
    # 创建处理器
    visualizer = ROSPointCloudVisualizer(
        sync_topic=args.sync_topic,
        intrinsics_json=args.intrinsics_json,
        extrinsics_json=args.extrinsics_json,
        data_root=args.data_root,
        cam_names=args.cam_names,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
    )
    
    # 运行：等待数据并保存
    try:
        print("等待同步数据...")
        count = 0
        while not rospy.is_shutdown():
            # 处理ROS回调（rospy.sleep()会让ROS处理回调）
            rospy.sleep(0.1)  # 每次等待0.1秒
            
            # 检查是否有数据（快速检查，不持有锁太久）
            has_data = False
            with visualizer.lock:
                has_data = visualizer.latest_sync_data is not None
            
            if has_data:
                print("✓ 接收到同步数据，开始处理...")
                visualizer.save_observation(args.save_path)
                break
            
            count += 1
            if count % 10 == 0:  # 每1秒打印一次
                print(f"等待中... (已等待 {count * 0.1:.1f} 秒)")
    except KeyboardInterrupt:
        print("\n退出...")


if __name__ == "__main__":
    main()
