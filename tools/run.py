import os
import shutil
from datetime import datetime, timedelta
from bisect import bisect_left
from typing import List
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import re
from tqdm import tqdm
import glob
from PIL import Image
import imageio
from multiprocessing import Pool
import pickle
import fire
# 3d diffuser actor package
from utils import common_utils
from utils.common_utils import LowDimObsDemo, LowDimObservation
from datasets_module.dataset_engine import image_to_float_array_inplace
# class LowDimObsDemo:
#     def __init__(self):
#         self._observations = []
#         self._expert_trajectories = []

#     def __getitem__(self, i):
#         return self._observations[i]
    
#     def __len__(self):
#         return len(self._observations)

#     def restore_state(self):
#         return

def convert_physical_robot_camera_to_rlbench_format(intrinsics_B, extrinsics_B):
    """
    将 B 体系的参数转为 A (RLBench) 体系的参数
    """
    # 1. 处理内参: 强制 fx, fy 为负值以适配 A 的逻辑
    # 注意：A 的 cx, cy 似乎是图像中心，确保 B 的 cx, cy 也是对应的
    intrinsics_A = np.array(intrinsics_B).copy()
    intrinsics_A[0, 0] = -np.abs(intrinsics_B[0][0])
    intrinsics_A[1, 1] = -np.abs(intrinsics_B[1][1])

    # 2. 处理外参 (坐标系变换)
    # A 的投影逻辑中，z 是直接等于 depth，而 fx/fy 是负数。
    # 这意味着在 A 的相机坐标系下：x 是向左的，y 是向上的。
    # 而 B 是 x-右, y-下, z-前。
    # 我们需要构造一个转换矩阵 M，将 B 的相机系转到 A 的相机系
    # M = [A_x, A_y, A_z] 在 B 系下的表示
    # A_x = -B_x, A_y = -B_y, A_z = B_z (根据 A 的投影公式推断)
    M = np.array([
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])
    
    # 修正后的外参 = 原外参 @ M 的逆 (或者是 M @ 原外参，取决于你要修正哪一边)
    # 在这里，我们需要改变的是相机坐标系自身的朝向定义
    extrinsics_A = np.array(extrinsics_B) @ M
    
    return intrinsics_A, extrinsics_A

def convert_physical_robot_openness_to_rlbench_format(openness):
    if openness > 25:
        return 1.0
    else:
        return 0.0

# class LowDimObservation:
#     def __init__(self, gripper_pose, gripper_open, joint_positions, misc):
#         self.gripper_pose = gripper_pose
#         self.gripper_open = gripper_open
#         self.joint_positions = joint_positions
#         self.misc = misc


class Runner:
    def __init__(self, camera_sync_precision: float = 0.03):
        self.camera_sync_precision = camera_sync_precision

    @staticmethod
    def _parse_timestamp(fname: str) -> datetime:
        stem = os.path.splitext(fname)[0]
        return datetime.strptime(stem, "%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _load_timestamps(dir_path: str) -> List[datetime]:
        files = sorted(os.listdir(dir_path))
        return [Runner._parse_timestamp(f) for f in files]

    @staticmethod
    def _closest(ts_list: List[datetime], target: datetime):
        idx = bisect_left(ts_list, target)
        candidates = []
        if idx > 0:
            candidates.append(ts_list[idx - 1])
        if idx < len(ts_list):
            candidates.append(ts_list[idx])
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs((x - target).total_seconds()))

    def organize_data(self, raw_root: str, out_root: str, episode: str) -> None:
        raw_episode_dir = os.path.join(raw_root, episode)

        cam_names = ["cam1", "cam2", "cam3", "cam4"]
        cam_dirs = {}
        for cam in cam_names:
            for m in ["color", "depth"]:
                name = f"{cam}_{m}"
                cam_dirs[name] = os.path.join(raw_episode_dir, name)

        robot_dir = os.path.join(raw_episode_dir, "robot_state")

        # ---------- output dirs ----------
        out_episode_dir = os.path.join(out_root, episode)
        out_dirs = {}
        for name in list(cam_dirs.keys()) + ["robot_state"]:
            out_dir = os.path.join(out_episode_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            out_dirs[name] = out_dir

        # ---------- load timestamps ----------
        ts_map = {k: self._load_timestamps(d) for k, d in cam_dirs.items()}
        robot_ts = self._load_timestamps(robot_dir)
        ts_map["robot_state"] = robot_ts

        start_ts = max(ts_list[0] for ts_list in ts_map.values())
        end_ts = robot_ts[-1]

        # ---------- build timeline ----------
        dt = timedelta(seconds=0.05)
        timeline = []
        t = start_ts
        while t <= end_ts:
            timeline.append(t)
            t += dt

        frame_idx = 0
        all_ok = True

        # 保存所有“将要 move 的操作”
        pending_moves = []  # List[Dict[sensor, src_path]]

        # ---------- sync / check ----------
        for cur_ts in timeline:
            matched = {}
            failed = False

            for sensor, ts_list in ts_map.items():
                closest_ts = self._closest(ts_list, cur_ts)

                if closest_ts is None:
                    print(f"[SKIP] {cur_ts} | {sensor} | no closest timestamp")
                    all_ok = False
                    failed = True
                    break

                diff = abs((closest_ts - cur_ts).total_seconds())
                if diff > self.camera_sync_precision:
                    print(
                        f"[SKIP] {cur_ts} | {sensor} | "
                        f"diff {diff:.4f}s > {self.camera_sync_precision}s"
                    )
                    all_ok = False
                    failed = True
                    break

                matched[sensor] = closest_ts

            if failed:
                continue

            src_files = {}
            for sensor, ts in matched.items():
                if sensor == "robot_state":
                    src_dir = robot_dir
                    ext = ".npy"
                else:
                    src_dir = cam_dirs[sensor]
                    ext = ".png"

                src_name = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3] + ext
                src_path = os.path.join(src_dir, src_name)

                if not os.path.exists(src_path):
                    print(f"[SKIP] {cur_ts} | {sensor} | file missing")
                    all_ok = False
                    failed = True
                    break

                src_files[sensor] = src_path

            if failed:
                continue

            print(f"[OK] frame {frame_idx}")
            pending_moves.append((frame_idx, src_files))
            frame_idx += 1

        # ---------- conditional move ----------
        if not all_ok:
            print("\n[ABORT] Detected SKIP, no files were moved.")
            return

        print("\n[WRITE] All frames OK, start moving files...")

        for frame_idx, src_files in pending_moves:
            for sensor, src_path in src_files.items():
                ext = ".npy" if sensor == "robot_state" else ".png"
                dst_path = os.path.join(out_dirs[sensor], f"{frame_idx}{ext}")
                shutil.move(src_path, dst_path)

        print(f"[DONE] Successfully wrote {len(pending_moves)} frames.")
    
    def convert_data_to_standard_format(self, episode_dir, save_dir) -> None:
        '''
        python tools/run.py convert_data_to_standard_format \
            --episode_dir data/20260102_s1_data/task1/episode0 \
            --save_dir /usr/app/Code/3d_diffuser_actor/data/rmt/20260102_s1_data_converted/train/task1/all_variations/episodes/episode0
        '''

        def _copy_data(src_path, dst_path):
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)
            return

        def _write_pkl(save_path, data):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            return

        def _read_depth_image_real_sense(path, depth_scale=1000.0):
            depth16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            z = depth16.astype(np.float32) / depth_scale
            return z
        
        # 读取calib_dict（只需要读取一次）
        calib_path = os.path.join(episode_dir, "calibrations.json")
        with open(calib_path, "r") as f:
            calib_dict = json.load(f)
        near = 0.01
        far = 4.5
        
        # 获取所有帧索引
        frame_indices = sorted([int(itm.stem) for itm in Path(episode_dir).glob("cam1_color/*.png")])
        err_msg = f"Frame indices are not consecutive: {frame_indices}"
        assert len(frame_indices) == max(frame_indices) + 1, err_msg
        
        # 预先加载所有gripper_pose数据（用于计算trajectory）
        max_frame_index = max(frame_indices)
        gripper_poses_dict = {}
        for frame_index in frame_indices:
            gripper_pose_path = os.path.join(episode_dir, "robot_state", f"{frame_index}.npy")
            gripper_poses_dict[frame_index] = np.load(gripper_pose_path)
        
        low_dim_obs_demo = LowDimObsDemo()
        variation_number = 0  # 可以根据需要修改
        for frame_index in tqdm(frame_indices, desc="Converting frames"):
            # front_rgb, front_depth
            src_path = os.path.join(episode_dir, "cam1_color", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, "front_rgb", f"{frame_index}.png")
            _copy_data(src_path, dst_path)
            src_path = os.path.join(episode_dir, "cam1_depth", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, "front_depth", f"{frame_index}.png")
            depth_m = _read_depth_image_real_sense(src_path)
            depth_m = np.clip(depth_m, near, far)
            depth_uint8 = common_utils.encode_depth_from_float_to_uint8(
                depth_m,
                bool_metric=True,
                near=near,
                far=far,
            )
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            common_utils.write_image(depth_uint8, dst_path)

            
            # nerf 8 rgb, nerf 8 depth
            src_path = os.path.join(episode_dir, "cam2_color", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/images", f"8.png")
            _copy_data(src_path, dst_path)
            src_path = os.path.join(episode_dir, "cam2_depth", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/depths", f"8.png")
            depth_m = _read_depth_image_real_sense(src_path)
            depth_m = np.clip(depth_m, near, far)
            depth_uint8 = common_utils.encode_depth_from_float_to_uint8(
                depth_m,
                bool_metric=True,
                near=near,
                far=far,
            )
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            common_utils.write_image(depth_uint8, dst_path)
            
            # nerf 16 rgb, nerf 16 depth
            src_path = os.path.join(episode_dir, "cam3_color", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/images", f"16.png")
            _copy_data(src_path, dst_path)
            src_path = os.path.join(episode_dir, "cam3_depth", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/depths", f"16.png")
            depth_m = _read_depth_image_real_sense(src_path)
            depth_m = np.clip(depth_m, near, far)
            depth_uint8 = common_utils.encode_depth_from_float_to_uint8(
                depth_m,
                bool_metric=True,
                near=near,
                far=far,
            )
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            common_utils.write_image(depth_uint8, dst_path)
            
            # nerf 36 rgb, nerf 36 depth
            src_path = os.path.join(episode_dir, "cam4_color", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/images", f"36.png")
            _copy_data(src_path, dst_path)
            src_path = os.path.join(episode_dir, "cam4_depth", f"{frame_index}.png")
            dst_path = os.path.join(save_dir, f"nerf_data/{frame_index}/depths", f"36.png")
            depth_m = _read_depth_image_real_sense(src_path)
            depth_m = np.clip(depth_m, near, far)
            depth_uint8 = common_utils.encode_depth_from_float_to_uint8(
                depth_m,
                bool_metric=True,
                near=near,
                far=far,
            )
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            common_utils.write_image(depth_uint8, dst_path)
            
            # low_dim_obs: gripper_pose, gripper_open, joint_positions
            gripper_state = gripper_poses_dict[frame_index]  # shape: (8,), [x, y, z, qx, qy, qz, qw, open]
            gripper_pose = gripper_state[:7]  # 前7个元素：位置(3) + 四元数(4)
            gripper_open = convert_physical_robot_openness_to_rlbench_format(float(gripper_state[7]))
            joint_positions = np.zeros(7, dtype=np.float32)
            
            intrinsics_A, extrinsics_A = convert_physical_robot_camera_to_rlbench_format(
                calib_dict['cam1']['intrinsics'],
                calib_dict['cam1']['extrinsics'])
            misc = {
                'front_camera_extrinsics': extrinsics_A,
                'front_camera_intrinsics': intrinsics_A,
                'front_camera_near': near,
                'front_camera_far': far,
            }
            
            # 保存nerf相机参数
            for cam, newcam in zip(["cam2", "cam3", "cam4"], ["8", "16", "36"]):
                save_path = Path(save_dir) / f'nerf_data/{frame_index}/poses/{newcam}.pkl'
                intrinsics_A, extrinsics_A = convert_physical_robot_camera_to_rlbench_format(
                    calib_dict[cam]['intrinsics'],
                    calib_dict[cam]['extrinsics'])
                _write_pkl(save_path, {
                    'extrinsic': extrinsics_A,
                    'intrinsic': intrinsics_A,
                    'near': near,
                    'far': far,
                })
            
            # low_dim_obs: misc
            low_dim_obs = LowDimObservation(
                gripper_pose=gripper_pose,  # 前7个元素：位置(3) + 四元数(4)
                gripper_open=gripper_open,
                joint_positions=joint_positions,
                misc=misc,
            )
            low_dim_obs_demo._observations.append(low_dim_obs)
            
            # expert_info: trajectory, stage, target_position
            # 获取未来第10帧的gripper_pose
            future_frame_index = frame_index + 10
            if future_frame_index <= max_frame_index and future_frame_index in gripper_poses_dict:
                future_gripper_state = gripper_poses_dict[future_frame_index]
                future_gripper_pose = future_gripper_state[:7]
                future_gripper_open = convert_physical_robot_openness_to_rlbench_format(float(future_gripper_state[7]))
                trajectory = np.concatenate([future_gripper_pose, [future_gripper_open]]).reshape(1, 8).astype(np.float32)
            else:
                # 如果没有未来第10帧，使用最后一帧（max）的值
                max_gripper_state = gripper_poses_dict[max_frame_index]
                max_gripper_pose = max_gripper_state[:7]
                max_gripper_open = convert_physical_robot_openness_to_rlbench_format(float(max_gripper_state[7]))
                trajectory = np.concatenate([max_gripper_pose, [max_gripper_open]]).reshape(1, 8).astype(np.float32)
            
            stage = 'N/A'
            target_position = np.zeros(3, dtype=np.float32)
            
            # 保存expert trajectory信息到expert_info目录
            expert_info_dict = {
                'trajectory': trajectory,
                'stage': stage,
            }
            expert_info_dict['debug_info'] = {'tar_position': target_position}
            expert_info_path = Path(save_dir) / "expert_info" / f"{frame_index}.pkl"
            _write_pkl(expert_info_path, expert_info_dict)
        
        # 保存low_dim_obs_demo
        low_dim_pickle_path = os.path.join(save_dir, "low_dim_obs.pkl")
        _write_pkl(low_dim_pickle_path, low_dim_obs_demo)
        
        # 保存variation_number（只需要保存一次）
        variation_pickle_path = os.path.join(save_dir, "variation_number.pkl")
        _write_pkl(variation_pickle_path, variation_number)
        
        print(f"[DONE] Successfully converted {len(frame_indices)} frames to standard format in {save_dir}")

    def check_calibration_one_task(self, out_root: str, save_base_dir: str = None, frame_skip: int = 5, num_processes: int = 4) -> None:
        """
        处理一个task目录下的所有episodes的校准检查
        
        Args:
            out_root: 数据根目录，例如 "./data/20260102_s1_data/task1/"
            save_base_dir: 保存目录的基础路径，如果为None则使用默认路径
            frame_skip: 帧采样间隔，例如5表示每5帧处理一帧
            num_processes: 多进程数量
        """
        out_root_path = Path(out_root)
        
        # 找到所有episode目录并按数字排序
        def episode_sort_key(x):
            episode_num_str = x.name.replace("episode", "")
            try:
                return int(episode_num_str)
            except ValueError:
                return 999999
        
        episode_dirs = sorted(out_root_path.glob("episode*"), key=episode_sort_key)
        
        print(f"找到 {len(episode_dirs)} 个episodes")
        
        for episode_dir in episode_dirs:
            episode = episode_dir.name
            print(f"\n处理 {episode}...")
            
            if save_base_dir is None:
                save_dir = f"./data/20260102_s1_data/check_calibration/task1/{episode}"
            else:
                save_dir = os.path.join(save_base_dir, episode)
            
            # 检查episode目录是否存在必要的文件
            cam1_color_dir = episode_dir / "cam1_color"
            if not cam1_color_dir.exists():
                print(f"[SKIP] {episode}: cam1_color目录不存在")
                continue
                
            num_frames = len(list(cam1_color_dir.glob("*.png")))
            if num_frames == 0:
                print(f"[SKIP] {episode}: 没有找到图像文件")
                continue
                
            print(f"  {episode}: 找到 {num_frames} 帧")
            indices = sorted(list(range(num_frames)))[::frame_skip]
            args_list = [(out_root, episode, i, save_dir) for i in indices]
            
            # 使用多进程执行
            with Pool(processes=num_processes) as pool:
                pool.map(_check_calibration_wrapper, args_list)
            
            # 生成GIF可视化
            gif_output_dir = Path(save_dir).parent / "gif"
            gif_output_dir.mkdir(parents=True, exist_ok=True)
            self.visualize_calibration(
                save_dir,
                str(gif_output_dir / f"{episode}.gif"))
            
            print(f"  {episode}: 完成")
        
        print(f"\n所有episodes处理完成！")

    def check_calibration(self, out_root: str, episode: str, index: int, save_dir: str='./', save_ply: bool=False) -> None:
        episode_dir = os.path.join(out_root, episode)
        calib_path = os.path.join(episode_dir, "calibrations.json")

        if not os.path.exists(calib_path):
            raise FileNotFoundError(calib_path)

        with open(calib_path, "r") as f:
            calib = json.load(f)

        def rgbd_to_base_pcd(
                rgb_img: np.ndarray,
                depth_path: str,
                K: np.ndarray,
                T_c2base: np.ndarray,
                depth_scale=1000.0,
                max_depth=2.0,
        ):
            # ---- copy 外参，避免原地修改 ----
            T = T_c2base.copy()

            rgb = rgb_img  # rgb_img 已经是RGB格式的numpy数组
            depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if rgb.shape[:2] != depth16.shape[:2]:
                depth16 = cv2.resize(depth16, (rgb.shape[1], rgb.shape[0]))
                assert False, "Depth image size does not match RGB image size"

            H, W = rgb.shape[:2]

            z = depth16.astype(np.float32) / depth_scale
            mask = (z > 0) & (z < max_depth)

            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u[mask]
            v = v[mask]
            z = z[mask]

            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            xyz_cam = np.stack([x, y, z], axis=1)

            xyz_hom = np.hstack([xyz_cam, np.ones((xyz_cam.shape[0], 1))])
            xyz_base = (T @ xyz_hom.T).T[:, :3]

            rgb_colors = rgb[mask] / 255.0

            return xyz_base, rgb_colors

        # --------------------------------------------------
        # 1. 读取四个相机，生成点云
        # --------------------------------------------------
        all_xyz_base = []
        all_rgb = []
        cached_rgb_images = {}  # 缓存RGB图像，避免重复读取

        for cam in ["cam1", "cam2", "cam3", "cam4"]:
            print(f"[LOAD] {cam}")

            rgb_path = os.path.join(
                episode_dir, f"{cam}_color", f"{index}.png"
            )
            depth_path = os.path.join(
                episode_dir, f"{cam}_depth", f"{index}.png"
            )

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"[SKIP] missing data for {cam}")
                continue

            # 读取RGB图像并转换为RGB格式，然后缓存
            img_bgr = cv2.imread(rgb_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            cached_rgb_images[cam] = img_rgb

            K = np.array(calib[cam]["intrinsics"], dtype=np.float32)
            T_c2base = np.array(calib[cam]["extrinsics"], dtype=np.float32)

            xyz, rgb = rgbd_to_base_pcd(
                img_rgb,  # 传递RGB图像数组而不是路径
                depth_path,
                K,
                T_c2base,
                depth_scale=1000.0,
                max_depth=2.0,
            )

            all_xyz_base.append(xyz)
            all_rgb.append(rgb)

        if not all_xyz_base:
            print("[ERROR] no valid point cloud")
            return

        all_xyz_base = np.vstack(all_xyz_base)
        all_rgb = np.vstack(all_rgb)

        # --------------------------------------------------
        # 2. 构造 Open3D 点云
        # --------------------------------------------------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_xyz_base)
        pcd.colors = o3d.utility.Vector3dVector(all_rgb)

        # --------------------------------------------------
        # 3. 辅助函数：将 TriangleMesh 转换为点云
        # --------------------------------------------------
        def mesh_to_pointcloud(mesh: o3d.geometry.TriangleMesh, num_points: int = 1000) -> o3d.geometry.PointCloud:
            """将 TriangleMesh 转换为点云（通过均匀采样表面点）"""
            return mesh.sample_points_uniformly(number_of_points=num_points)

        # --------------------------------------------------
        # 4. 添加可视化辅助元素（世界坐标系）
        # --------------------------------------------------
        world_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        world_frame_pcd = mesh_to_pointcloud(world_frame_mesh, num_points=500)
        pcd = pcd + world_frame_pcd

        # --------------------------------------------------
        # 5. 读取并添加 robot_state 可视化
        # --------------------------------------------------
        robot_state_path = os.path.join(
            episode_dir, "robot_state", f"{index}.npy"
        )

        if os.path.exists(robot_state_path):
            state = np.load(robot_state_path)

            if state.shape[0] < 7:
                print("[WARN] robot_state length < 7, skip visualization")
            else:
                x, y, z = state[0:3]
                qx, qy, qz, qw = state[3:7]

                # ---- 四元数 -> 旋转矩阵 (Open3D 接受 [w, x, y, z])
                R = o3d.geometry.get_rotation_matrix_from_quaternion(
                    [qw, qx, qy, qz]
                )

                # ---- 构造 4x4 位姿矩阵
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [x, y, z]

                # ---- 位置：红色小球
                robot_sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.04
                )
                robot_sphere_mesh.paint_uniform_color([1.0, 0.0, 0.0])
                robot_sphere_mesh.transform(T)
                robot_sphere_pcd = mesh_to_pointcloud(robot_sphere_mesh, num_points=300)
                pcd = pcd + robot_sphere_pcd

                # ---- 朝向：机器人坐标系
                robot_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2, origin=[0, 0, 0]
                )
                robot_frame_mesh.transform(T)
                robot_frame_pcd = mesh_to_pointcloud(robot_frame_mesh, num_points=500)
                pcd = pcd + robot_frame_pcd
        else:
            print(f"[WARN] robot_state not found: {robot_state_path}")

        # --------------------------------------------------
        # 6. 保存为 PLY 文件
        # --------------------------------------------------
        if save_ply:
            output_ply_path = os.path.join(save_dir, f"calibration_check_frame_{index}.ply")
            os.makedirs(save_dir, exist_ok=True)
            o3d.io.write_point_cloud(output_ply_path, pcd)
            print(f"[SAVE] Point cloud saved to: {output_ply_path}")

        # --------------------------------------------------
        # 7. 将点云投影到四个相机的图像平面并保存
        # --------------------------------------------------
        # 获取点云的numpy数组
        points_base = np.asarray(pcd.points)[::2, :]
        colors = np.asarray(pcd.colors)[::2, :]
        
        for cam in ["cam1", "cam2", "cam3", "cam4"]:
            # 从缓存获取RGB图像
            if cam not in cached_rgb_images:
                print(f"[SKIP] missing RGB image for {cam}, skip projection")
                continue
            
            img_rgb = cached_rgb_images[cam]
            H, W = img_rgb.shape[:2]
            
            # 转换为BGR格式用于后续绘制和保存
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # 获取相机参数
            K = np.array(calib[cam]["intrinsics"], dtype=np.float32)
            T_c2base = np.array(calib[cam]["extrinsics"], dtype=np.float32)
            
            # 计算从base到camera的变换（外参的逆矩阵）
            T_base2c = np.linalg.inv(T_c2base)
            
            # 将点云从base坐标系变换到camera坐标系
            points_hom = np.hstack([points_base, np.ones((points_base.shape[0], 1))])
            points_cam = (T_base2c @ points_hom.T).T[:, :3]
            
            # 过滤掉相机后方的点（z < 0）
            mask_valid = points_cam[:, 2] > 0.01  # 至少1cm距离
            points_cam_valid = points_cam[mask_valid]
            colors_valid = colors[mask_valid]
            depths_valid = points_cam_valid[:, 2]  # 保存深度信息（z坐标）
            
            if len(points_cam_valid) == 0:
                print(f"[SKIP] no valid points for {cam} projection")
                continue
            
            # 使用内参投影到图像平面
            x_proj = points_cam_valid[:, 0] / points_cam_valid[:, 2] * K[0, 0] + K[0, 2]
            y_proj = points_cam_valid[:, 1] / points_cam_valid[:, 2] * K[1, 1] + K[1, 2]
            
            # 过滤掉图像范围外的点
            mask_in_image = (x_proj >= 0) & (x_proj < W) & (y_proj >= 0) & (y_proj < H)
            x_proj = x_proj[mask_in_image].astype(int)
            y_proj = y_proj[mask_in_image].astype(int)
            colors_valid = colors_valid[mask_in_image]
            depths_valid = depths_valid[mask_in_image]  # 同时过滤深度信息
            
            # 创建投影图像（复制原始图像，使用BGR格式用于绘制）
            img_proj_bgr = img_bgr.copy()
            
            # 创建深度缓冲区（z-buffer），初始化为很大的值
            depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
            
            # 将颜色从[0,1]范围转换到[0,255]，并转换为BGR格式
            colors_rgb_uint8 = (colors_valid * 255).astype(np.uint8)
            colors_bgr_uint8 = colors_rgb_uint8[:, [2, 1, 0]].astype(np.uint8)  # RGB -> BGR
            
            # 根据深度绘制点：使用for循环，对于每个点，只在更近时才覆盖颜色
            radius = 2
            # 创建圆形模板的偏移坐标（相对于中心点的偏移）
            y_offsets, x_offsets = np.ogrid[-radius:radius+1, -radius:radius+1]
            circle_mask = (x_offsets**2 + y_offsets**2) <= radius**2
            y_circle, x_circle = np.where(circle_mask)
            y_circle = y_circle - radius  # 转换为相对于中心的偏移
            x_circle = x_circle - radius
            
            num_points = len(x_proj)
            for i in range(num_points):
                x_center, y_center = x_proj[i], y_proj[i]
                depth = depths_valid[i]
                color = colors_bgr_uint8[i]
                
                # 计算当前点圆形区域内所有像素的坐标
                x_pixels = x_center + x_circle
                y_pixels = y_center + y_circle
                
                # 过滤掉超出图像边界的像素
                valid_mask = (x_pixels >= 0) & (x_pixels < W) & (y_pixels >= 0) & (y_pixels < H)
                x_pixels_valid = x_pixels[valid_mask]
                y_pixels_valid = y_pixels[valid_mask]
                
                if len(x_pixels_valid) == 0:
                    continue
                
                # 批量检查深度：只有当当前点更近时才更新
                # 使用向量化操作比较深度
                closer_mask = depth < depth_buffer[y_pixels_valid, x_pixels_valid]
                if np.any(closer_mask):
                    # 只更新更近的像素
                    y_closer = y_pixels_valid[closer_mask]
                    x_closer = x_pixels_valid[closer_mask]
                    depth_buffer[y_closer, x_closer] = depth
                    img_proj_bgr[y_closer, x_closer] = color
            
            # 保存投影图像
            output_proj_path = os.path.join(
                save_dir, f"{cam}_idx{index}_projection.png"
            )
            Path(output_proj_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_proj_path, img_proj_bgr)
            print(f"[SAVE] Projection image saved to: {output_proj_path}")


    def visualize_calibration(self, save_dir: str, output_gif_path) -> None:
        """
        将校准检查文件夹中的投影图像组合成2x2布局并生成GIF
        
        Args:
            save_dir: 包含投影图像的目录（格式: {cam}_idx{index}_projection.png）
        """
        # 找到所有投影图像文件
        pattern = os.path.join(save_dir, "cam*_idx*_projection.png")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"[ERROR] No projection images found in {save_dir}")
            return
        
        # 按索引分组图像
        indices_dict = {}  # {index: {cam1: path, cam2: path, ...}}
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            # 解析文件名格式: cam1_idx0_projection.png
            parts = filename.replace("_projection.png", "").split("_idx")
            if len(parts) != 2:
                continue
            cam_name = parts[0]  # cam1, cam2, etc.
            index = int(parts[1])
            
            if index not in indices_dict:
                indices_dict[index] = {}
            indices_dict[index][cam_name] = filepath
        
        # 获取所有索引并排序
        indices = sorted(indices_dict.keys())
        
        if not indices:
            print(f"[ERROR] No valid indices found")
            return
        
        print(f"[INFO] Found {len(indices)} frames to visualize")
        
        # 为每个索引组合图像
        frames = []
        cam_names = ["cam1", "cam2", "cam3", "cam4"]
        
        for idx in tqdm(indices, desc="Combining images"):
            cam_images = []
            cam_paths = indices_dict[idx]
            
            # 加载四个相机的图像
            for cam in cam_names:
                if cam in cam_paths:
                    img = Image.open(cam_paths[cam])
                    # 转换为RGB格式（如果是RGBA或其他格式）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    cam_images.append(img)
                else:
                    # 如果某个相机缺失，创建黑色占位图像
                    if cam_images:
                        # 使用已有图像的尺寸
                        placeholder = Image.new('RGB', cam_images[0].size, color='black')
                    else:
                        placeholder = Image.new('RGB', (640, 480), color='black')
                    cam_images.append(placeholder)
            
            # 确保所有图像尺寸相同（使用第一个非空图像的尺寸）
            target_size = cam_images[0].size
            cam_images = [img.resize(target_size) if img.size != target_size else img 
                         for img in cam_images]
            
            # 组合成2x2布局: [cam1, cam2; cam3, cam4]
            img1, img2, img3, img4 = cam_images
            
            # 转换为numpy数组进行拼接
            img1_arr = np.array(img1)
            img2_arr = np.array(img2)
            img3_arr = np.array(img3)
            img4_arr = np.array(img4)
            
            # 水平拼接：上排 [cam1, cam2]，下排 [cam3, cam4]
            top_row = np.hstack([img1_arr, img2_arr])
            bottom_row = np.hstack([img3_arr, img4_arr])
            
            # 垂直拼接
            combined_arr = np.vstack([top_row, bottom_row])
            
            # 转换回PIL Image并调整尺寸到 640x480
            combined_img = Image.fromarray(combined_arr)
            # 使用LANCZOS重采样以获得更好的图像质量
            combined_img = combined_img.resize((420, 360), Image.Resampling.LANCZOS)
            frames.append(combined_img)
        
        if frames:
            frames[0].save(
                output_gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,  # 每帧200ms
                loop=0,
                format="GIF"
            )
            print(f"[SAVE] Visualization GIF saved to: {output_gif_path}")
        else:
            print("[ERROR] No frames to save")

    def visualize_image(self, image_dir: str) -> None:
        image1_dir='/home/pyun/Docker/RM01/Data/episode0/cam1_color_image_raw_compressed'
        image2_dir='/home/pyun/Docker/RM01/Data/episode0/cam2_color_image_raw_compressed'
        image3_dir='/home/pyun/Docker/RM01/Data/episode0/cam3_color_image_raw_compressed'
        image4_dir='/home/pyun/Docker/RM01/Data/episode0/cam4_color_image_raw_compressed'
        
        # get image path list (image1_dir)
        image1_path = Path(image1_dir)
        image_files = sorted([f for f in image1_path.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        
        if not image_files:
            raise ValueError(f"No image files found in {image1_dir}")
        
        # Extract timestamps from filenames (format: YYYYMMDD_HHMMSS_mmm.png -> extract HHMMSS.mmm)
        def extract_timestamp(filepath: Path) -> float:
            """Extract timestamp from filename. Format: 20251220_172659_411.png -> 172659.411"""
            filename = filepath.stem
            # Match pattern: YYYYMMDD_HHMMSS_mmm
            # Extract HHMMSS and mmm, combine as HHMMSS.mmm
            match = re.search(r'_(\d{6})_(\d{3})$', filename)
            if match:
                time_part = match.group(1)  # e.g., "172659"
                ms_part = match.group(2)    # e.g., "411"
                timestamp_str = f"{time_part}.{ms_part}"  # e.g., "172659.411"
                return float(timestamp_str)
            # Fallback to file modification time
            return os.path.getmtime(filepath)
        
        # Get timestamps for image1 using absolute time
        image1_timestamps = [extract_timestamp(f) for f in image_files]
        t_start = image1_timestamps[0]  # First frame absolute timestamp
        t_end = image1_timestamps[-1]   # Last frame absolute timestamp
        
        # get timestamp list with image1 as reference [use the first frame as t0 and compute future timestampes at 1 Hz][freq 5 Hz]
        # 5 Hz means 0.2 seconds per frame
        frame_interval = 1  # seconds
        target_timestamps = np.arange(t_start, t_end + frame_interval, frame_interval)
        
        # Process each camera directory separately
        def load_camera_images(camera_dir: str) -> tuple:
            """Load images and timestamps for a camera directory."""
            cam_path = Path(camera_dir)
            files = sorted([f for f in cam_path.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
            if not files:
                return [], []
            # Extract absolute timestamps
            timestamps = [extract_timestamp(f) for f in files]
            return files, timestamps
        
        # Load images for each camera
        cam1_files, cam1_timestamps = load_camera_images(image1_dir)
        cam2_files, cam2_timestamps = load_camera_images(image2_dir)
        cam3_files, cam3_timestamps = load_camera_images(image3_dir)
        cam4_files, cam4_timestamps = load_camera_images(image4_dir)
        
        def find_closest_image(target_time: float, files: list, timestamps: list) -> Path:
            """Find the image file with timestamp closest to target_time."""
            if not files or not timestamps:
                return None
            idx = np.argmin([abs(t - target_time) for t in timestamps])
            return files[idx]
        
        # for each timestamp
        frames = []
        for target_time in tqdm(target_timestamps):
            # Process each camera separately
            # Get closest image from cam1
            cam1_file = find_closest_image(target_time, cam1_files, cam1_timestamps)
            cam1_img = Image.open(cam1_file) if cam1_file and cam1_file.exists() else Image.new('RGB', (640, 480), color='black')
            
            # Get closest image from cam2
            cam2_file = find_closest_image(target_time, cam2_files, cam2_timestamps)
            cam2_img = Image.open(cam2_file) if cam2_file and cam2_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Get closest image from cam3
            cam3_file = find_closest_image(target_time, cam3_files, cam3_timestamps)
            cam3_img = Image.open(cam3_file) if cam3_file and cam3_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Get closest image from cam4
            cam4_file = find_closest_image(target_time, cam4_files, cam4_timestamps)
            cam4_img = Image.open(cam4_file) if cam4_file and cam4_file.exists() else Image.new('RGB', cam1_img.size, color='black')
            
            # Resize all images to same size if needed
            images = [cam1_img, cam2_img, cam3_img, cam4_img]
            sizes = [img.size for img in images]
            target_size = max(sizes, key=lambda x: x[0] * x[1])
            images = [img.resize(target_size) if img.size != target_size else img for img in images]
            
            # concatenate image [2x2] [cam1, cam2; cam3, cam4]
            img1, img2, img3, img4 = images
            top_row = np.hstack([np.array(img1), np.array(img2)])
            bottom_row = np.hstack([np.array(img3), np.array(img4)])
            combined = np.vstack([top_row, bottom_row])
            combined = Image.fromarray(combined).resize((640, 480))
            frames.append(combined)
        
        # visualize image with gif
        if frames:
            output_path = 'visualization.gif'
            imageio.mimsave(str(output_path), frames, duration=frame_interval, loop=0)
            print(f"GIF saved to {output_path}")
        else:
            raise ValueError("No frames to visualize")

    def visualize_standard_format(self, data_dir: str, frame_idx: int = 0) -> None:
        '''
        python tools/run.py visualize_standard_format \
            --data_dir data/20260102_s1_data_converted/task1/episode0 \
            --frame_idx 0
        '''
        # 辅助函数：从深度图转换为点云（RLBench格式）
        def pointcloud_from_depth_and_camera_params_inplace(
            depth: np.ndarray, extrinsics: np.ndarray,
            intrinsics: np.ndarray,
            cache_key: tuple = None) -> np.ndarray:
            """
            Memory-friendly conversion from depth (meters) to world point cloud.
            Consistent with RLBench's pointcloud_from_depth_and_camera_params_inplace
            """
            assert depth.ndim == 2, "depth must be HxW"
            H, W = depth.shape
            # Extract intrinsics
            if intrinsics.shape[0] >= 3 and intrinsics.shape[1] >= 3:
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])
            else:
                raise ValueError(f'Invalid intrinsics shape: {intrinsics.shape}')
            # Normalized pixel grid (cached)
            if not hasattr(self, '_uv_cache'):
                self._uv_cache = {}
            key = cache_key if cache_key is not None else (H, W, fx, fy, cx, cy)
            if key in self._uv_cache:
                u_norm, v_norm = self._uv_cache[key]
            else:
                u = np.arange(W, dtype=np.float32)
                v = np.arange(H, dtype=np.float32)
                uu, vv = np.meshgrid(u, v)  # (H, W)
                u_norm = (uu - cx) / fx
                v_norm = (vv - cy) / fy
                self._uv_cache[key] = (u_norm, v_norm)
            # Camera coordinates
            depth = depth.astype(np.float32, copy=False)
            x = u_norm * depth
            y = v_norm * depth
            z = depth
            # Camera-to-world transform from extrinsics
            R = extrinsics[:3, :3].astype(np.float32, copy=False)
            t = extrinsics[:3, 3].astype(np.float32, copy=False)
            # Allocate output and fill in-place
            pcd = np.empty((H, W, 3), dtype=np.float32)
            pcd[..., 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0]
            pcd[..., 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1]
            pcd[..., 2] = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2]
            return pcd
        
        # 辅助函数：读取深度图并转换为米（与check_calibration中的方法一致）
        def read_depth_image(depth_path: Path, near: float, far: float) -> np.ndarray:
            """
            读取深度图并转换为米单位
            与check_calibration中的rgbd_to_base_pcd函数使用相同的深度读取方式
            """
            depth = image_to_float_array_inplace(common_utils.read_image(depth_path), common_utils.DEPTH_SCALE)
            depth_m = near + depth * (far - near)
            return depth_m

        data_path = Path(data_dir)
        
        # 读取low_dim_obs获取相机参数
        low_dim_obs_path = data_path / "low_dim_obs.pkl"
        if not low_dim_obs_path.exists():
            raise FileNotFoundError(f"low_dim_obs.pkl not found in {data_dir}")
        
        with open(low_dim_obs_path, "rb") as f:
            low_dim_obs_demo = pickle.load(f)
        
        if frame_idx >= len(low_dim_obs_demo):
            raise ValueError(f"frame_idx {frame_idx} out of range (max: {len(low_dim_obs_demo)-1})")
        
        obs = low_dim_obs_demo[frame_idx]
        
        # 获取front相机的参数（与convert_data_to_standard_format中保存的键名一致）
        front_intrinsics = np.array(obs.misc['front_camera_intrinsics'], dtype=np.float32)
        front_extrinsics = np.array(obs.misc['front_camera_extrinsics'], dtype=np.float32)
        front_near = obs.misc['front_camera_near']
        front_far = obs.misc['front_camera_far']
        
        # 读取front depth和rgb
        front_depth_path = data_path / "front_depth" / f"{frame_idx}.png"
        front_rgb_path = data_path / "front_rgb" / f"{frame_idx}.png"
        
        front_depth_m = read_depth_image(front_depth_path, front_near, front_far)
        front_rgb = cv2.imread(str(front_rgb_path))
        if front_rgb is not None:
            front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_BGR2RGB)
        
        # 转换front点云
        front_pcd = pointcloud_from_depth_and_camera_params_inplace(
            front_depth_m, front_extrinsics, front_intrinsics
        )
        # 裁剪点云
        front_pcd[..., 0] = np.clip(front_pcd[..., 0], -2.5, 2.5)
        front_pcd[..., 1] = np.clip(front_pcd[..., 1], -2.5, 2.5)
        front_pcd[..., 2] = np.clip(front_pcd[..., 2], 0, 2)
        
        # 收集所有点云和颜色（用于合并）
        all_points = []
        all_colors = []
        
        # 保存front点云
        mask_valid = (front_pcd[..., 2] > 0.01) & (front_pcd[..., 2] < 2.0)
        front_points_valid = front_pcd[mask_valid]
        if front_rgb is not None:
            front_colors = front_rgb[mask_valid] / 255.0
        else:
            front_colors = np.ones((np.sum(mask_valid), 3), dtype=np.float32) * 0.5
        
        # 创建并保存front点云
        front_pcd_o3d = o3d.geometry.PointCloud()
        front_pcd_o3d.points = o3d.utility.Vector3dVector(front_points_valid)
        front_pcd_o3d.colors = o3d.utility.Vector3dVector(front_colors)
        front_output_path = data_path / f"pointcloud_frame_{frame_idx}_front.ply"
        o3d.io.write_point_cloud(str(front_output_path), front_pcd_o3d)
        print(f"[SAVE] Front point cloud saved to: {front_output_path} ({len(front_points_valid)} points)")
        
        all_points.append(front_points_valid)
        all_colors.append(front_colors)
        
        # 读取nerf相机 (8, 16, 36) - 对应 cam2, cam3, cam4
        for cam_name in ["8", "16", "36"]:
            # 读取相机参数（路径已更新：nerf_data/{frame_idx}/poses/{cam_name}.pkl）
            cam_pose_path = data_path / f"nerf_data/{frame_idx}/poses/{cam_name}.pkl"
            if not cam_pose_path.exists():
                print(f"[WARN] Camera pose not found: {cam_pose_path}, skipping")
                continue
            
            with open(cam_pose_path, "rb") as f:
                cam_params = pickle.load(f)
            
            cam_intrinsics = np.array(cam_params['intrinsic'], dtype=np.float32)
            cam_extrinsics = np.array(cam_params['extrinsic'], dtype=np.float32)
            cam_near = cam_params['near']
            cam_far = cam_params['far']
            
            # 读取depth和rgb（路径已更新：nerf_data/{frame_idx}/depths/{cam_name}.png 和 nerf_data/{frame_idx}/images/{cam_name}.png）
            cam_depth_path = data_path / f"nerf_data/{frame_idx}/depths/{cam_name}.png"
            cam_rgb_path = data_path / f"nerf_data/{frame_idx}/images/{cam_name}.png"
            
            if not cam_depth_path.exists() or not cam_rgb_path.exists():
                print(f"[WARN] Camera {cam_name} images not found, skipping")
                continue
            
            cam_depth_m = read_depth_image(cam_depth_path, cam_near, cam_far)
            cam_rgb = cv2.imread(str(cam_rgb_path))
            if cam_rgb is not None:
                cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB)
            
            # 转换点云
            cam_pcd = pointcloud_from_depth_and_camera_params_inplace(
                cam_depth_m, cam_extrinsics, cam_intrinsics
            )
            # 裁剪点云
            cam_pcd[..., 0] = np.clip(cam_pcd[..., 0], -2.5, 2.5)
            cam_pcd[..., 1] = np.clip(cam_pcd[..., 1], -2.5, 2.5)
            cam_pcd[..., 2] = np.clip(cam_pcd[..., 2], 0, 2)
            
            # 添加有效点
            mask_valid = (cam_pcd[..., 2] > 0.01) & (cam_pcd[..., 2] < 2.0)
            cam_points_valid = cam_pcd[mask_valid]
            if cam_rgb is not None:
                cam_colors = cam_rgb[mask_valid] / 255.0
            else:
                cam_colors = np.ones((np.sum(mask_valid), 3), dtype=np.float32) * 0.5
            
            # 创建并保存当前相机的点云
            cam_pcd_o3d = o3d.geometry.PointCloud()
            cam_pcd_o3d.points = o3d.utility.Vector3dVector(cam_points_valid)
            cam_pcd_o3d.colors = o3d.utility.Vector3dVector(cam_colors)
            cam_output_path = data_path / f"pointcloud_frame_{frame_idx}_cam{cam_name}.ply"
            o3d.io.write_point_cloud(str(cam_output_path), cam_pcd_o3d)
            print(f"[SAVE] Camera {cam_name} point cloud saved to: {cam_output_path} ({len(cam_points_valid)} points)")
            
            all_points.append(cam_points_valid)
            all_colors.append(cam_colors)
        
        # 合并所有点云（可选：也保存合并版本）
        if len(all_points) == 0:
            raise ValueError("No valid point clouds found")
        
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        # 创建Open3D点云并保存合并版本
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        
        # 保存合并的PLY文件
        merged_output_path = data_path / f"merged_pointcloud_frame_{frame_idx}.ply"
        o3d.io.write_point_cloud(str(merged_output_path), merged_pcd)
        print(f"[SAVE] Merged point cloud saved to: {merged_output_path}")
        print(f"[INFO] Total points: {len(merged_points)}")
        print(f"[INFO] Point cloud bounds: x=[{merged_points[:, 0].min():.2f}, {merged_points[:, 0].max():.2f}], "
              f"y=[{merged_points[:, 1].min():.2f}, {merged_points[:, 1].max():.2f}], "
              f"z=[{merged_points[:, 2].min():.2f}, {merged_points[:, 2].max():.2f}]")

def _check_calibration_wrapper(args):
    """包装函数，用于多进程调用 check_calibration"""
    out_root, episode, index, save_dir = args
    runner = Runner()
    runner.check_calibration(out_root, episode, index, save_dir)
    return index

if __name__ == "__main__":
    fire.Fire(Runner)