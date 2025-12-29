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

    def check_calibration(self, out_root: str, episode: str, index: int, save_dir: str='./') -> None:
        episode_dir = os.path.join(out_root, episode)
        calib_path = os.path.join(episode_dir, "calibrations.json")

        if not os.path.exists(calib_path):
            raise FileNotFoundError(calib_path)

        with open(calib_path, "r") as f:
            calib = json.load(f)

        def rgbd_to_base_pcd(
                rgb_path: str,
                depth_path: str,
                K: np.ndarray,
                T_c2base: np.ndarray,
                depth_scale=1000.0,
                max_depth=2.0,
        ):
            # ---- copy 外参，避免原地修改 ----
            T = T_c2base.copy()

            bgr = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if rgb.shape[:2] != depth16.shape[:2]:
                depth16 = cv2.resize(depth16, (rgb.shape[1], rgb.shape[0]))

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

            K = np.array(calib[cam]["intrinsics"], dtype=np.float32)
            T_c2base = np.array(calib[cam]["extrinsics"], dtype=np.float32)

            xyz, rgb = rgbd_to_base_pcd(
                rgb_path,
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
        output_ply_path = os.path.join(save_dir, f"calibration_check_frame_{index}.ply")
        os.makedirs(save_dir, exist_ok=True)
        o3d.io.write_point_cloud(output_ply_path, pcd)
        print(f"[SAVE] Point cloud saved to: {output_ply_path}")

        # --------------------------------------------------
        # 7. 将点云投影到四个相机的图像平面并保存
        # --------------------------------------------------
        # 获取点云的numpy数组
        points_base = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        for cam in ["cam1", "cam2", "cam3", "cam4"]:
            rgb_path = os.path.join(
                episode_dir, f"{cam}_color", f"{index}.png"
            )
            
            if not os.path.exists(rgb_path):
                print(f"[SKIP] missing RGB image for {cam}, skip projection")
                continue
            
            # 读取原始图像
            img_bgr = cv2.imread(rgb_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            
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
            
            # 创建投影图像（复制原始图像，使用BGR格式用于绘制）
            img_proj_bgr = img_bgr.copy()
            
            # 在图像上绘制投影点
            # 将颜色从[0,1]范围转换到[0,255]，并转换为BGR格式
            colors_rgb_uint8 = (colors_valid * 255).astype(np.uint8)
            colors_bgr_uint8 = colors_rgb_uint8[:, [2, 1, 0]]  # RGB -> BGR
            
            # 绘制点（使用小圆点）
            for i in range(len(x_proj)):
                color_bgr = tuple(map(int, colors_bgr_uint8[i]))
                cv2.circle(img_proj_bgr, (x_proj[i], y_proj[i]), 2, 
                          color=color_bgr, thickness=-1)
            
            # 保存投影图像
            output_proj_path = os.path.join(
                save_dir, f"{cam}_idx{index}_projection.png"
            )
            cv2.imwrite(output_proj_path, img_proj_bgr)
            print(f"[SAVE] Projection image saved to: {output_proj_path}")


    def visualize_calibration(self, save_dir: str) -> None:
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
        
        # 保存为GIF
        output_gif_path = os.path.join("calibration_visualization.gif")
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

if __name__ == "__main__":
    runner = Runner()
    # runner.organize_data("./data_raw", "./data", "episode0")
    runner.check_calibration("./data", "episode0", 0, f"./data/check_calibration")
    # for i in range(197):
    #     runner.check_calibration("./data", "episode0", i, f"./data/check_calibration")
    # runner.visualize_calibration("./data/check_calibration")
