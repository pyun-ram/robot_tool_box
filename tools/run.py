import os
import shutil
from datetime import datetime, timedelta
from bisect import bisect_left
from typing import List
import json
import numpy as np
import cv2
import open3d as o3d


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

    def check_calibration(self, out_root: str, episode: str, index: int) -> None:
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
        # 3. 可视化辅助元素（世界坐标系）
        # --------------------------------------------------
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        # 3.5 读取并可视化 robot_state
        # --------------------------------------------------
        robot_state_path = os.path.join(
            episode_dir, "robot_state", f"{index}.npy"
        )

        robot_geoms = []

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
                robot_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.04
                )
                robot_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                robot_sphere.transform(T)

                # ---- 朝向：机器人坐标系
                robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2, origin=[0, 0, 0]
                )
                robot_frame.transform(T)

                robot_geoms.extend([robot_sphere, robot_frame])
        else:
            print(f"[WARN] robot_state not found: {robot_state_path}")

        # --------------------------------------------------
        # 4. 显示
        # --------------------------------------------------
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Calibration Check - frame {index}")
        vis.add_geometry(pcd)
        vis.add_geometry(world_frame)

        for g in robot_geoms:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    runner = Runner()
    # runner.organize_data("./data_raw", "./data", "episode0")
    runner.check_calibration("./data", "episode0", 0)
