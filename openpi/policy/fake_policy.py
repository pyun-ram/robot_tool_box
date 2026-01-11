import logging
import torch
import numpy as np
from typing import Dict, Optional, Any
from openpi_client import base_policy as _base_policy

VISUALZATION = False

class FakePolicy(_base_policy.BasePolicy):
    """假的Policy实现，用于测试"""
    
    def __init__(self, camera_config: Optional[Dict] = None):
        """初始化假的Policy"""
        self._camera_config = camera_config
        logging.info("FakePolicy initialized")

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

        # 保存obs到本地npy文件
        if VISUALZATION:
            import os
            os.makedirs("./src/temp", exist_ok=True)
            np.save("./src/temp/obs.npy", obs)
        
        # 返回假的action（7个关节的零数组）
        # num_joints = 7
        action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 0.41908594192841403, 43.30252697496553])
        return {
            # "joint_angles": np.zeros(num_joints, dtype=np.float32)
            "action": action
        }
    
    def reset(self) -> None:
        """重置policy状态"""
        pass