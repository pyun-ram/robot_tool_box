import logging
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
from openpi_client import base_policy as _base_policy

# 尝试导入tap和DiffuserActor
try:
    import tap
    from diffuser_actor import DiffuserActor
    HAS_DIFFUSER = True
except ImportError:
    HAS_DIFFUSER = False
    logging.warning("tap or DiffuserActor not available, using fallback")

# 尝试导入其他可能需要的函数
try:
    from utils import count_parameters, get_gripper_loc_bounds
except ImportError:
    def count_parameters(model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_gripper_loc_bounds(bounds_str, task=None, buffer=0.08):
        """解析gripper位置边界"""
        # 简单实现，实际应该根据bounds_str解析
        return np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0

VISUALZATION = False


if HAS_DIFFUSER:
    class Arguments(tap.Tap):
        """模型参数配置类"""
        cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
        image_size: str = "256,256"
        max_episodes_per_task: int = 20
        instructions: Optional[Path] = None
        seed: int = 0
        tasks: Tuple[str, ...] = ("pick_moving_target",)
        variations: Tuple[int, ...] = (0,)
        checkpoint: Optional[Path] = None
        accumulate_grad_batches: int = 1
        val_freq: int = 500
        gripper_loc_bounds: Optional[str] = None
        gripper_loc_bounds_buffer: float = 0.08
        eval_only: int = 0

        # Training and validation datasets
        dense_interpolation: int = 1
        interpolation_length: int = 2

        # Logging to base_log_dir/exp_log_dir/run_log_dir
        base_log_dir: Path = Path(__file__).parent / "train_logs"
        exp_log_dir: str = "exp"
        run_log_dir: str = "run"

        # Main training parameters
        num_workers: int = 1
        batch_size: int = 16
        batch_size_val: int = 4
        cache_size: int = 100
        cache_size_val: int = 100
        lr: float = 1e-4
        wd: float = 5e-3  # used only for CALVIN
        train_iters: int = 200_000
        val_iters: int = -1  # -1 means heuristically-defined
        max_episode_length: int = 5  # -1 for no limit

        # Data augmentations
        image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

        # Model
        backbone: str = "clip"  # one of "resnet", "clip"
        embedding_dim: int = 120
        num_vis_ins_attn_layers: int = 2
        use_instruction: int = 1
        rotation_parametrization: str = '6D'
        quaternion_format: str = 'xyzw'
        diffusion_timesteps: int = 100
        keypose_only: int = 1
        num_history: int = 3
        relative_action: int = 0
        lang_enhanced: int = 0
        fps_subsampling_factor: int = 5
else:
    # Fallback class when tap is not available
    class Arguments:
        """模型参数配置类（fallback版本）"""
        cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
        image_size: str = "256,256"
        max_episodes_per_task: int = 20
        instructions: Optional[Path] = None
        seed: int = 0
        tasks: Tuple[str, ...] = ("pick_moving_target",)
        variations: Tuple[int, ...] = (0,)
        checkpoint: Optional[Path] = None
        accumulate_grad_batches: int = 1
        val_freq: int = 500
        gripper_loc_bounds: Optional[str] = None
        gripper_loc_bounds_buffer: float = 0.08
        eval_only: int = 0
        dense_interpolation: int = 1
        interpolation_length: int = 2
        base_log_dir: Path = Path(__file__).parent / "train_logs"
        exp_log_dir: str = "exp"
        run_log_dir: str = "run"
        num_workers: int = 1
        batch_size: int = 16
        batch_size_val: int = 4
        cache_size: int = 100
        cache_size_val: int = 100
        lr: float = 1e-4
        wd: float = 5e-3
        train_iters: int = 200_000
        val_iters: int = -1
        max_episode_length: int = 5
        image_rescale: str = "0.75,1.25"
        backbone: str = "clip"
        embedding_dim: int = 120
        num_vis_ins_attn_layers: int = 2
        use_instruction: int = 1
        rotation_parametrization: str = '6D'
        quaternion_format: str = 'xyzw'
        diffusion_timesteps: int = 100
        keypose_only: int = 1
        num_history: int = 3
        relative_action: int = 0
        lang_enhanced: int = 0
        fps_subsampling_factor: int = 5
        
        def parse_args(self, args=None):
            """Fallback parse_args方法"""
            return self
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 20
    instructions: Optional[Path] = None
    seed: int = 0
    tasks: Tuple[str, ...] = ("pick_moving_target",)
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.08
    eval_only: int = 0

    # Training and validation datasets
    dense_interpolation: int = 1
    interpolation_length: int = 2

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'xyzw'
    diffusion_timesteps: int = 100
    keypose_only: int = 1
    num_history: int = 3
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5


def get_model(args: Arguments) -> Optional[Any]:
    """Initialize the model."""
    if not HAS_DIFFUSER:
        logging.warning("DiffuserActor not available, cannot initialize model")
        return None
    
    # Initialize model with arguments
    _model = DiffuserActor(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        use_instruction=bool(args.use_instruction),
        fps_subsampling_factor=args.fps_subsampling_factor,
        gripper_loc_bounds=args.gripper_loc_bounds,
        rotation_parametrization=args.rotation_parametrization,
        quaternion_format=args.quaternion_format,
        diffusion_timesteps=args.diffusion_timesteps,
        nhist=args.num_history,
        relative=bool(args.relative_action),
        lang_enhanced=bool(args.lang_enhanced)
    )
    print("Model parameters:", count_parameters(_model))
    return _model


class FakePolicy(_base_policy.BasePolicy):
    """3DA Policy实现，使用DiffuserActor模型"""
    
    def __init__(
        self, 
        camera_config: Optional[Dict] = None,
        checkpoint_path: Optional[Path] = None,
        args: Optional[Arguments] = None
    ):
        """
        初始化Policy
        
        Args:
            camera_config: 相机配置字典
            checkpoint_path: 模型checkpoint路径
            args: Arguments对象，如果为None则使用默认参数
        """
        self._camera_config = camera_config
        
        # 初始化参数
        if args is None:
            self.args = Arguments()
            if HAS_DIFFUSER:
                self.args = self.args.parse_args([])  # 使用默认参数
        else:
            self.args = args
        
        # 处理gripper_loc_bounds
        if self.args.gripper_loc_bounds is None:
            self.args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
        else:
            self.args.gripper_loc_bounds = get_gripper_loc_bounds(
                self.args.gripper_loc_bounds,
                task=self.args.tasks[0] if len(self.args.tasks) == 1 else None,
                buffer=self.args.gripper_loc_bounds_buffer,
            )
        
        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        if HAS_DIFFUSER:
            self.model = get_model(self.args)
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # 加载checkpoint
                if checkpoint_path is not None and checkpoint_path.exists():
                    logging.info(f"Loading checkpoint from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    logging.info("Checkpoint loaded successfully")
        
        logging.info(f"FakePolicy initialized (model available: {self.model is not None})")

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
        推理方法，使用DiffuserActor模型进行推理
        
        Args:
            obs: observation字典
            
        Returns:
            包含action的字典
        """
        obs = self.preprocess_obs(obs)

        # 保存obs到本地npy文件
        if VISUALZATION:
            import os
            os.makedirs("./src/temp", exist_ok=True)
            np.save("./src/temp/obs.npy", obs)
        
        # 如果模型可用，使用模型进行推理
        if self.model is not None:
            with torch.no_grad():
                # 准备输入（需要根据DiffuserActor的实际输入格式调整）
                # 这里假设模型需要特定的输入格式
                try:
                    # 调用模型推理（需要根据实际模型接口调整）
                    # action = self.model(obs)  # 示例，需要根据实际接口调整
                    # action = action.cpu().numpy()
                    # 暂时使用fallback
                    action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                                      0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                                      0.41908594192841403, 43.30252697496553])
                except Exception as e:
                    logging.error(f"Model inference failed: {e}, using fallback action")
                    action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                                      0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                                      0.41908594192841403, 43.30252697496553])
        else:
            # 模型不可用，返回fallback action
            action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                              0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                              0.41908594192841403, 43.30252697496553])
        
        return {
            "action": action
        }
    
    def reset(self) -> None:
        """重置policy状态"""
        pass