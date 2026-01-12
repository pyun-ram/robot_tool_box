import os
import sys
import tap
import logging
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
from openpi_client import base_policy as _base_policy

# 尝试导入tap和DiffuserActor
DIFFUSER_HOME = "/home/extra3/yupeng/codespace/updated_3d_diffuser_actor"
sys.path.insert(0, str(DIFFUSER_HOME))
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from utils.common_utils_deploy import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)


try:
    DIFFUSER_HOME = "/home/extra3/yupeng/codespace/updated_3d_diffuser_actor"
    sys.path.insert(0, str(DIFFUSER_HOME))
    from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
    from utils.common_utils_deploy import (
        load_instructions,
        get_gripper_loc_bounds,
        round_floats
    )
    HAS_DIFFUSER = True
except ImportError:
    HAS_DIFFUSER = False
    logging.warning("tap or DiffuserActor not available, using fallback")

print(f'HAS_DIFFUSER: {HAS_DIFFUSER}')
# # 尝试导入其他可能需要的函数
# try:
#     from utils import count_parameters, get_gripper_loc_bounds
# except ImportError:
#     def count_parameters(model):
#         """计算模型参数数量"""
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     def get_gripper_loc_bounds(bounds_str, task=None, buffer=0.08):
#         """解析gripper位置边界"""
#         # 简单实现，实际应该根据bounds_str解析
#         return np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0

VISUALZATION = False

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

class Arguments(tap.Tap):
    """模型参数配置类"""
    checkpoint: Path = "src/ckpts/epoch_94999.pth"
    seed: int = 2
    device: str = "cuda"
    headless: int = 0
    max_tries: int = 10
    tasks: Tuple[str, ...] = ("pick_moving_target_from_belt",)
    image_size: str = "256,256"
    verbose: int = 0
    max_episodes_per_task: int = 20
    instructions: Optional[Path] = "src/instructions/rmt/rmt_instructions_real_world_dyna5task.pkl"
    variations: Tuple[int, ...] = (-1,)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    gripper_loc_bounds_file: str = "/home/extra3/yupeng/codespace/updated_3d_diffuser_actor/tasks/18_peract_tasks_location_bounds.json"
    gripper_loc_bounds_buffer: float = 0.08
    single_task_gripper_loc_bounds: int = 0
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

    # Act3D model parameters
    num_query_cross_attn_layers: int = 2
    num_ghost_point_cross_attn_layers: int = 2
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    regress_position_offset: int = 0

    # 3D Diffuser Actor model parameters
    diffusion_timesteps: int = 100
    num_history: int = 3
    num_future_frames: int = 10
    fps_subsampling_factor: int = 5
    lang_enhanced: int = 0
    dense_interpolation: int = 1
    interpolation_length: int = 2
    relative_action: int = 0
    denoise_model: str = "rectified_flow"  # "ddpm" or "rectified_flow"
    num_inference_steps: int = 10  # inference steps, RF uses 10

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'xyzw'

def get_model(args: Arguments) -> Optional[Any]:
    """Initialize the model."""
    if not HAS_DIFFUSER:
        logging.warning("DiffuserActor not available, cannot initialize model")
        return None

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None

    print('Gripper workspace')
    # gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )

    
    # Initialize model with arguments
    if args.test_model == "3d_diffuser_actor":
        _model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            quaternion_format=args.quaternion_format,
            diffusion_timesteps=args.diffusion_timesteps,
            denoise_model=args.denoise_model,
            num_inference_steps=args.num_inference_steps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced)
        )
    else:
        raise NotImplementedError

    return _model


class DAPolicy(_base_policy.BasePolicy):
    """3DA Policy实现，使用DiffuserActor模型"""
    
    def __init__(
        self, 
        camera_config: Optional[Dict] = None,
        args: Optional[Arguments] = None
    ):
        """
        初始化Policy
        
        Args:
            camera_config: 相机配置字典
            args: Arguments对象，如果为None则使用默认参数
        """
        self._camera_config = camera_config
        
        # 初始化参数

        self.args = Arguments()
        if HAS_DIFFUSER:
            self.args = self.args.parse_args([])  # 使用默认参数
        
        # 初始化模型
        self.device = torch.device(self.args.device)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if HAS_DIFFUSER:
            self.model = get_model(self.args)
            
            self.model = self.model.to(self.device)

            # Load model weights
            if Path(self.args.checkpoint).is_file():
                model_dict = torch.load(self.args.checkpoint, map_location="cpu")
                model_dict_weight = {}
                for key in model_dict["weight"]:
                    _key = key[7:]
                    model_dict_weight[_key] = model_dict["weight"][key]
                self.model.load_state_dict(model_dict_weight, strict=True)
            else:
                print("Do not load model weights from", self.args.checkpoint)
            self.model.eval()
        
        # mask
        self.tra_mask = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        
        # instruction
        if self.args.instructions:
        # if False:
            data = load_instructions(self.args.instructions)
            self.instr = data[self.args.tasks[0]][0].to(self.device)
            # self.instr = torch.load(self.args.instructions).to(self.device)
        else:
            self.instr = torch.zeros(1, 53, 512).to(self.device)
        
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
                    vv = v.astype(np.float32, copy=False) / 255.0
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

    @staticmethod
    def _to_channel_first_c3hw(x: torch.Tensor, name: str) -> torch.Tensor:
        """
        把图像/点云从 (C,H,W,3) 变成 (C,3,H,W)。
        如果已经是 (C,3,H,W) 就原样返回。
        """
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(f"Expected {name} shape (C,*,*,*), got {tuple(x.shape)}")

        # (C,3,H,W)
        if x.shape[1] == 3:
            return x

        # (C,H,W,3)
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"Cannot convert {name} to (C,3,H,W), got {tuple(x.shape)}")

    # ---------- B) 通用工具：给所有 tensor 加 batch 维 (B=1) ----------
    @staticmethod
    def _ensure_batch_dim(obs: Dict[str, Any], B: int = 1) -> Dict[str, Any]:
        """
        obs 中所有 torch.Tensor：如果最前面不是 batch 维，就在最前面 unsqueeze(0)。
        这里采用“若 x.shape[0] != B 则加一维”的简单策略，适配你要求 B=1 的场景。
        """
        assert B == 1, "当前实现按你的需求固定 B=1（可自行扩展）"

        for k, v in list(obs.items()):
            if torch.is_tensor(v):
                # 已经是 B=1 就不动；否则在最前面加一维
                if v.ndim == 0:
                    obs[k] = v.view(1)
                elif v.shape[0] == B:
                    obs[k] = v
                else:
                    obs[k] = v.unsqueeze(0)
        return obs


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

    # ---------- C) 抽出来：depth(mm)->点云(world)，输出 (C,3,H,W) ----------
    def _depth_mm_to_world_points(self, depth_mm: torch.Tensor) -> torch.Tensor:
        """
        输入：
          depth_mm: (C,H,W) CUDA tensor, dtype 可为 uint16/float 等，单位 mm
        输出：
          points_world: (C,3,H,W) float32 CUDA tensor，world 坐标系点云
        """
        if not torch.is_tensor(depth_mm):
            raise TypeError("depth_mm must be a torch.Tensor")
        if depth_mm.ndim != 3:
            raise ValueError(f"Expected depth shape (C,H,W), got {tuple(depth_mm.shape)}")

        device = depth_mm.device
        C, H, W = depth_mm.shape

        # mm -> m, float32
        Z = depth_mm.to(dtype=torch.float32).mul_(0.001)  # (C,H,W)
        valid = Z > 0.0

        # cache: uu/vv/cx/cy/inv_fx/inv_fy/R/t
        if (not hasattr(self, "_uu")) or (self._uu.shape[-2:] != (H, W)) or (self._uu.device != device):
            self._prepare_cam_cache_torch(H, W, device)

        # (C,H,W)
        X = (self._uu - self._cx) * Z * self._inv_fx
        Y = (self._vv - self._cy) * Z * self._inv_fy

        # (C,HW,3)
        P = torch.stack([X, Y, Z], dim=-1).reshape(C, -1, 3)

        # world: Pw = P @ R^T + t
        Pw = torch.bmm(P, self._R.transpose(1, 2)) + self._t[:, None, :]  # (C,HW,3)

        # (C,H,W,3) -> (C,3,H,W)
        points_world = Pw.reshape(C, H, W, 3).permute(0, 3, 1, 2).contiguous()

        # mask 无效点
        points_world = points_world * valid[:, None, :, :].to(points_world.dtype)

        return points_world

    # ---------- 3) preprocess_obs：整合 + 换通道 + 加 batch ----------
    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        目标：
          - obs['rgbs'] : (1,C,3,H,W)
          - obs['pcds'] : (1,C,3,H,W)  (world points)
          - obs 里所有 tensor 都加 B=1 在最前面
        """
        # 先转 tensor & 上 GPU（你已有）
        obs = self.obs_to_tensor(obs)

        # 1) pcds: depth(mm) -> world points (C,3,H,W)
        if "pcds" not in obs:
            raise KeyError("obs missing key 'pcds'")
        obs["pcds"] = self._depth_mm_to_world_points(obs["pcds"])

        # 2) rgbs: -> (C,3,H,W)
        if "rgbs" in obs and torch.is_tensor(obs["rgbs"]):
            obs["rgbs"] = self._to_channel_first_c3hw(obs["rgbs"], "obs['rgbs']")

        # 3) 给所有 tensor 加 batch 维：B=1
        obs = self._ensure_batch_dim(obs, B=1)

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
        
        # 如果模型可用，使用模型进行推理
        if self.model is not None:
            with torch.no_grad():
                # 准备输入（需要根据DiffuserActor的实际输入格式调整）
                # 这里假设模型需要特定的输入格式
                try:
                    # 调用模型推理（需要根据实际模型接口调整）
                    # action = self.model(obs)  

                    # input_dict = {
                    #     "trajectory_mask": self.tra_mask,
                    #     "rgb_obs": obs["rgbs"],
                    #     "pcd_obs": obs["pcds"],
                    #     "instruction": self.instr,
                    #     "curr_gripper": obs["curr_gripper_history"],
                    #     # 你也可以把其它需要复现推理的东西加进来
                    # }

                    # # 2) 递归转 numpy
                    # input_dict_np = to_numpy_tree(input_dict)

                    # # 3) 存成 .npy（注意：这是 pickle 形式）
                    # import time
                    # save_path = Path(f"model_inputs{time.time()}.npy")
                    # np.save(save_path, np.array(input_dict_np, dtype=object), allow_pickle=True)
                    
                    output_dict = self.model(
                        gt_trajectory=None,
                        trajectory_mask=self.tra_mask,
                        rgb_obs=obs['rgbs'],
                        pcd_obs=obs['pcds'],
                        instruction=self.instr,
                        curr_gripper=obs['curr_gripper_history'],
                        run_inference=True,
                    )
                    action = output_dict["action"].cpu().numpy().squeeze()
                    logging.info(action)
                    # 暂时使用fallback
                    # action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                    #                   0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                    #                   0.41908594192841403, 43.30252697496553])
                except Exception as e:
                    logging.error(obs['rgbs'].shape)
                    logging.error(obs['pcds'].shape)
                    logging.error(obs['curr_gripper'].shape)
                    logging.error(f"Model inference failed: {e}, using fallback action")
                    action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                                      0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                                      0.41908594192841403, 43.30252697496553])
        else:
            # 模型不可用，返回fallback action
            action = np.array([0.4616922824963358, -0.16665824442351365, 0.9153847316042458, 
                              0.00043769274192836314, 0.016118425994695856, 0.9078033807199066, 
                              0.41908594192841403, 43.30252697496553])
        
        # 保存obs到本地npy文件
        if VISUALZATION:
            obs['effector_pos'] = action[:3]
            import os
            os.makedirs("./src/temp", exist_ok=True)
            np.save("./src/temp/obs.npy", obs)

        return {
            "action": action
        }
    
    def reset(self) -> None:
        """重置policy状态"""
        pass