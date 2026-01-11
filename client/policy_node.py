"""策略服务器 - 运行在 conda_env_b (Python 3.10)"""
import logging
import os
import sys
import tap
import torch
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

DIFFUSER_HOME = "/home/extra3/yupeng/codespace/updated_3d_diffuser_actor"
sys.path.insert(0, str(DIFFUSER_HOME))
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v3 import ForesightDiffuserActorV3
# from utils.common_utils import get_gripper_loc_bounds
from utils.common_utils_deploy import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)

logging.basicConfig(level=logging.INFO, format='[SERVER] %(message)s')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from openpi.serving.ipc_core import BasePolicy, WebsocketPolicyServer, WebsocketRelayServer



class Arguments(tap.Tap):
    """模型参数配置类"""
    checkpoint: Path = "src/ckpts/epoch_84999.pth"
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
    denoise_model: str = "ddpm"  # "ddpm" or "rectified_flow"
    num_inference_steps: int = 100  # inference steps, RF uses 10

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'xyzw'



class PolicyServer(BasePolicy):
    """策略服务器实现"""
    def __init__(self):
        self._model = None
        self._device = None
        self._initialized = False
        self._init_error = None
        self._model_name = None

        args = Arguments().parse_args(args=[]) 
        init_result = self.initialize(args)

        if init_result["status"] == "error":
            # 根据需求选择：是抛出异常还是仅仅记录日志
            logging.error(f"构造函数中的自动初始化失败: {init_result['message']}")

    # def initialize(self, model_name: str, model_args: Dict, checkpoint: str, device: str = "cuda") -> Dict:
    #     """初始化模型
        
    #     Args:
    #         model_name: 模型名称 ("3d_diffuser_actor" 或 "foresight_diffuser_actor_v3")
    #         model_args: 模型参数字典
    #         checkpoint: checkpoint 路径
    #         device: 设备 ("cuda" 或 "cpu")
        
    #     Returns:
    #         包含初始化状态的字典
    #     """
    #     try:
    #         self._device = torch.device(device)
    #         self._model_name = model_name
    #         logging.info(f"初始化模型: {model_name}, checkpoint: {checkpoint}")
            
    #         # 创建模型
    #         if model_name == "3d_diffuser_actor":
    #             self._model = DiffuserActor(
    #                 backbone=model_args['backbone'],
    #                 image_size=tuple(model_args['image_size']),
    #                 embedding_dim=model_args['embedding_dim'],
    #                 num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
    #                 use_instruction=model_args['use_instruction'],
    #                 fps_subsampling_factor=model_args['fps_subsampling_factor'],
    #                 gripper_loc_bounds=model_args['gripper_loc_bounds'],
    #                 rotation_parametrization=model_args['rotation_parametrization'],
    #                 quaternion_format=model_args['quaternion_format'],
    #                 diffusion_timesteps=model_args['diffusion_timesteps'],
    #                 denoise_model=model_args['denoise_model'],
    #                 num_inference_steps=model_args['num_inference_steps'],
    #                 nhist=model_args['nhist'],
    #                 relative=model_args['relative'],
    #                 lang_enhanced=model_args['lang_enhanced'],
    #             ).to(self._device)
    #         elif model_name == "foresight_diffuser_actor_v3":
    #             self._model = ForesightDiffuserActorV3(
    #                 backbone=model_args['backbone'],
    #                 image_size=tuple(model_args['image_size']),
    #                 embedding_dim=model_args['embedding_dim'],
    #                 num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
    #                 use_instruction=model_args['use_instruction'],
    #                 fps_subsampling_factor=model_args['fps_subsampling_factor'],
    #                 gripper_loc_bounds=model_args['gripper_loc_bounds'],
    #                 rotation_parametrization=model_args['rotation_parametrization'],
    #                 quaternion_format=model_args['quaternion_format'],
    #                 diffusion_timesteps=model_args['diffusion_timesteps'],
    #                 denoise_model=model_args['denoise_model'],
    #                 num_inference_steps=model_args['num_inference_steps'],
    #                 nhist=model_args['nhist'],
    #                 relative=model_args['relative'],
    #                 lang_enhanced=model_args['lang_enhanced'],
    #                 bool_classifier_free_guidance=model_args['bool_classifier_free_guidance'],
    #                 classifier_free_guidance_w=model_args['classifier_free_guidance_w'],
    #                 classifier_free_guidance_dropout_prob=model_args['classifier_free_guidance_dropout_prob'],
    #             ).to(self._device)
    #         else:
    #             raise ValueError(f"不支持的模型类型: {model_name}，仅支持 '3d_diffuser_actor' 和 'foresight_diffuser_actor_v3'")
            
    #         # 加载 checkpoint
    #         # 将相对路径转换为绝对路径（相对于项目根目录）
    #         checkpoint_path = Path(checkpoint)
    #         if not checkpoint_path.is_absolute():
    #             checkpoint_path = PROJECT_ROOT / checkpoint_path
            
    #         if checkpoint_path.is_file():
    #             model_dict = torch.load(str(checkpoint_path), map_location="cpu")
    #             model_dict_weight = {}
    #             for key in model_dict["weight"]:
    #                 _key = key[7:]  # 移除 "module." 前缀
    #                 model_dict_weight[_key] = model_dict["weight"][key]
    #             self._model.load_state_dict(model_dict_weight)
    #             logging.info(f"模型权重加载成功: {checkpoint_path}")
    #         else:
    #             assert False, f"Checkpoint 文件不存在: {checkpoint_path}"
            
    #         # 设置为 eval 模式
    #         self._model.eval()
    #         self._initialized = True
    #         self._init_error = None
            
    #         logging.info("模型初始化完成")
    #         return {
    #             "status": "success",
    #             "message": "模型初始化成功",
    #             "model_name": model_name,
    #             "device": str(self._device)
    #         }
    #     except Exception as e:
    #         self._initialized = False
    #         self._init_error = str(e)
    #         logging.error(f"模型初始化失败: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return {
    #             "status": "error",
    #             "message": f"模型初始化失败: {str(e)}",
    #             "error": str(e)
    #         }
    
    def initialize(self, args: Arguments) -> Dict:
        """使用 Arguments 对象初始化模型"""
        try:
            self._device = torch.device(args.device)
            self._model_name = args.test_model
            logging.info(f"初始化模型: {self._model_name}, checkpoint: {args.checkpoint}")
            
            # 处理 image_size (从 "256,256" 字符串转为 tuple)
            img_size = tuple(map(int, args.image_size.split(',')))

            if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
                task = args.tasks[0]
            else:
                task = None


            gripper_loc_bounds = get_gripper_loc_bounds(
                args.gripper_loc_bounds_file,
                task=task, buffer=args.gripper_loc_bounds_buffer,
            )
            
            # 提取通用的基础参数
            base_kwargs = {
                "backbone": args.backbone,
                "image_size": img_size,
                "embedding_dim": args.embedding_dim,
                "num_vis_ins_attn_layers": args.num_vis_ins_attn_layers,
                "use_instruction": bool(args.use_instruction),
                "fps_subsampling_factor": args.fps_subsampling_factor,
                "gripper_loc_bounds": gripper_loc_bounds, # 注意这里通常传路径或预加载的bounds
                "rotation_parametrization": args.rotation_parametrization,
                "quaternion_format": args.quaternion_format,
                "diffusion_timesteps": args.diffusion_timesteps,
                "denoise_model": args.denoise_model,
                "num_inference_steps": args.num_inference_steps,
                "nhist": args.num_history,
                "relative": bool(args.relative_action),
                "lang_enhanced": bool(args.lang_enhanced),
            }

            # 创建模型
            if self._model_name == "3d_diffuser_actor":
                self._model = DiffuserActor(**base_kwargs).to(self._device)
                
            elif self._model_name == "foresight_diffuser_actor_v3":
                # 扩展 v3 特有的参数
                v3_kwargs = {
                    **base_kwargs,
                    "bool_classifier_free_guidance": getattr(args, 'bool_classifier_free_guidance', True),
                    "classifier_free_guidance_w": getattr(args, 'classifier_free_guidance_w', 0.1),
                    "classifier_free_guidance_dropout_prob": getattr(args, 'classifier_free_guidance_dropout_prob', 0.1),
                }
                self._model = ForesightDiffuserActorV3(**v3_kwargs).to(self._device)
            else:
                raise ValueError(f"不支持的模型: {self._model_name}")

            # 加载 checkpoint (处理 Path 对象)
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                # 假设 PROJECT_ROOT 已在外部定义
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            
            if checkpoint_path.is_file():
                model_dict = torch.load(str(checkpoint_path), map_location="cpu")
                # 兼容 DataParallel 的 "module." 前缀
                weights = model_dict.get("weight", model_dict) # 增加容错
                new_weights = {k.replace("module.", ""): v for k, v in weights.items()}
                
                self._model.load_state_dict(new_weights)
                logging.info(f"成功加载权重: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

            self._model.eval()
            self._initialized = True
            
            return {
                "status": "success",
                "model_name": self._model_name,
                "device": str(self._device)
            }

        except Exception as e:
            self._initialized = False
            logging.error(f"初始化失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def infer(self, obs: Dict) -> Dict:
        """推理接口
        
        Args:
            obs: 包含以下字段的字典:
                - "fake_traj": torch.Tensor
                - "traj_mask": torch.Tensor
                - "rgbs": torch.Tensor
                - "pcds": torch.Tensor
                - "instr": torch.Tensor
                - "gripper": torch.Tensor
                - "run_inference": bool
                - 其他 kwargs
        
        Returns:
            包含 "action" 字段的字典
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化")
        
        if self._model is None:
            raise RuntimeError("模型未创建")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        print("Run inference ... ")
        # 提取参数
        fake_traj = obs["fake_traj"]
        traj_mask = obs["traj_mask"]
        rgbs = obs["rgbs"]
        pcds = obs["pcds"]
        instr = obs["instr"]
        gripper = obs["gripper"]
        run_inference = obs.get("run_inference", True)
        
        # 提取其他 kwargs（包括 next_rgb_obs, next_pcd_obs, next_mask_obs, next_gripper, next_frame_relative_id 等）
        kwargs = {k: v for k, v in obs.items() 
                 if k not in ["fake_traj", "traj_mask", "rgbs", "pcds", "instr", "gripper", "run_inference"]}
        
        # 确保张量在正确的设备上
        fake_traj = fake_traj.to(self._device)
        traj_mask = traj_mask.to(self._device)
        rgbs = rgbs.to(self._device)
        pcds = pcds.to(self._device)
        instr = instr.to(self._device)
        gripper = gripper.to(self._device)
        
        # 转换 kwargs 中的张量
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self._device)
            elif v is None:
                # 保持 None 值不变（如 next_mask_obs 可能为 None）
                pass
        
        # 推理
        with torch.no_grad():
            output_dict = self._model(
                fake_traj,
                traj_mask,
                rgbs,
                pcds,
                instr,
                gripper,
                run_inference=run_inference,
                **kwargs
            )
        
        # 返回结果（确保 action 在 CPU 上以便序列化）
        result = {
            "action": output_dict["action"].cpu() if isinstance(output_dict["action"], torch.Tensor) else output_dict["action"]
        }
        
        # 如果 output_dict 中有其他字段，也返回
        for k, v in output_dict.items():
            if k != "action":
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu()
                else:
                    result[k] = v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[SERVER] infer: {(time.time() - t0)*1000:.2f}ms")
        print("Inference done ... ")
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8766, help="服务器端口")
    args = parser.parse_args()
    
    policy = PolicyServer()
    server = WebsocketRelayServer(policy, host=args.host, port=args.port, 
                                  target_host="127.0.0.1", target_port=8767)
    print("🚀 策略服务器运行中...")
    server.serve_forever()
