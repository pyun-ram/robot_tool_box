"""WorldModel 服务器 - 运行在 conda_env_b (Python 3.10)"""
import logging
import os
import sys
import torch
import time
from pathlib import Path
from typing import Dict, Optional
DIFFUSER_HOME = "/home/extra3/yupeng/codespace/updated_3d_diffuser_actor"
sys.path.insert(0, str(DIFFUSER_HOME))

logging.basicConfig(level=logging.INFO, format='[WORLD_SERVER] %(message)s')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from openpi.serving.ipc_core import BasePolicy, WebsocketPolicyServer, WebsocketRelayServer


class DummyWorldModelServer(BasePolicy):
    """WorldModel 调试伪类：返回固定格式的 Mock 数据"""
    def __init__(self):
        self._initialized = False
        self._device = "cpu"
        logging.info("DummyWorldModelServer 已创建 (调试模式)")

    def initialize(self, **kwargs) -> Dict:
        """模拟初始化，保存参数并返回成功"""
        self._initialized = True
        logging.info("DummyWorldModelServer 初始化成功")
        return {
            "status": "success",
            "message": "DummyWorldModel 配置初始化成功",
            "device": "cpu"
        }

    def infer(self, request: Dict) -> Dict:
        """模拟推理路由"""
        if not self._initialized:
            raise RuntimeError("WorldModel 配置未初始化")
        
        request_type = request.get("type")
        # 模拟计算延迟
        time.sleep(0.05) 
        
        if request_type == "init":
            return self._handle_init(request)
        elif request_type == "update":
            return self._handle_update(request)
        elif request_type == "predict":
            return self._handle_predict(request)
        else:
            raise ValueError(f"未知的请求类型: {request_type}")

    def _handle_init(self, request: Dict) -> Dict:
        """返回固定的 init 结果"""
        logging.info("[Dummy] 处理 init 请求")
        # 模拟一个 1x10 的 fake tensor 结构
        fake_network_t = np.zeros((1, 10), dtype=np.float32)
        
        return {
            "status": "success",
            "message": "Dummy WorldModel init 成功",
            "network_t": fake_network_t,
        }

    def _handle_update(self, request: Dict) -> Dict:
        """返回固定的 update 结果"""
        logging.info(f"[Dummy] 处理 update 请求: cur_t={request.get('cur_t')}")
        fake_network_t = np.zeros((1, 10), dtype=np.float32)
        
        return {
            "status": "success",
            "message": "Dummy WorldModel update 成功",
            "network_t": fake_network_t,
        }

    def _handle_predict(self, request: Dict) -> Dict:
        """返回固定长度的列表和数据"""
        num_frames = request.get("num_future_frames", 30)
        # logging.info(f"[Dummy] 处理 predict 请求: num_frames={num_frames}")
        logging.info(request)

        return {
            "status": "success",
            "image_list": [fake_img for _ in range(num_frames)],
            "depth_list": [fake_depth for _ in range(num_frames)],
            "semantic_mask_list": None,
            "target_time_list": [float(i) for i in range(num_frames)],
            "eepose_list": [fake_pose for _ in range(num_frames)],
            "openness_list": [1.0 for _ in range(num_frames)],
        }

    def infer(self, request: Dict) -> Dict:
        """处理推理请求
        
        Args:
            request: 包含以下字段的字典:
                - "type": "init" | "update" | "predict"
                - 其他字段根据类型不同而不同
        
        Returns:
            包含结果的字典
        """
        if not self._initialized:
            raise RuntimeError("WorldModel 配置未初始化")
        
        request_type = request.get("type")
        

        return self._handle_predict(request)



if __name__ == "__main__":
    import argparse
    #disable_deterministic_algorithms()    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8767, help="服务器端口")
    args = parser.parse_args()
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('medium')
    world_model_server = DummyWorldModelServer()
    server = WebsocketPolicyServer(world_model_server, host=args.host, port=args.port)
    print("🚀 WorldModel 服务器运行中...")
    server.serve_forever()
