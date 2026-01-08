"""最小化策略服务器 - 运行在 conda_env_b (Python 3.10)"""
import logging
import time

import numpy as np

from ipc_core import BasePolicy, WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format='[SERVER] %(message)s')

class SimplePolicy(BasePolicy):
    """简单的模拟策略"""
    def __init__(self):
        logging.info("初始化策略...")
        time.sleep(0.5)  # 模拟加载时间
        logging.info("策略就绪")

    def infer(self, obs: dict) -> dict:
        """接收 image 和 pcd，返回 action"""
        infer_start = time.time()
        img = obs["image"]  # (H, W, 3) numpy array
        pcd = obs["pcd"]    # (N, 3) numpy array
        
        # 模拟推理时间（可以调整或移除来测试纯通信延迟）
        # 注释掉 sleep 来测试纯通信延迟，或调整值来模拟不同的推理时间
        # time.sleep(0.02)  # 20ms 模拟推理
        
        # 返回 action (7DOF 关节角度 + 夹爪)
        action = np.random.randn(7).astype(np.float32)
        
        infer_time = (time.time() - infer_start) * 1000
        if hasattr(self, '_step'):
            self._step += 1
        else:
            self._step = 1
        
        if self._step % 100 == 0:
            logging.info(f"推理时间: {infer_time:.2f} ms")
        
        return {
            "joint_pos": action,
            "gripper": 1.0
        }

if __name__ == "__main__":
    policy = SimplePolicy()
    server = WebsocketPolicyServer(policy, host="0.0.0.0", port=8765)
    print("🚀 服务器运行中...")
    server.serve_forever()

