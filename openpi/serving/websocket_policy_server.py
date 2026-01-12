import asyncio
import http
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
import websockets

# # 添加项目路径
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        config: dict = None,
        metadata: dict = None,
    ) -> None:
        """
        初始化WebsocketPolicyServer
        
        Args:
            policy: BasePolicy实例
            config: 配置文件
            host: 服务器host（如果提供config_path，此参数将被覆盖）
            port: 服务器port（如果提供config_path，此参数将被覆盖）
            metadata: 服务器元数据
        """
        self._policy = policy
        
        # 如果提供了config_path，从配置文件加载
        if config is not None:
            self.host = config.get("host", "0.0.0.0")
            self.port = config.get("port", 8000)
            self.api_key = config.get("api_key", None)
            self.policy_config = config.get("policy", {})
            self.model_type = self.policy_config.get("model_type", "default")
        else:
            # 使用传入的参数或默认值
            self.host = host or "0.0.0.0"
            self.port = port
            self.api_key = None
            self.policy_config = {}
            self.model_type = "default"
        
        self._metadata = metadata or {}
        
        # 初始化policy model和world model（框架阶段）
        self.policy_model = None
        self.world_model = None
        
        # 框架阶段：计数器，用于决定是否返回action（仅用于演示）
        self._request_count = 0
        
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        logging.info(f"WebsocketPolicyServer initialized (model_type={self.model_type}, host={self.host}, port={self.port})")

    def handle_request(self, obs: Dict) -> Dict:
        """
        处理请求，返回响应
        
        Args:
            obs: observation字典
            
        Returns:
            响应字典，包含type字段（"update_world_model"或"action"）
        """
        if self.model_type == "3dafdp":
            # 3dafdp模型：先更新world model，然后决定是否返回action
            self._update_world_model(obs)
            
            # 根据内部逻辑决定是否返回action（框架阶段使用简单逻辑）
            if self._should_return_action():
                action = self._policy_model_infer(obs)
                return {
                    "type": "action",
                    "arm_gripper_pos": action
                }
            else:
                return {
                    "type": "update_world_model",
                    "status": "ok"
                }
        else:
            # 标准模型：直接返回action
            action = self._policy_model_infer(obs)
            return {
                "type": "action",
                "arm_gripper_pos": action
            }
    
    def _update_world_model(self, obs: Dict) -> None:
        """
        更新World Model（框架阶段）
        
        Args:
            obs: observation字典
        """
        # 框架阶段：占位函数
        # 实际实现时需要调用world_model.update()
        logging.debug("Updating world model")
        pass
    
    def _policy_model_infer(self, obs: Dict) -> np.ndarray:
        """
        Policy模型推理
        
        Args:
            obs: observation字典
            
        Returns:
            joint angles数组
        """
        # 调用policy的infer方法
        action_dict = self._policy.infer(obs)
        
        # 如果返回的是字典，尝试提取joint_angles
        if isinstance(action_dict, dict):
            if "joint_angles" in action_dict:
                return action_dict["joint_angles"]
            # 如果没有joint_angles，尝试其他可能的键
            for key in ["action", "actions", "angles"]:
                if key in action_dict:
                    return np.array(action_dict[key], dtype=np.float32)
            # 如果都没有，返回零数组作为占位符
            num_joints = 7  # 假设7个关节（实际应该从配置或模型中获取）
            return np.zeros(num_joints, dtype=np.float32)
        else:
            # 如果返回的是数组，直接返回
            return np.array(action_dict, dtype=np.float32)
    
    def _should_return_action(self) -> bool:
        """
        判断是否应该返回action（框架阶段）
        
        Returns:
            是否返回action
        """
        # 框架阶段：简单逻辑，每3次请求返回一次action
        # 实际实现时需要根据world model的状态决定
        self._request_count += 1
        return self._request_count % 3 == 0

    def serve(self) -> None:
        """启动服务器"""
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # 使用handle_request方法处理请求
                response = self.handle_request(obs)
                infer_time = time.monotonic() - infer_time

                # 如果返回的是action类型，添加timing信息
                if response.get("type") == "action":
                    response["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        response["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                else:
                    # 对于update_world_model类型，也添加timing信息
                    response["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
