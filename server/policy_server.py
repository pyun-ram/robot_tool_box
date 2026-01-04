"""Policy Server

统一WebSocket服务器，接收policy_model_node的请求，
根据模型类型和内部逻辑返回不同类型的响应（update_world_model或action）。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_server_config

# 尝试导入websocket server（如果可用）
try:
    from openpi.serving import websocket_policy_server
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    logging.warning("openpi.serving not available, using basic WebSocket server")


class PolicyServer:
    """Policy Server类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化Policy Server
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        server_config = get_server_config(config_path)
        self.host = server_config.get("host", "0.0.0.0")
        self.port = server_config.get("port", 8000)
        self.api_key = server_config.get("api_key")
        self.policy_config = server_config.get("policy", {})
        self.model_type = self.policy_config.get("model_type", "default")
        
        # 初始化policy model和world model（框架阶段）
        self.policy_model = None
        self.world_model = None
        
        # 框架阶段：计数器，用于决定是否返回action（仅用于演示）
        self._request_count = 0
        
        logging.info(f"Policy server initialized (model_type={self.model_type}, host={self.host}, port={self.port})")
    
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
                    "joint_angles": action
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
                "joint_angles": action
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
        Policy模型推理（框架阶段）
        
        Args:
            obs: observation字典
            
        Returns:
            joint angles数组
        """
        # 框架阶段：返回零数组作为占位符
        # 实际实现时需要调用policy_model.infer()
        num_joints = 7  # 假设7个关节（实际应该从配置或模型中获取）
        return np.zeros(num_joints, dtype=np.float32)
    
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
    
    def serve(self):
        """启动服务器"""
        if HAS_OPENPI:
            # 使用openpi的WebSocket服务器（需要实现Policy接口）
            # 框架阶段使用基本实现
            logging.info("Using openpi WebSocket server (framework mode)")
            # 这里需要实现BasePolicy接口
            # 暂时使用基本实现
            self._serve_basic()
        else:
            # 使用基本WebSocket服务器
            self._serve_basic()
    
    def _serve_basic(self):
        """
        基本WebSocket服务器实现（框架阶段）
        
        注意：实际部署时应该使用openpi的WebSocket服务器
        """
        try:
            import websockets
            import asyncio
            
            async def handler(websocket, path):
                logging.info(f"Client connected: {websocket.remote_address}")
                try:
                    async for message in websocket:
                        # 解析消息（框架阶段，实际需要使用msgpack_numpy）
                        try:
                            import json
                            obs = json.loads(message)
                        except Exception:
                            # 如果是二进制消息，需要msgpack解析
                            logging.error("Binary message parsing not implemented in framework")
                            continue
                        
                        # 处理请求
                        response = self.handle_request(obs)
                        
                        # 发送响应（框架阶段使用JSON）
                        try:
                            response_json = json.dumps(response, default=str)
                            await websocket.send(response_json)
                        except Exception as e:
                            logging.error(f"Error sending response: {e}")
                except websockets.exceptions.ConnectionClosed:
                    logging.info("Client disconnected")
                except Exception as e:
                    logging.error(f"Error in handler: {e}")
            
            async def main():
                async with websockets.serve(handler, self.host, self.port):
                    logging.info(f"Policy server started on ws://{self.host}:{self.port}")
                    await asyncio.Future()  # 永远运行
            
            asyncio.run(main())
        except ImportError:
            logging.error("websockets not available. Please install: pip install websockets")
            raise
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            raise


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, force=True)
    
    try:
        server = PolicyServer()
        server.serve()
    except KeyboardInterrupt:
        logging.info("Server stopped")
    except Exception as e:
        logging.error(f"Server error: {e}")
        raise


if __name__ == '__main__':
    main()

