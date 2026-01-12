"""Policy Server

统一WebSocket服务器，接收policy_model_node的请求，
根据模型类型和内部逻辑返回不同类型的响应（update_world_model或action）。

PolicyServer 是对 WebsocketPolicyServer 的配置包装器。
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_server_config, get_camera_config


from openpi.serving.websocket_policy_server import WebsocketPolicyServer
# from openpi_client import base_policy as _base_policy
# from openpi.policy.fake_policy import FakePolicy
from openpi.policy.da_policy import DAPolicy
HAS_OPENPI = True

# 尝试导入websocket server（如果可用）
try:
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    # from openpi_client import base_policy as _base_policy
    # from openpi.policy.fake_policy import FakePolicy
    from openpi.policy.da_policy import DAPolicy
    from openpi.policy.expert_policy import ExpertPolicy
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    WebsocketPolicyServer = None
    _base_policy = None
    logging.warning("openpi.serving not available, using basic WebSocket server")

print(f"HAS_OPENPI = {HAS_OPENPI}")

class PolicyServer:
    """Policy Server类 - WebsocketPolicyServer的配置包装器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化Policy Server
        
        Args:
            config_path: 配置文件路径
        """
        if not HAS_OPENPI:
            raise ImportError("openpi.serving.websocket_policy_server is required but not available")
        
        # 加载配置
        server_config = get_server_config(config_path)
        self._config = server_config
        self._policy_config = server_config.get("policy", {})
        self._model_type = self._policy_config.get("model_type", "default")

        # Load camera configuration
        self._cameras_config = get_camera_config(config_path)
        
        logging.info(f"PolicyServer initialized (model_type={self._model_type}, host={server_config.get('host')}, port={server_config.get('port')})")
    
    def serve(self):
        """启动服务器"""
        # 创建假的policy对象
        # fake_policy = FakePolicy(camera_config=self._cameras_config)
        da_policy = DAPolicy(camera_config=self._cameras_config)
        # da_policy = ExpertPolicy(camera_config=self._cameras_config)
        
        # 创建WebsocketPolicyServer实例（它会处理所有逻辑）
        ws_server = WebsocketPolicyServer(
            policy=da_policy,
            config=self._config,
            metadata={"model_type": self._model_type}
        )
        
        # 启动服务器
        logging.info("Starting WebsocketPolicyServer...")
        ws_server.serve()


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

