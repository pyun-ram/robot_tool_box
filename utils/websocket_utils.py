"""WebSocket客户端工具模块

封装WebSocket连接，用于与Policy Server通信。
"""

import logging
import time
from typing import Dict, Optional

import websockets.sync.client

# 尝试导入msgpack_numpy，如果不存在则使用临时路径
try:
    import sys
    from pathlib import Path
    tmp_path = Path(__file__).parent.parent / "tmp" / "openpi_client"
    if tmp_path.exists():
        sys.path.insert(0, str(tmp_path.parent))
        from openpi_client import msgpack_numpy
    else:
        # 如果找不到，创建一个简单的包装
        import msgpack
        import numpy as np
        
        class SimpleMsgpackNumpy:
            class Packer:
                def pack(self, obj):
                    return msgpack.packb(obj, default=self._pack_array)
            
            @staticmethod
            def _pack_array(obj):
                if isinstance(obj, np.ndarray):
                    return {
                        b"__ndarray__": True,
                        b"data": obj.tobytes(),
                        b"dtype": str(obj.dtype),
                        b"shape": obj.shape,
                    }
                return obj
            
            @staticmethod
            def unpackb(data):
                return msgpack.unpackb(data, raw=False)
        
        msgpack_numpy = SimpleMsgpackNumpy()
except ImportError:
    raise ImportError("msgpack_numpy not available. Please ensure openpi_client is accessible.")


class WebSocketClient:
    """WebSocket客户端类，用于与Policy Server通信"""
    
    def __init__(
        self,
        server_uri: str,
        api_key: Optional[str] = None,
        reconnect_interval: float = 5.0
    ):
        """
        初始化WebSocket客户端
        
        Args:
            server_uri: 服务器URI（如"ws://10.11.5.2:8000"）
            api_key: API密钥（可选）
            reconnect_interval: 重连间隔（秒）
        """
        if not server_uri.startswith("ws"):
            self._uri = f"ws://{server_uri}"
        else:
            self._uri = server_uri
        
        self._api_key = api_key
        self._reconnect_interval = reconnect_interval
        self._packer = msgpack_numpy.Packer()
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Optional[Dict] = None
        
        self.connect()
    
    def connect(self) -> None:
        """连接到服务器"""
        logging.info(f"Connecting to server at {self._uri}...")
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, additional_headers=headers
            )
            # 接收服务器元数据（如果有）
            try:
                metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=1.0))
                self._server_metadata = metadata
            except Exception:
                # 如果没有元数据，继续
                self._server_metadata = {}
            logging.info(f"Connected to server at {self._uri}")
        except Exception as e:
            logging.error(f"Failed to connect to server: {e}")
            raise
    
    def _wait_for_server(self) -> None:
        """等待服务器可用（重连机制）"""
        while True:
            try:
                self.connect()
                return
            except Exception:
                logging.info(f"Still waiting for server at {self._uri}...")
                time.sleep(self._reconnect_interval)
    
    def send_observation(self, obs: Dict) -> None:
        """
        发送observation到服务器
        
        Args:
            obs: observation字典
        """
        if self._ws is None:
            self._wait_for_server()
        
        try:
            data = self._packer.pack(obs)
            self._ws.send(data)
        except Exception as e:
            logging.error(f"Error sending observation: {e}")
            # 尝试重连
            self._wait_for_server()
            data = self._packer.pack(obs)
            self._ws.send(data)
    
    def receive_response(self) -> Dict:
        """
        接收服务器响应
        
        Returns:
            响应字典，包含type字段（"update_world_model"或"action"）
        """
        if self._ws is None:
            self._wait_for_server()
        
        try:
            response = self._ws.recv()
            if isinstance(response, str):
                # 如果服务器发送字符串，可能是错误消息
                raise RuntimeError(f"Error in server:\n{response}")
            return msgpack_numpy.unpackb(response)
        except Exception as e:
            logging.error(f"Error receiving response: {e}")
            # 尝试重连
            self._wait_for_server()
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in server:\n{response}")
            return msgpack_numpy.unpackb(response)
    
    def send_and_receive(self, obs: Dict) -> Dict:
        """
        发送observation并接收响应（便捷方法）
        
        Args:
            obs: observation字典
            
        Returns:
            响应字典
        """
        self.send_observation(obs)
        return self.receive_response()
    
    def get_server_metadata(self) -> Optional[Dict]:
        """
        获取服务器元数据
        
        Returns:
            服务器元数据字典
        """
        return self._server_metadata
    
    def close(self) -> None:
        """关闭连接"""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._server_metadata = None

