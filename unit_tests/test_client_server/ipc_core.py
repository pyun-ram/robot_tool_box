"""最小化进程间通信核心模块 - WebSocket + msgpack + numpy"""
import abc
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

import msgpack
import numpy as np
import websockets.asyncio.server as _server
import websockets.sync.client

# --- msgpack numpy 支持 ---
def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

class Packer(msgpack.Packer):
    def __init__(self):
        super().__init__(default=pack_array)

def unpackb(packed):
    return msgpack.unpackb(packed, object_hook=unpack_array)

# --- 基础接口 ---
class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        pass
    
    def reset(self) -> None:
        pass

# --- 客户端实现 ---
class WebsocketClientPolicy(BasePolicy):
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"连接服务器 {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = unpackb(conn.recv())
                logging.info("连接成功!")
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                time.sleep(1)

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"服务器错误: {response}")
        return unpackb(response)

# --- 服务器实现 ---
logger = logging.getLogger("PolicyServer")

class WebsocketPolicyServer:
    def __init__(self, policy: BasePolicy, host: str = "0.0.0.0", port: int = 8765):
        self._policy = policy
        self._host = host
        self._port = port

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"启动服务器 {self._host}:{self._port}")
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        packer = Packer()
        await websocket.send(packer.pack({"status": "ready"}))
        while True:
            try:
                obs = unpackb(await websocket.recv())
                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"处理错误: {e}")
                await websocket.close(1011, "内部错误")
                raise

