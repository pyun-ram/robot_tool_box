"""进程间通信核心模块 - WebSocket + msgpack + numpy + torch"""
import abc
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

import msgpack
import numpy as np
import websockets.asyncio.server as _server
import websockets.sync.client

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- msgpack numpy + torch 支持 ---
def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    
    # 支持 torch.Tensor
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        # 转换为 numpy 数组进行序列化
        if obj.is_cuda:
            obj = obj.cpu()
        # 确保数组是可写的，避免警告
        np_array = obj.detach().numpy()
        if not np_array.flags.writeable:
            np_array = np_array.copy()
        return {
            b"__tensor__": True,
            b"data": np_array.tobytes(),
            b"dtype": np_array.dtype.str,
            b"shape": np_array.shape,
            b"requires_grad": obj.requires_grad
        }
    
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    
    # 支持 torch.Tensor
    if b"__tensor__" in obj:
        np_array = np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        tensor = torch.from_numpy(np_array)
        if obj.get(b"requires_grad", False):
            tensor.requires_grad_(True)
        return tensor
    
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
                request = unpackb(await websocket.recv())
                
                # 检查是否是初始化请求
                if isinstance(request, dict) and request.get("type") == "initialize":
                    # 调用策略的 initialize 方法
                    if hasattr(self._policy, "initialize"):
                        result = self._policy.initialize(
                            model_name=request["model_name"],
                            model_args=request["model_args"],
                            checkpoint=request["checkpoint"],
                            device=request.get("device", "cuda")
                        )
                        await websocket.send(packer.pack(result))
                    else:
                        await websocket.send(packer.pack({
                            "status": "error",
                            "message": "策略不支持初始化"
                        }))
                else:
                    # 正常的推理请求
                    action = self._policy.infer(request)
                    await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                logger.error(f"处理错误: {e}")
                import traceback
                traceback.print_exc()
                await websocket.close(1011, "内部错误")
                raise


class WebsocketRelayServer:
    def __init__(self, 
                 policy: BasePolicy, 
                 host: str = "0.0.0.0", 
                 port: int = 8765, 
                 target_host: str = "127.0.0.1", 
                 target_port: int = 9000):
        self._policy = policy
        self._host = host
        self._port = port
        self._target_uri = f"ws://{target_host}:{target_port}"
        self._packer = Packer()
        self._target_ws = None

    async def _connect_to_target(self):
        """建立并保持与下游服务器的连接"""
        while True:
            try:
                if self._target_ws is None or self._target_ws.closed:
                    logger.info(f"正在连接目标服务器 {self._target_uri}...")
                    self._target_ws = await websockets.connect(
                        self._target_uri, compression=None, max_size=None
                    )
                    # 接收握手信息（如果下游服务器有的话）
                    await self._target_ws.recv() 
                    logger.info("目标服务器连接成功")
                await asyncio.sleep(5) # 心跳检查
            except Exception as e:
                logger.error(f"目标服务器连接失败: {e}")
                await asyncio.sleep(1)

    async def _handler(self, websocket):
        """处理来自原始客户端的请求"""
        logger.info("原始客户端已连接")
        # 1. 握手确认
        await websocket.send(self._packer.pack({"status": "ready"}))
        
        while True:
            try:
                # 2. 接收推理请求
                raw_request = await websocket.recv()
                request = unpackb(raw_request)

                # 3. 立即返回确认回复 (ACK)
                await websocket.send(self._packer.pack({"status": "received", "msg": "Inference started"}))

                # 4. 异步执行推理并转发 (不阻塞上面的 recv 循环)
                asyncio.create_task(self._process_and_forward(request))

            except websockets.ConnectionClosed:
                logger.info("原始客户端连接关闭")
                break
            except Exception as e:
                logger.error(f"处理循环出错: {e}")
                break

    async def _process_and_forward(self, request: Dict):
        """核心逻辑：本地推理 -> 发送到另一台服务器"""
        try:
            # 执行本地 Policy 推理
            # 注意：如果 policy.infer 是同步阻塞的，建议改用 run_in_executor
            loop = asyncio.get_event_loop()
            action = await loop.run_in_executor(None, self._policy.infer, request)

            if action is None:
                logger.info("本次policy.infer不转发")
                return

            # 检查下游连接是否可用
            if self._target_ws and not self._target_ws.closed:
                # 5. 发送处理后的数据到另一台服务器
                await self._target_ws.send(self._packer.pack(action))
                logger.debug("数据已转发至目标服务器")
            else:
                logger.warning("丢失目标服务器连接，数据丢弃")
        except Exception as e:
            logger.error(f"推理或转发失败: {e}")

    async def run(self):
        # 同时运行：1. 服务器监听 2. 目标客户端连接维护
        logger.info(f"启动中转服务器于 {self._host}:{self._port}")
        
        # 启动维护下游连接的任务
        asyncio.create_task(self._connect_to_target())
        
        async with websockets.serve(self._handler, self._host, self._port, compression=None, max_size=None):
            await asyncio.Future()  # 永久运行

    def serve_forever(self):
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            pass

