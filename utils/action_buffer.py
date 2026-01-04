"""Action Buffer模块

维护action历史队列，实现时间平滑（Time Ensemble）。
支持线程安全的多线程读写。
"""

import threading
from collections import deque
from typing import Optional
import numpy as np


class ActionBuffer:
    """Action Buffer类，用于存储和平滑action（joint angles）"""
    
    def __init__(
        self,
        buffer_size: int = 10,
        smoothing_method: str = "weighted_average",
        weights: str = "exponential"
    ):
        """
        初始化Action Buffer
        
        Args:
            buffer_size: 历史action数量
            smoothing_method: 平滑方法（"weighted_average"）
            weights: 权重类型（"exponential", "linear", "uniform"）
        """
        self.buffer_size = buffer_size
        self.smoothing_method = smoothing_method
        self.weights = weights
        
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
    
    def add_action(self, action: np.ndarray) -> None:
        """
        添加新action（joint angles）
        
        Args:
            action: joint angles数组，shape为(N,)，N为关节数量
        """
        with self._lock:
            # 确保是numpy数组并复制
            action = np.array(action, dtype=np.float32).copy()
            self._buffer.append(action)
    
    def get_smoothed_action(self) -> Optional[np.ndarray]:
        """
        获取平滑后的action（joint angles）
        
        Returns:
            平滑后的joint angles数组，如果buffer为空则返回None
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            
            if self.smoothing_method == "weighted_average":
                return self._weighted_average()
            else:
                # 默认返回最新action
                return self._buffer[-1].copy()
    
    def get_latest_action(self) -> Optional[np.ndarray]:
        """
        获取最新的action（不平滑）
        
        Returns:
            最新的joint angles数组，如果buffer为空则返回None
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1].copy()
    
    def reset(self) -> None:
        """清空buffer"""
        with self._lock:
            self._buffer.clear()
    
    def size(self) -> int:
        """
        获取当前buffer大小
        
        Returns:
            当前buffer中action的数量
        """
        with self._lock:
            return len(self._buffer)
    
    def _weighted_average(self) -> np.ndarray:
        """
        计算加权平均
        
        Returns:
            加权平均后的action数组
        """
        if len(self._buffer) == 0:
            return None
        
        buffer_array = np.array(list(self._buffer))  # shape: (buffer_size, N)
        
        # 计算权重
        n = len(self._buffer)
        if self.weights == "exponential":
            # 指数权重：越新的权重越大
            weights = np.exp(np.linspace(0, 2, n))
        elif self.weights == "linear":
            # 线性权重
            weights = np.linspace(1, n, n)
        elif self.weights == "uniform":
            # 均匀权重
            weights = np.ones(n)
        else:
            weights = np.ones(n)  # 默认均匀权重
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 加权平均
        weighted_avg = np.average(buffer_array, axis=0, weights=weights)
        
        return weighted_avg.astype(np.float32)

