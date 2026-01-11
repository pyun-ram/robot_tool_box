"""配置加载器模块

提供统一的配置加载接口，支持从YAML文件加载配置。
"""

import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，如果为None，使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        if self._config is None:
            raise ValueError(f"Config file is empty: {self.config_path}")
    
    def get_server_config(self) -> Dict[str, Any]:
        """
        获取服务器配置
        
        Returns:
            服务器配置字典
        """
        if self._config is None:
            raise RuntimeError("Config not loaded")
        
        server_config = self._config.get("server", {})
        if not server_config:
            raise ValueError("Server config not found in config file")
        
        return server_config
    
    def get_client_config(self) -> Dict[str, Any]:
        """
        获取客户端配置
        
        Returns:
            客户端配置字典
        """
        if self._config is None:
            raise RuntimeError("Config not loaded")
        
        client_config = self._config.get("client", {})
        if not client_config:
            raise ValueError("Client config not found in config file")
        
        return client_config
    
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        获取完整配置
        
        Returns:
            完整配置字典
        """
        if self._config is None:
            raise RuntimeError("Config not loaded")
        
        return self._config
        
    
    def reload_config(self) -> None:
        """重新加载配置文件"""
        self._load_config()


# 全局配置加载器实例（可选，用于单例模式）
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    获取配置加载器实例（单例模式）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置加载器实例
    """
    global _global_config_loader
    if _global_config_loader is None or config_path is not None:
        _global_config_loader = ConfigLoader(config_path)
    return _global_config_loader


# 便捷函数
def get_server_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """获取服务器配置的便捷函数"""
    loader = get_config_loader(config_path)
    return loader.get_server_config()


def get_client_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """获取客户端配置的便捷函数"""
    loader = get_config_loader(config_path)
    return loader.get_client_config()

def get_camera_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """获取相机配置的函数"""
    loader = get_config_loader(config_path)
    _cameras_config = loader.get_client_config().get("camera", None)
    calib_path = _cameras_config.get("calib_path", None)
    with open(calib_path, "r") as f:
        camera_config = json.load(f)
        cam_names = sorted(camera_config.keys())

        instrin = []
        extrin = []

        for cam in cam_names:
            instrin.append(np.array(camera_config[cam]["intrinsics"], dtype=np.float32))
            extrin.append(np.array(camera_config[cam]["extrinsics"], dtype=np.float32))

        camera_config = {
            "intrinsics": np.stack(instrin, axis=0),
            "extrinsics": np.stack(extrin, axis=0),
        }

    return camera_config

