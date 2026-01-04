"""ROS消息类型辅助函数

提供ROS消息类型的创建和解析辅助函数。
在实际部署时，应该使用自定义ROS消息类型。
框架阶段使用标准消息类型组合。
"""

from typing import Dict, List, Optional
import numpy as np


def create_sensor_sync_message_dict(
    cam1_rgb,
    cam1_depth,
    cam2_rgb,
    cam2_depth,
    cam3_rgb,
    cam3_depth,
    cam4_rgb,
    cam4_depth,
    gripper_pose: np.ndarray,
    header=None
) -> Dict:
    """
    创建传感器同步消息字典（用于内部传递）
    
    注意：这是框架实现阶段的辅助函数。
    实际部署时应使用自定义ROS消息类型。
    
    Args:
        cam1_rgb: cam1的RGB图像（numpy数组或ROS Image消息）
        cam1_depth: cam1的Depth图像
        cam2_rgb: cam2的RGB图像
        cam2_depth: cam2的Depth图像
        cam3_rgb: cam3的RGB图像
        cam3_depth: cam3的Depth图像
        cam4_rgb: cam4的RGB图像
        cam4_depth: cam4的Depth图像
        gripper_pose: gripper位姿，shape为(8,)，[x,y,z,rx,ry,rz,rw,openness]
        header: ROS Header消息（可选）
        
    Returns:
        消息字典
    """
    return {
        "cam1_rgb": cam1_rgb,
        "cam1_depth": cam1_depth,
        "cam2_rgb": cam2_rgb,
        "cam2_depth": cam2_depth,
        "cam3_rgb": cam3_rgb,
        "cam3_depth": cam3_depth,
        "cam4_rgb": cam4_rgb,
        "cam4_depth": cam4_depth,
        "gripper_pose": np.array(gripper_pose, dtype=np.float32),
        "header": header
    }


def parse_sensor_sync_message_dict(msg_dict: Dict) -> Dict:
    """
    解析传感器同步消息字典
    
    Args:
        msg_dict: 消息字典
        
    Returns:
        解析后的数据字典
    """
    return {
        "cam1_rgb": msg_dict.get("cam1_rgb"),
        "cam1_depth": msg_dict.get("cam1_depth"),
        "cam2_rgb": msg_dict.get("cam2_rgb"),
        "cam2_depth": msg_dict.get("cam2_depth"),
        "cam3_rgb": msg_dict.get("cam3_rgb"),
        "cam3_depth": msg_dict.get("cam3_depth"),
        "cam4_rgb": msg_dict.get("cam4_rgb"),
        "cam4_depth": msg_dict.get("cam4_depth"),
        "gripper_pose": np.array(msg_dict.get("gripper_pose", [0]*8), dtype=np.float32),
        "header": msg_dict.get("header")
    }

