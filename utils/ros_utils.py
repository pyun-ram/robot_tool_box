"""ROS工具函数模块

提供ROS消息转换等工具函数。
"""

import numpy as np
from typing import Optional


def ros_image_to_numpy(msg) -> Optional[np.ndarray]:
    """
    ROS Image消息转numpy数组
    
    Args:
        msg: ROS Image消息（sensor_msgs.msg.Image）
        
    Returns:
        numpy数组，shape为(H, W, C)，如果失败则返回None
    """
    try:
        import cv2
        from cv_bridge import CvBridge
        
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        return cv_image
    except ImportError:
        raise ImportError("cv_bridge not installed. Please install: sudo apt-get install ros-<distro>-cv-bridge")
    except Exception as e:
        print(f"Error converting ROS image to numpy: {e}")
        return None


def numpy_to_ros_image(arr: np.ndarray, encoding: str = "rgb8") -> Optional[object]:
    """
    numpy数组转ROS Image消息
    
    Args:
        arr: numpy数组，shape为(H, W, C)
        encoding: 图像编码格式（如"rgb8", "bgr8", "mono8"等）
        
    Returns:
        ROS Image消息，如果失败则返回None
    """
    try:
        import cv2
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(arr, encoding=encoding)
        return ros_image
    except ImportError:
        raise ImportError("cv_bridge not installed. Please install: sudo apt-get install ros-<distro>-cv-bridge")
    except Exception as e:
        print(f"Error converting numpy to ROS image: {e}")
        return None


def ros_pointcloud_to_numpy(msg) -> Optional[np.ndarray]:
    """
    ROS PointCloud消息转numpy数组
    
    Args:
        msg: ROS PointCloud消息（sensor_msgs.msg.PointCloud2）
        
    Returns:
        numpy数组，shape为(N, 3)，N为点数量，如果失败则返回None
    """
    try:
        import sensor_msgs.point_cloud2 as pc2
        
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if len(points) == 0:
            return None
        
        return np.array(points, dtype=np.float32)
    except ImportError:
        raise ImportError("sensor_msgs not available")
    except Exception as e:
        print(f"Error converting ROS pointcloud to numpy: {e}")
        return None


def numpy_to_ros_pointcloud(points: np.ndarray, frame_id: str = "base_link") -> Optional[object]:
    """
    numpy数组转ROS PointCloud消息
    
    Args:
        points: numpy数组，shape为(N, 3)，N为点数量
        frame_id: 坐标系ID
        
    Returns:
        ROS PointCloud2消息，如果失败则返回None
    """
    try:
        import sensor_msgs.point_cloud2 as pc2
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Header
        
        header = Header()
        header.frame_id = frame_id
        header.stamp = None  # 应该设置为当前时间
        
        # 创建点云消息
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1)
        ]
        
        points_list = points.tolist()
        cloud = pc2.create_cloud(header, fields, points_list)
        return cloud
    except ImportError:
        raise ImportError("sensor_msgs not available")
    except Exception as e:
        print(f"Error converting numpy to ROS pointcloud: {e}")
        return None


def get_ros_time() -> float:
    """
    获取ROS时间（秒）
    
    Returns:
        ROS时间戳（秒）
    """
    try:
        import rospy
        return rospy.Time.now().to_sec()
    except ImportError:
        raise ImportError("rospy not available")
    except Exception as e:
        print(f"Error getting ROS time: {e}")
        return 0.0

