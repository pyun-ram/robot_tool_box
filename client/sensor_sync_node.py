"""传感器同步节点

订阅ROS原始话题（4个相机的RGB/Depth和gripper pose），
使用message_filters同步消息，以20Hz频率发布同步后的数据。
"""

import rospy
import sys
from pathlib import Path
from typing import Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_client_config
from utils.ros_message_helper import create_sensor_sync_message_dict

# ROS消息类型
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import message_filters


class SensorSyncNode:
    """传感器同步节点类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化传感器同步节点
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        client_config = get_client_config(config_path)
        self.ros_config = client_config.get("ros", {})
        self.camera_topics = self.ros_config.get("camera_topics", {})
        self.gripper_pose_topic = self.ros_config.get("gripper_pose_topic", "/gripper_pose")
        self.sync_topic = self.ros_config.get("sync_topic", "/sensor_sync/data")
        self.sync_rate = self.ros_config.get("sync_rate", 20.0)
        self.sync_precision = self.ros_config.get("sync_precision", 0.05)
        
        # 初始化ROS节点
        rospy.init_node('sensor_sync_node', anonymous=True)
        
        # 创建订阅者
        self._create_subscribers()
        
        # 创建发布者（使用字典在框架阶段，实际部署时使用自定义消息类型）
        # 注意：在实际部署时，应该使用自定义ROS消息类型
        # 这里使用一个简单的发布者作为框架
        self.pub = rospy.Publisher(self.sync_topic, Header, queue_size=10)
        
        # 同步器
        self._create_synchronizer()
        
        # 定时器（20Hz发布）
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.sync_rate), self._publish_callback)
        
        # 存储最新同步的数据
        self.latest_sync_data = None
        self.lock = rospy.Lock()
        
        rospy.loginfo(f"Sensor sync node initialized, publishing at {self.sync_rate} Hz")
    
    def _create_subscribers(self):
        """创建订阅者"""
        # 订阅4个相机的RGB和Depth话题
        self.cam1_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam1"]["rgb"], Image
        )
        self.cam1_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam1"]["depth"], Image
        )
        self.cam2_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam2"]["rgb"], Image
        )
        self.cam2_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam2"]["depth"], Image
        )
        self.cam3_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam3"]["rgb"], Image
        )
        self.cam3_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam3"]["depth"], Image
        )
        self.cam4_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam4"]["rgb"], Image
        )
        self.cam4_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam4"]["depth"], Image
        )
        
        # 订阅gripper pose话题（需要根据实际消息类型调整）
        # 这里假设使用Pose消息，实际可能需要自定义消息类型
        try:
            self.gripper_pose_sub = message_filters.Subscriber(
                self.gripper_pose_topic, Pose
            )
        except Exception:
            rospy.logwarn(f"Could not subscribe to gripper_pose topic: {self.gripper_pose_topic}")
            self.gripper_pose_sub = None
    
    def _create_synchronizer(self):
        """创建消息同步器"""
        # 收集所有订阅者
        subs = [
            self.cam1_rgb_sub,
            self.cam1_depth_sub,
            self.cam2_rgb_sub,
            self.cam2_depth_sub,
            self.cam3_rgb_sub,
            self.cam3_depth_sub,
            self.cam4_rgb_sub,
            self.cam4_depth_sub,
        ]
        
        if self.gripper_pose_sub is not None:
            subs.append(self.gripper_pose_sub)
        
        # 创建近似时间同步器
        # 注意：实际部署时可能需要根据gripper_pose的消息类型调整
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=10,
            slop=self.sync_precision
        )
        
        if self.gripper_pose_sub is not None:
            self.sync.registerCallback(self._sync_callback_with_gripper)
        else:
            self.sync.registerCallback(self._sync_callback_without_gripper)
    
    def _sync_callback_with_gripper(self, cam1_rgb, cam1_depth, cam2_rgb, cam2_depth,
                                    cam3_rgb, cam3_depth, cam4_rgb, cam4_depth, gripper_pose):
        """同步回调函数（包含gripper pose）"""
        # 转换gripper_pose为numpy数组（假设Pose消息包含8个值）
        # 实际部署时需要根据实际消息类型调整
        gripper_pose_array = self._pose_to_array(gripper_pose)
        
        # 创建同步消息字典
        sync_data = create_sensor_sync_message_dict(
            cam1_rgb=cam1_rgb,
            cam1_depth=cam1_depth,
            cam2_rgb=cam2_rgb,
            cam2_depth=cam2_depth,
            cam3_rgb=cam3_rgb,
            cam3_depth=cam3_depth,
            cam4_rgb=cam4_rgb,
            cam4_depth=cam4_depth,
            gripper_pose=gripper_pose_array,
            header=cam1_rgb.header
        )
        
        with self.lock:
            self.latest_sync_data = sync_data
    
    def _sync_callback_without_gripper(self, cam1_rgb, cam1_depth, cam2_rgb, cam2_depth,
                                       cam3_rgb, cam3_depth, cam4_rgb, cam4_depth):
        """同步回调函数（不包含gripper pose）"""
        # 创建默认gripper pose（全零）
        gripper_pose_array = np.zeros(8, dtype=np.float32)
        
        sync_data = create_sensor_sync_message_dict(
            cam1_rgb=cam1_rgb,
            cam1_depth=cam1_depth,
            cam2_rgb=cam2_rgb,
            cam2_depth=cam2_depth,
            cam3_rgb=cam3_rgb,
            cam3_depth=cam3_depth,
            cam4_rgb=cam4_rgb,
            cam4_depth=cam4_depth,
            gripper_pose=gripper_pose_array,
            header=cam1_rgb.header
        )
        
        with self.lock:
            self.latest_sync_data = sync_data
    
    def _pose_to_array(self, pose: Pose) -> np.ndarray:
        """
        将Pose消息转换为numpy数组（8维：[x,y,z,rx,ry,rz,rw,openness]）
        
        注意：这是框架实现，实际部署时需要根据实际的gripper_pose消息类型调整
        """
        
        # 假设Pose消息包含位置和四元数，openness需要从其他地方获取
        # 这里只是框架实现
        arr = np.zeros(8, dtype=np.float32)
        arr[0] = pose.position.x if hasattr(pose, 'position') else 0.0
        arr[1] = pose.position.y if hasattr(pose, 'position') else 0.0
        arr[2] = pose.position.z if hasattr(pose, 'position') else 0.0
        arr[3] = pose.orientation.x if hasattr(pose, 'orientation') else 0.0
        arr[4] = pose.orientation.y if hasattr(pose, 'orientation') else 0.0
        arr[5] = pose.orientation.z if hasattr(pose, 'orientation') else 0.0
        arr[6] = pose.orientation.w if hasattr(pose, 'orientation') else 0.0
        arr[7] = 0.0  # openness，需要从实际消息中获取
        
        return arr
    
    def _publish_callback(self, event):
        """定时发布回调函数（20Hz）"""
        with self.lock:
            if self.latest_sync_data is not None:
                # 在实际部署时，这里应该发布自定义ROS消息类型
                # 框架阶段，我们可以使用一个简单的Header消息作为占位符
                # 实际数据可以通过其他方式传递（如共享内存、文件等）
                
                # 发布header（占位符）
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "sensor_sync"
                self.pub.publish(header)
                
                # 注意：实际同步的数据（latest_sync_data）需要通过其他机制传递给其他节点
                # 在框架阶段，可以考虑使用rospy的参数服务器或其他机制
                # 实际部署时应该使用自定义消息类型
    
    def run(self):
        """运行节点"""
        rospy.loginfo("Sensor sync node started")
        rospy.spin()


def main():
    """主函数"""
    try:
        node = SensorSyncNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

