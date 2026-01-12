"""传感器同步节点

订阅ROS原始话题（4个相机的RGB/Depth和gripper pose），
使用message_filters同步消息，以20Hz频率发布同步后的数据。
"""
import cv2
import rospy
import threading
import sys
import time
from pathlib import Path
from typing import Dict
from collections import deque
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_tool_box.config_loader import get_client_config
from utils.ros_message_helper import create_sensor_sync_message_dict

# ROS消息类型
from sensor_msgs.msg import Image, CompressedImage
from astribot_msgs.msg import GripperState, SyncPackage
# from geometry_msgs.msg import Pose
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
        self.sync_topic = self.ros_config.get("sync_topic", "/sensor_sync_package")
        self.sync_rate = self.ros_config.get("sync_rate", 20.0)
        self.sync_precision = self.ros_config.get("sync_precision", 0.05)
        
        # 初始化ROS节点
        rospy.init_node('sensor_sync_node', anonymous=True)
        
        # 创建订阅者
        self._create_subscribers()
        
        # 创建发布者（使用字典在框架阶段，实际部署时使用自定义消息类型）
        # 注意：在实际部署时，应该使用自定义ROS消息类型
        # 这里使用一个简单的发布者作为框架
        self.pub = rospy.Publisher(self.sync_topic, SyncPackage, queue_size=10)
        
        # 同步器
        self._create_synchronizer()
        
        # 定时器（20Hz发布）
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.sync_rate), self._publish_callback)
        
        # 存储最新同步的数据
        self.latest_sync_data = None
        self.lock = threading.Lock()
        
        # Gripper pose历史队列（FIFO，最大长度20）
        self.gripper_history = deque(maxlen=20)
        
        # 用于控制图片保存频率（每1秒保存一次）
        self.last_save_time = time.time()
        self.save_interval = 1e9  # never
        self.save_lock = threading.Lock()  # 保护保存操作的锁
        
        rospy.loginfo(f"Sensor sync node initialized, publishing at {self.sync_rate} Hz")
    
    def _create_subscribers(self):
        """创建订阅者"""
        # 订阅4个相机的RGB和Depth话题
        self.cam1_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam1"]["rgb"], CompressedImage
        )
        self.cam1_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam1"]["depth"], Image
        )
        self.cam2_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam2"]["rgb"], CompressedImage
        )
        self.cam2_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam2"]["depth"], Image
        )
        self.cam3_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam3"]["rgb"], CompressedImage
        )
        self.cam3_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam3"]["depth"], Image
        )
        self.cam4_rgb_sub = message_filters.Subscriber(
            self.camera_topics["cam4"]["rgb"], CompressedImage
        )
        self.cam4_depth_sub = message_filters.Subscriber(
            self.camera_topics["cam4"]["depth"], Image
        )
        
        # 订阅gripper pose话题（需要根据实际消息类型调整）
        # 这里假设使用Pose消息，实际可能需要自定义消息类型
        try:
            self.gripper_pose_sub = message_filters.Subscriber(
                self.gripper_pose_topic, GripperState
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
        # 提取当前gripper_pose数据
        if hasattr(gripper_pose, 'data'):
            current_gripper_data = np.array(gripper_pose.data, dtype=np.float32)
        else:
            # 如果gripper_pose本身就是一个数组或列表
            current_gripper_data = np.array(gripper_pose, dtype=np.float32)
        
        # 将当前gripper_pose添加到历史队列（FIFO，自动维护20长度）
        with self.lock:
            self.gripper_history.append(current_gripper_data.copy())
            
            # 从历史队列中提取当前、前第10帧、前第20帧的gripper_pose
            history_len = len(self.gripper_history)
            
            if history_len == 0:
                # 如果队列为空（不应该发生），使用当前数据填充
                rospy.logwarn("Gripper history queue is empty, using current data")
                current_frame = current_gripper_data
                frame_10_ago = current_gripper_data
                frame_20_ago = current_gripper_data
            else:
                # 当前帧（最新）
                current_frame = self.gripper_history[-1]
                
                # 前第10帧（如果存在）
                if history_len >= 10:
                    frame_10_ago = self.gripper_history[-10]
                else:
                    # 如果历史不足10帧，使用最早的帧填充
                    frame_10_ago = self.gripper_history[0]
                
                # 前第20帧（如果存在）
                if history_len >= 20:
                    frame_20_ago = self.gripper_history[-20]
                else:
                    # 如果历史不足20帧，使用最早的帧填充
                    frame_20_ago = self.gripper_history[0]
            
            # 组合成数组：[当前帧, 前第10帧, 前第20帧]
            # gripper_combined = np.concatenate([current_frame, frame_10_ago, frame_20_ago])
            gripper_combined = np.concatenate([frame_20_ago, frame_10_ago, current_frame])
        
        sync_data = SyncPackage()
        
        sync_data.header.stamp = rospy.Time.now()
        sync_data.header.frame_id = "base_link"

        # # 保存图片（每1秒一次，不阻塞）
        # current_time = time.time()
        # should_save = False

        # if current_time - self.last_save_time >= self.save_interval:
        #     should_save = True
        #     self.last_save_time = current_time
        
        # if should_save:
        #     # 在单独线程中保存图片，避免阻塞回调函数
        #     np_rgb = np.frombuffer(cam2_rgb.data, dtype=np.uint8)
        #     rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
            
        #     try:
        #         cv2.imwrite(f"debug_rgbs_{time.time():.6f}.png", rgb_img)
        #     except Exception as e:
        #         rospy.logerr(f"Error saving image: {e}")
            

        # 2. 赋值 RGB 数组
        # 数组顺序约定：[cam1, cam2, cam3, cam4]
        sync_data.rgb_list = [
            cam1_rgb, 
            cam2_rgb, 
            cam3_rgb, 
            cam4_rgb
        ]

        # 3. 赋值 Depth 数组
        sync_data.depth_list = [
            cam1_depth, 
            cam2_depth, 
            cam3_depth, 
            cam4_depth
        ]

        # 4. 赋值 Gripper 状态（包含当前、前第10帧、前第20帧）
        # gripper_combined 形状应该是 (24,)，如果每帧是8个元素（8*3=24）
        # 创建GripperState对象
        gripper_state = GripperState()
        gripper_state.header = sync_data.header
        gripper_state.data = gripper_combined.tolist()
        sync_data.gripper_hist = gripper_state
        
        with self.lock:
            rospy.logdebug(f"updated - gripper history length: {history_len}, combined shape: {gripper_combined.shape}")
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
    
    def _publish_callback(self, event):
        """定时发布回调函数（20Hz）"""
        with self.lock:
            if self.latest_sync_data is not None:
                # 在实际部署时，这里应该发布自定义ROS消息类型
                # 框架阶段，我们可以使用一个简单的Header消息作为占位符
                # 实际数据可以通过其他方式传递（如共享内存、文件等）
                
                # 发布header（占位符）
                self.pub.publish(self.latest_sync_data)
                print("updated")

                current_time = time.time()
                should_save = False
                if current_time - self.last_save_time >= self.save_interval:
                    should_save = True
                    self.last_save_time = current_time
                
                if should_save:
                    # 在单独线程中保存图片，避免阻塞回调函数
                    np_rgb = np.frombuffer(self.latest_sync_data.rgb_list[1].data, dtype=np.uint8)
                    rgb_img = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
                    
                    try:
                        cv2.imwrite(f"pub_rgbs_{time.time():.6f}.png", rgb_img)
                    except Exception as e:
                        rospy.logerr(f"Error saving image: {e}")
                
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

