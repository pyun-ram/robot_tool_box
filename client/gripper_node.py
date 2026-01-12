#!/usr/bin/env python
import rospy
import threading
from astribot_msgs.msg import RobotCartesianState, RobotJointState, GripperState
from std_msgs.msg import Header

class GripperPosePublisher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('gripper_pose_publisher', anonymous=True)

        # 创建发布者，发布类型是 GripperState 消息到 /gripper_pose 话题
        self.pub = rospy.Publisher('/gripper_pose', GripperState, queue_size=10)
        
        # 创建2Hz的发布者，发布到 /curr_gripper 话题
        self.curr_gripper_pub = rospy.Publisher('/curr_gripper', GripperState, queue_size=10)

        # 用于存储接收到的数据
        self.lock = threading.Lock()
        self.current_pose = None  # 存储 RobotCartesianState 的 pose
        self.open_state = None    # 存储 RobotJointState 的 position[0]

        # 订阅话题
        self.cartesian_sub = rospy.Subscriber('/astribot_arm_right/endpoint_current_states', 
                        RobotCartesianState, self.cartesian_state_callback)
        self.joint_sub = rospy.Subscriber('/astribot_gripper_right/joint_space_states', 
                        RobotJointState, self.joint_state_callback)

        # 等待话题可用（可选，如果话题不存在会超时）
        rospy.loginfo("检查话题可用性...")
        try:
            rospy.loginfo(f"等待话题: /astribot_arm_right/endpoint_current_states")
            test_msg = rospy.wait_for_message('/astribot_arm_right/endpoint_current_states', RobotCartesianState, timeout=5.0)
            rospy.loginfo(f"话题 /astribot_arm_right/endpoint_current_states 可用，消息类型: {type(test_msg)}")
            rospy.loginfo(f"消息字段: {[attr for attr in dir(test_msg) if not attr.startswith('_')]}")
        except rospy.ROSException:
            rospy.logwarn("话题 /astribot_arm_right/endpoint_current_states 在5秒内未收到消息，继续等待...")
        
        try:
            rospy.loginfo(f"等待话题: /astribot_gripper_right/joint_space_states")
            test_msg = rospy.wait_for_message('/astribot_gripper_right/joint_space_states', RobotJointState, timeout=5.0)
            rospy.loginfo(f"话题 /astribot_gripper_right/joint_space_states 可用，消息类型: {type(test_msg)}")
            rospy.loginfo(f"消息字段: {[attr for attr in dir(test_msg) if not attr.startswith('_')]}")
        except rospy.ROSException:
            rospy.logwarn("话题 /astribot_gripper_right/joint_space_states 在5秒内未收到消息，继续等待...")
        
        rospy.loginfo("订阅者已创建，开始接收消息...")

        # 设置发布频率为 250Hz
        self.rate = rospy.Rate(250)  # 250Hz

    def cartesian_state_callback(self, msg):
        """处理 RobotCartesianState 消息的回调函数"""
        try:
            # 首次收到消息时打印结构信息
            if not hasattr(self, '_cartesian_received'):
                rospy.loginfo("首次收到 RobotCartesianState 消息")
                rospy.loginfo(f"消息类型: {type(msg)}")
                rospy.loginfo(f"消息字段: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
                self._cartesian_received = True
            
            with self.lock:
                # 提取 pose 信息
                # 检查消息结构，可能字段名不同
                if hasattr(msg, 'pose'):
                    self.current_pose = msg.pose
                elif hasattr(msg, 'current_pose'):
                    self.current_pose = msg.current_pose
                elif hasattr(msg, 'state') and hasattr(msg.state, 'pose'):
                    self.current_pose = msg.state.pose
                else:
                    rospy.logwarn_throttle(1.0, f"RobotCartesianState 消息结构未知，可用字段: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
                    return
                rospy.logdebug("收到 RobotCartesianState 消息")
        except Exception as e:
            rospy.logerr(f"处理 RobotCartesianState 消息时出错: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def joint_state_callback(self, msg):
        """处理 RobotJointState 消息的回调函数"""
        try:
            # 首次收到消息时打印结构信息
            if not hasattr(self, '_joint_received'):
                rospy.loginfo("首次收到 RobotJointState 消息")
                rospy.loginfo(f"消息类型: {type(msg)}")
                rospy.loginfo(f"消息字段: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
                if hasattr(msg, 'position'):
                    rospy.loginfo(f"position 类型: {type(msg.position)}, 长度: {len(msg.position) if hasattr(msg.position, '__len__') else 'N/A'}")
                self._joint_received = True
            
            with self.lock:
                # 提取 position[0] 作为 open_state
                if hasattr(msg, 'position') and len(msg.position) > 0:
                    self.open_state = msg.position[0]
                elif hasattr(msg, 'joint_position') and len(msg.joint_position) > 0:
                    self.open_state = msg.joint_position[0]
                elif hasattr(msg, 'state') and hasattr(msg.state, 'position') and len(msg.state.position) > 0:
                    self.open_state = msg.state.position[0]
                else:
                    rospy.logwarn_throttle(1.0, f"RobotJointState 消息结构未知，可用字段: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
                    return
                rospy.logdebug("收到 RobotJointState 消息")
        except Exception as e:
            rospy.logerr(f"处理 RobotJointState 消息时出错: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def run(self):
        """主循环，发布数据"""
        while not rospy.is_shutdown():
            timestamp = rospy.Time.now()

            # 创建 GripperState 消息
            gripper_pose = GripperState()
            gripper_pose.header = Header()
            gripper_pose.header.stamp = timestamp

            with self.lock:
                # 检查是否有有效数据
                if self.current_pose is not None and self.open_state is not None:
                    # 从 pose 中提取位置和旋转四元数
                    x = self.current_pose.position.x
                    y = self.current_pose.position.y
                    z = self.current_pose.position.z
                    qx = self.current_pose.orientation.x
                    qy = self.current_pose.orientation.y
                    qz = self.current_pose.orientation.z
                    qw = self.current_pose.orientation.w

                    # 将位置、旋转四元数和开闭状态加入 data 数组
                    gripper_pose.data = [x, y, z, qx, qy, qz, qw, self.open_state]

                    # 发布数据
                    self.pub.publish(gripper_pose)
                else:
                    # 如果数据还未准备好，跳过本次发布
                    missing = []
                    if self.current_pose is None:
                        missing.append("RobotCartesianState")
                    if self.open_state is None:
                        missing.append("RobotJointState")
                    rospy.logwarn_throttle(1.0, f"等待接收机器人状态数据，缺失: {', '.join(missing)}")

            # 按照设定的频率控制循环频率
            self.rate.sleep()

if __name__ == '__main__':
    try:
        publisher = GripperPosePublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
