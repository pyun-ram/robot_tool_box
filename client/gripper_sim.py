#!/usr/bin/env python
import rospy
from astribot_msgs.msg import GripperState
from std_msgs.msg import Header

def gripper_pose_publisher():
    # 初始化 ROS 节点
    rospy.init_node('gripper_pose_publisher', anonymous=True)

    # 创建发布者，发布类型是 GripperState 消息到 /gripper_pose 话题
    pub = rospy.Publisher('/gripper_pose', GripperState, queue_size=10)

    # 设置发布频率为 30Hz
    rate = rospy.Rate(30)  # 30Hz

    # 创建一个 Joy 消息
    gripper_pose = GripperState()

    # 发送数据的循环
    while not rospy.is_shutdown():
        timestamp = rospy.Time.now()

        # 获取时间戳并将其转化为 float64 格式 (秒数 + 纳秒数的比例)
        timestamp_sec = timestamp.to_sec()

        # 设置位置 (x, y, z)
        x = 0.5
        y = 0.2
        z = 0.1

        # 设置旋转四元数 (qx, qy, qz, qw) - 旋转90度绕z轴
        qx = 0.0
        qy = 0.0
        qz = 0.707  # sin(π/4)
        qw = 0.707  # cos(π/4)

        # 设置开闭状态，1表示打开
        open_state = 1

        # 将时间戳加入 header
        gripper_pose.header = Header()
        gripper_pose.header.stamp = timestamp

        # 将位置和旋转四元数加入 axes 数组
        gripper_pose.data = [x, y, z, qx, qy, qz, qw, open_state]

        # 发布数据
        pub.publish(gripper_pose)

        # 按照设定的频率控制循环频率
        rate.sleep()

if __name__ == '__main__':
    try:
        gripper_pose_publisher()
    except rospy.ROSInterruptException:
        pass
