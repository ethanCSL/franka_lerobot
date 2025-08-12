#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveAction, MoveGoal
import actionlib
from scipy.spatial.transform import Rotation as R
import math
import signal
import sys

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class FrankaFixedPoseHorizontal:
    def __init__(self):
        rospy.init_node("franka_fixed_pose_horizontal", anonymous=False)

        # 設定安全 Z 高度
        self.safe_z = 0.12

        # 建立 publisher
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=10
        )

        # 建立 gripper action client
        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        # 建立 PoseStamped message
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "panda_link0"

        # 移動到指定固定位置，並設定夾爪水平
        self.go_to_target_position(0.05, 0.05, self.safe_z)

        # 設定 gripper 開口寬度（可調）
        self.send_gripper_command(width=0.08)

        rospy.loginfo("Target pose published. Node will now idle.")
        rospy.spin()

    def go_to_target_position(self, x, y, z):
        self.pose_msg.pose.position.x = x
        self.pose_msg.pose.position.y = y
        self.pose_msg.pose.position.z = z
        self.set_horizontal_orientation(yaw_deg=0.0)

        rate = rospy.Rate(30)  # 每秒 30 次
        for _ in range(90):  # 發送 3 秒鐘
            self.pose_msg.header.stamp = rospy.Time.now()
            self.pose_pub.publish(self.pose_msg)
            rate.sleep()

        rospy.loginfo(f"Published target pose continuously: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    def set_horizontal_orientation(self, yaw_deg=0.0):
        """夾爪水平朝下的 orientation，yaw 可旋轉夾爪方向"""
        roll = math.radians(170)      
        pitch = 0.0
        yaw = math.radians(yaw_deg)
        quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

        self.pose_msg.pose.orientation.x = quat[0]
        self.pose_msg.pose.orientation.y = quat[1]
        self.pose_msg.pose.orientation.z = quat[2]
        self.pose_msg.pose.orientation.w = quat[3]

    def send_gripper_command(self, width):
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.05
        self.gripper_client.send_goal(goal)
        rospy.loginfo("Sent gripper width command: %.3f m", width)
        self.gripper_client.wait_for_result()
        rospy.loginfo("Gripper action completed.")

if __name__ == "__main__":
    try:
        FrankaFixedPoseHorizontal()
    except rospy.ROSInterruptException:
        pass
