from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveAction, MoveGoal
import actionlib
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import os

import math
import signal
import time
import signal
import sys
import rospy

import pyspacemouse
import time
import math

from scipy.spatial.transform import Rotation as R


shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    #print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

class FrankaHardcodedPose:
    def __init__(self):
        rospy.init_node("franka_hardcoded_pose", anonymous=False)

        # Publisher
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=10
        )

        # Gripper action client
        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        # Create hard-coded PoseStamped
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "panda_link0"
        self.pose_msg.pose.position.x = 0.39320224962484573
        self.pose_msg.pose.position.y = 0.10384112115914704
        self.pose_msg.pose.position.z = 0.0680917613642974
        self.pose_msg.pose.orientation.x = 0.9307023734119501
        self.pose_msg.pose.orientation.y = -0.3657716269392883
        self.pose_msg.pose.orientation.z = 0.00021703032883955154
        self.pose_msg.pose.orientation.w = 0.0020400856319083786

        self.initial_angle = self.quaternion_to_euler([0.9307023734119501, -0.3657716269392883, 0.00021703032883955154, 0.0020400856319083786])

        self.pose_pub.publish(self.pose_msg)
        time.sleep(1)

        # Optionally: send gripper width once
        self.send_gripper_command(width=0.08)
        self.pre_width=0.08

        rospy.loginfo("Publishing hard-coded pose at 10Hz.")
        self.run_loop()

    def run_loop(self):
        self.source_replay()
        rospy.spin()

    def send_gripper_command(self, width):
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.05
        self.gripper_client.send_goal(goal)
        rospy.loginfo("Sent gripper width command: %.3f m", width)
        self.gripper_client.wait_for_result()
        rospy.loginfo("Gripper action completed.")

    def source_replay(self):
        rate = rospy.Rate(50)
        cnt = 0
        pos_ratio = 0.0025
        angle_ratio = 0.01
        success = pyspacemouse.open(DeviceNumber=0)

        # === Constraint === #
        x_min, x_max = 0.3, 0.6
        y_min, y_max = -0.3, 0.3
        z_min, z_max = 0.05, 0.5

        while not rospy.is_shutdown():
            state = pyspacemouse.read()

            # === Add Constraint === #
            # Update position
            self.pose_msg.pose.position.x += (state.x * pos_ratio)
            self.pose_msg.pose.position.y += (state.y * pos_ratio)
            self.pose_msg.pose.position.z += (state.z * pos_ratio)

            # Range limit
            self.pose_msg.pose.position.x = max(min(self.pose_msg.pose.position.x, x_max), x_min)
            self.pose_msg.pose.position.y = max(min(self.pose_msg.pose.position.y, y_max), y_min)
            self.pose_msg.pose.position.z = max(min(self.pose_msg.pose.position.z, z_max), z_min)

            # Range limit(quat)
            roll_min, roll_max = -np.pi/2, np.pi/2
            pitch_min, pitch_max = -np.pi/4, np.pi/4
            yaw_min, yaw_max = -np.pi, np.pi

            #self.initial_angle[0] = max(min(self.initial_angle[0], roll_max), roll_min)
            #self.initial_angle[1] = max(min(self.initial_angle[1], pitch_max), pitch_min)
            self.initial_angle[2] = max(min(self.initial_angle[2], yaw_max), yaw_min)

            # === Original === #
            
            # self.pose_msg.pose.position.x += (state.x * pos_ratio)
            # self.pose_msg.pose.position.y += (state.y * pos_ratio)
            # self.pose_msg.pose.position.z += (state.z * pos_ratio)

            # self.initial_angle[0] += (state.roll * angle_ratio)
            # self.initial_angle[1] += (state.pitch * angle_ratio)
            #sself.initial_angle[2] += (state.yaw * angle_ratio)
            quat = self.euler_to_quaternion(self.initial_angle)

            self.pose_msg.pose.orientation.x = quat[0]
            self.pose_msg.pose.orientation.y = quat[1]
            self.pose_msg.pose.orientation.z = quat[2]
            self.pose_msg.pose.orientation.w = quat[3]

            #print(self.pose_msg.pose.position.x)

            if state.buttons[0] == 0 and self.pre_width != 0.08:
                self.send_gripper_command(width=0.08)
                self.pre_width = 0.08
            elif state.buttons[0] != 0 and self.pre_width != 0.03:
                self.send_gripper_command(width=0.03)
                self.pre_width = 0.03
            self.pose_msg.header.stamp = rospy.Time.now()
            self.pose_pub.publish(self.pose_msg)
            rate.sleep()

    def euler_to_quaternion(self, angle_list):
        """
        使用 scipy 將歐拉角 (roll, pitch, yaw) 轉為四元數 (x, y, z, w)
        輸入角度需為弧度制
        """
        quat = R.from_euler('xyz', angle_list).as_quat()
        # scipy 回傳順序為 [x, y, z, w]
        return quat[0], quat[1], quat[2], quat[3]

    def quaternion_to_euler(self, quat):
        """
        將四元數轉換為歐拉角 (roll, pitch, yaw)
        quat: [x, y, z, w]
        回傳: roll, pitch, yaw (弧度)
        """
        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return [roll, pitch, yaw]


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler) 
    try:
        FrankaHardcodedPose()
    except rospy.ROSInterruptException:
        pass
