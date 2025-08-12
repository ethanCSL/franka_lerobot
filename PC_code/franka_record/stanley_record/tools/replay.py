from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveAction, MoveGoal, FrankaState
import actionlib
from dynamic_reconfigure.client import Client

from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import math
import signal
import time
import signal
import sys
import rospy

shutdown_requested = False

def franka_state_callback(self, msg):
    pos = msg.O_T_EE[:3]  # position x, y, z
    rot = np.array(msg.O_T_EE).reshape((4, 4))[:3, :3]
    quat = R.from_matrix(rot).as_quat()  # Convert rotation matrix to quaternion (x,y,z,w)

    # gripper width (需要另外從 gripper topic 或透過 action 回報抓取)
    # 為了簡化這裡暫時設定成 None
    self.real_robot_states.append([
        pos[0], pos[1], pos[2],
        quat[0], quat[1], quat[2], quat[3],
        None,  # Gripper width - optional
        None   # Force z - optional
    ])

def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

class FrankaHardcodedPose:
    def __init__(self):
        rospy.init_node("franka_hardcoded_pose", anonymous=False)
        self.file_path = os.path.expanduser('~/.cache/huggingface/lerobot/ethanCSL/0804_wipe/data/chunk-000/episode_000000.parquet')
        self.obs_states = self.parquet_reader()

        self.replayed_values = []
        self.dataset_values = []
        # Publisher
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=10
        )
        self.dyn_client = Client("/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node", timeout=5.0)

        # Gripper action client
        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        self.real_robot_states = []  # 用來儲存實際 robot pose
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.franka_state_callback)

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

    def parquet_reader(self):
        df = pd.read_parquet(self.file_path)
        obs_states = df['observation.state']
        print("Load Data Success")
        return obs_states
    
    def source_replay(self):
        rate = rospy.Rate(10000)
        cnt = 0
        force_scale = 0.005
        
        for obs in self.obs_states:
            if rospy.is_shutdown() or shutdown_requested:
                rospy.loginfo("Shutdown requested, stopping replay...")
                break
            self.pose_msg.pose.position.x = obs[0]
            self.pose_msg.pose.position.y = obs[1]
            #self.pose_msg.pose.position.z = obs[2]
            adjusted_z = obs[2] + force_scale * obs[8]  # obs[8] 是 force_z
            self.pose_msg.pose.position.z = adjusted_z
 
            self.pose_msg.pose.orientation.x = obs[3]
            self.pose_msg.pose.orientation.y = obs[4]
            self.pose_msg.pose.orientation.z = obs[5]
            self.pose_msg.pose.orientation.w = obs[6]

            if cnt % 1 == 0:
                self.send_gripper_command(width=obs[7])

                fz = obs[8]  # force.z from recording
                ext_force = [0.0, 0.0, fz, 0.0, 0.0, 0.0]
                print(f"Replaying with force.z = {fz}")

                self.pose_msg.header.stamp = rospy.Time.now()
                self.pose_pub.publish(self.pose_msg)

                print(round(round(obs[3],4)))
                print(round(round(obs[4],4)))
                print(round(round(obs[5],4)))
                print(round(round(obs[6],4)))
            
            self.replayed_values.append([
                self.pose_msg.pose.position.x,
                self.pose_msg.pose.position.y,
                self.pose_msg.pose.position.z,
                self.pose_msg.pose.orientation.x,
                self.pose_msg.pose.orientation.y,
                self.pose_msg.pose.orientation.z,
                self.pose_msg.pose.orientation.w,
                obs[7],  # gripper width 
                obs[8],  # force_z
            ])

            self.dataset_values.append(obs[:9])
            cnt += 1
            rate.sleep()
        self.plot_comparison()
        
    def plot_comparison(self):
        replay = np.array(self.replayed_values)
        data = np.array(self.dataset_values)
        labels = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper", "fz"]

        fig, axs = plt.subplots(9, 1, figsize=(12, 18))

        # 嘗試裁切 real robot 狀態到對應長度
        real = np.array(self.real_robot_states)
        if real.shape[0] >= replay.shape[0]:
            real = real[:replay.shape[0], :]
        else:
            print("⚠️ Real robot states shorter than replayed data, padding with zeros.")
            pad_len = replay.shape[0] - real.shape[0]
            pad = np.zeros((pad_len, 9))
            real = np.vstack([real, pad])

        for i in range(9):
            axs[i].plot(replay[:, i], label='Replayed', linestyle='-')
            axs[i].plot(data[:, i], label='Dataset', linestyle='--')
            axs[i].plot(real[:, i], label='Real Robot', linestyle=':')
            axs[i].set_title(labels[i])
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()


    def euler_to_quaternion(self, roll_rad, pitch_rad, yaw_rad):
        """
        輸入：
            roll_rad, pitch_rad, yaw_rad - 已經是弧度單位
        輸出：
            四元數 x, y, z, w
        """
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return x, y, z, w


if __name__ == "__main__":
    try:
        FrankaHardcodedPose()
    except rospy.ROSInterruptException:
        pass
