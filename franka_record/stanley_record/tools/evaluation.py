#client.py
import os
import sys
import math
import time
import signal
import json
import struct
import socket
import rospy
import actionlib
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, WrenchStamped
from franka_gripper.msg import MoveAction, MoveGoal, GraspGoal, GraspAction
from actionlib_msgs.msg import GoalStatus
from franka_gripper.msg import GraspEpsilon, GraspGoal, GraspResult
from sensor_msgs.msg import JointState

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class FrankaHardcodedPose:
    def __init__(self):
        rospy.init_node("franka_hardcoded_pose", anonymous=False)

        # ROS publishers/subscribers
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=10
        )
        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        self.pose_sub = rospy.Subscriber("/robot0_eef_pose", PoseStamped, self.pose_cb)
        self.gripper_sub = rospy.Subscriber("/robot0_gripper_width", Float32, self.gripper_cb)
        self.eye_sub = rospy.Subscriber("/robot0_eye_in_hand_image", Image, self.eye_cb)
        self.agent_sub = rospy.Subscriber("/robot0_agentview_image", Image, self.agent_cb)
        self.force_sub = rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.ext_force_callback, queue_size=10)

        self.gripper_effort = None
        rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_effort_callback)
        self.gripper_state = None 

        self.gripper = 0.08  # 初始張開 8 cm
        self.gripper_delta_accum = 0.0
        self.last_gripper_command_time = rospy.Time.now()

        self.bridge = CvBridge()
        self.pose = [0.393202,
                    0.103841,
                    0.068092,
                    0.930702,
                    -0.365772,
                    0.000217,
                    0.002040
        ]
        #self.gripper = 0.08
        self.eye_image = None
        self.agent_image = None
        self.ext_force = 0
        self.action = None
        self.gripper_effort = None
        self.gripper_now = 0.08

        # Socket connection
        SERVER_IP = '192.168.1.162' #FET
        #SERVER_IP = '10.100.4.192' #TT
        #SERVER_IP = '192.168.123.128' #ASUS
        PORT = 5001
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((SERVER_IP, PORT))

        # Initial pose
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "panda_link0"
        self.pose_msg.pose.position.x = 0.393202
        self.pose_msg.pose.position.y = 0.103841
        self.pose_msg.pose.position.z = 0.068092
        self.pose_msg.pose.orientation.x = 0.930702
        self.pose_msg.pose.orientation.y = -0.365772
        self.pose_msg.pose.orientation.z = 0.000217
        self.pose_msg.pose.orientation.w = 0.002040
        self.pose_pub.publish(self.pose_msg)

        self.send_gripper_command(width=self.gripper)
        #self.send_gripper_delta(0.0)
        rospy.loginfo("Publishing hard-coded pose at 10Hz.")

    def gripper_effort_callback(self, msg):
        if 'panda_finger_joint1' in msg.name:
            idx = msg.name.index('panda_finger_joint1')
            self.gripper_effort = msg.effort[idx]

    def pose_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.pose = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

    def gripper_cb(self, msg):
        self.gripper_now = msg.data

    def eye_cb(self, msg):
        self.eye_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def agent_cb(self, msg):
        self.agent_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def ext_force_callback(self, msg: WrenchStamped):
        try:
            self.ext_force = msg.wrench.force.z

        except Exception as e:
            rospy.logerr(f"Error in force callback: {e}")

    def send_gripper_command(self, width):
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.15
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()
        result = self.gripper_client.get_result()
        rospy.loginfo("Sent gripper width command: %.7f m", width)
        rospy.loginfo(f"Gripper command result: {result}")
        rospy.loginfo("Gripper action completed.")

    def send_data(self, sock, data_type: str, data_bytes: bytes):
        data_type = data_type.ljust(10).encode()
        data_len = struct.pack('>I', len(data_bytes))
        sock.sendall(data_len + data_type + data_bytes)

    def recv_data(self, sock):
        raw_len = sock.recv(4)
        if not raw_len:
            return None, None
        data_len = struct.unpack('>I', raw_len)[0]
        data_type = sock.recv(10).decode().strip()
        data = b''
        while len(data) < data_len:
            packet = sock.recv(data_len - len(data))
            if not packet:
                break
            data += packet
        return data_type, data

    def data_client(self):
        start = time.time()

        # 1⃣️ 发 state list
        state = self.pose + [self.gripper_now]
        # state = self.pose + [self.gripper_now] + [self.ext_force]
        #print(f"state:{state}")
        self.send_data(self.client_socket, "list", json.dumps(state).encode())

        # 2⃣️ 等待影像到达
        wait_start = time.time()
        while (self.eye_image is None or self.eye_image.size == 0) and not rospy.is_shutdown():
            if time.time() - wait_start > 5.0:
                rospy.logwarn("等待 eye_in_hand 圖像逾時")
                break
            rospy.sleep(0.05)

        while (self.agent_image is None or self.agent_image.size == 0) and not rospy.is_shutdown():
            if time.time() - wait_start > 5.0:
                rospy.logwarn("等待 agentview 圖像逾時")
                break
            rospy.sleep(0.05)

        if self.eye_image is None or self.eye_image.size == 0 or \
        self.agent_image is None or self.agent_image.size == 0:
            rospy.logwarn("影像資料不完整，跳過本次推論")
            return

        # 3⃣️ 编码并发送
        ret1, img1_bytes = cv2.imencode('.jpg', self.eye_image)
        if not ret1:
            rospy.logerr("imencode eye_image 失败")
            return
        self.send_data(self.client_socket, "img1", img1_bytes.tobytes())

        ret2, img2_bytes = cv2.imencode('.jpg', self.agent_image)
        if not ret2:
            rospy.logerr("imencode agent_image 失败")
            return
        self.send_data(self.client_socket, "img2", img2_bytes.tobytes())

        # 4⃣️ 接收并处理 action
        data_type, data = self.recv_data(self.client_socket)
        if data_type == "list":
            self.action = json.loads(data.decode())
            rospy.loginfo(f"收到 server 回傳: {self.action}")

        # 5⃣️ 控制整帧速率
        # elapsed = time.time() - start
        # time.sleep(max(0, 1/30 - elapsed))

    def evaluate(self):
        self.data_client()
        if self.action != None:
            self.pose_msg.pose.position.x = self.action[0] 
            self.pose_msg.pose.position.y = self.action[1] 
            self.pose_msg.pose.position.z = self.action[2] 
            self.pose_msg.pose.orientation.x = self.action[3] 
            self.pose_msg.pose.orientation.y = self.action[4] 
            self.pose_msg.pose.orientation.z = self.action[5] 
            self.pose_msg.pose.orientation.w = self.action[6] 
            self.pose_pub.publish(self.pose_msg)
            self.gripper = self.action[7] 

            try:
                if self.gripper < 0.04 and self.gripper_state != 'closed':
                    print(f"gripper state: close, {self.gripper}")
                    goal = GraspGoal()
                    goal.width = 0.05 #0.05
                    goal.speed = 0.05
                    goal.force = 80.0
                    goal.epsilon.inner = 0.02
                    goal.epsilon.outer = 0.02
                    self.grasp_client.send_goal(goal)
                    self.gripper_state = 'closed'

                elif self.gripper >= 0.055 and self.gripper_state != 'open':
                    print(f"gripper state: open, {self.gripper}")
                    move_goal = MoveGoal()
                    move_goal.width = 0.08
                    move_goal.speed = 0.05
                    self.move_client.send_goal(move_goal)
                    self.gripper_state = 'open'

            except Exception as e:
                rospy.logerr(f"Exception sending grasp goal: {e}")


if __name__ == "__main__":
    try:
        fk = FrankaHardcodedPose()
        while not rospy.is_shutdown() and not shutdown_requested:
            fk.evaluate()
    except rospy.ROSInterruptException:
        pass