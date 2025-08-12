
'''
Support switching between Joint position control and impedance control
'''
#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import actionlib
from threading import Thread

# Franka_ros
from franka_msgs.msg import FrankaState
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, HomingAction, HomingGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController

class DisplacementToPoseController:
    def __init__(self):
        rospy.init_node("displacement_to_pose_controller")

        # === Publishers ===
        self.pose_pub = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        self.state_pose_pub = rospy.Publisher("/robot0_eef_pose", PoseStamped, queue_size=10)
        self.state_gripper_pub = rospy.Publisher("/robot0_gripper_width", Float32, queue_size=10)
        self.reset_done_pub = rospy.Publisher("/reset_done", Float32, queue_size=1)

        # === Subscribers ===
        rospy.Subscriber("/displacement", Float32MultiArray, self.displacement_callback)
        rospy.Subscriber("/finger", Float32, self.finger_callback)

        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.franka_state_callback)
        rospy.Subscriber("/franka_gripper/joint_states", JointState, self.gripper_state_callback)

        rospy.Subscriber("/done", Float32, self.done_callback)

        # === Robot State ===
        self.initial_position = [0.43, 0.17, 0.10] 
        #self.initial_position = [0.4, 0.12, 0.08] 
        self.initial_quaternion = [0.93, -0.36, 0, 0]
        self.current_position = None
        self.current_quaternion = None
        self.target_position = None
        self.target_quaternion = None
        self.state_received = False

        self.left_qpos = 0.08
        self.is_clamp = False 
        self.initialized = False

        self.in_joint_mode = False
        self.in_impedance_mode = True
        self.last_motion_finished = False 
        self.prev_position = None
        self.motion_static_counter = 0

        # Initialize gripper action clients
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
       
        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()

        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)

        self.gripper_client = actionlib.SimpleActionClient(
            "/franka_gripper/move",
            MoveAction
        )

        #rospy.loginfo("Waiting for homing server...")
        self.homing_client.wait_for_server()
        self.homing_client.send_goal(HomingGoal())
        self.homing_client.wait_for_result()
        #rospy.loginfo("Gripper homed.")

        # === Start up ===
        rospy.sleep(1.0)
        rospy.loginfo("Waiting for initial robot state...")
        while not self.state_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Received initial state.")

        self.move_to_joint_start_position()
        rospy.sleep(1.0)

        # === publish current joint position to prevent jumping before switching to impedance control === #
        self.publish_current_joint_hold_position()
        rospy.sleep(1.0)

        self.switch_to_impedance_controller() 

        # goal = GraspGoal()
        # goal.width = 0.0                   # Fully close 0.079
        # goal.speed = 0.1                   # Moderate closing speed
        # goal.force = 80.0                  # Strong grip
        # goal.epsilon.inner = 0.0          # Accept small final gap
        # goal.epsilon.outer = 0.08          # Accept small overshoot
        # self.grasp_client.send_goal(goal)

        self.target_position = self.current_position.copy()
        self.target_quaternion = self.current_quaternion.copy()

        self.initialized = True 

        rospy.sleep(1.0)
        self.start_state_publisher_thread()

    def publish_current_joint_hold_position(self):
        pub = rospy.Publisher(
            '/position_joint_trajectory_controller/command',
            JointTrajectory,
            queue_size=10
        )
        rospy.sleep(1.0)

        traj = JointTrajectory()
        traj.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        point = JointTrajectoryPoint()
        point.positions = list(self.last_joint_position)
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration(1.0)

        traj.points = [point]
        pub.publish(traj)
        rospy.loginfo("Published hold trajectory to sync joint controller.")

    def wait_and_switch_to_impedance(self):
        self.wait_for_motion_finished(timeout=10.0)

        if not rospy.is_shutdown():
            rospy.loginfo("[WAIT DONE=0] Switching back to impedance control now.")
            self.switch_to_impedance_controller()
            self.in_joint_mode = False
            self.in_impedance_mode = True
            self.stop_pose_thread = False

    def wait_for_motion_finished(self, timeout=10.0):
        start_time = rospy.Time.now()
        rospy.loginfo("[WAIT] Checking for motion to finish...")

        while not rospy.is_shutdown():
            if self.last_motion_finished:
                rospy.loginfo("[WAIT] Motion is finished.")
                return
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("Timeout waiting for motion to finish before switching controller.")
                return
            rospy.loginfo_throttle(1.0, "[WAIT] Robot still moving...") 
            rospy.sleep(0.1)

    def switch_to_impedance_controller(self):
        rospy.wait_for_service('/controller_manager/switch_controller')

        pose_msg = self.create_pose_msg(self.current_position, self.current_quaternion)
        for _ in range(10):  # publish for a few cycles
            self.pose_pub.publish(pose_msg)
            rospy.sleep(0.02)

        try:
            switch = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            resp = switch(
                start_controllers=['cartesian_impedance_example_controller'],
                stop_controllers=['position_joint_trajectory_controller'],
                strictness=2
            )
            if resp.ok:
                rospy.loginfo("Switched to cartesian impedance controller.")
            else:
                rospy.logwarn("Failed to switch controllers!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        #rospy.loginfo("Waiting for homing server...")

    # def move_to_joint_start_position(self):
    #     pub = rospy.Publisher(
    #         '/position_joint_trajectory_controller/command',
    #         JointTrajectory,
    #         queue_size=10
    #     )
        
    #     joint_names = [
    #         'panda_joint1', 'panda_joint2', 'panda_joint3',
    #         'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
    #     ]

    #     # Replace this with your desired joint angles
    #     joint_motion = [0.0392, 0.1765, 0.3052, -2.5813, -0.1484, 2.8729, 1.9007]
    #     #joint_motion = [0.0392, 0.1765, 0.2252, -2.6613, -0.1484, 2.8729, 1.9007]

    #     traj = JointTrajectory()
    #     traj.joint_names = joint_names

    #     # === Use two points to prevent jumping ===#        
    #     point1 = JointTrajectoryPoint()
    #     point1.positions = joint_motion
    #     point1.velocities = [0.2] * 7
    #     point1.accelerations = [0.5] * 7
    #     point1.time_from_start = rospy.Duration(2.0) 

    #     point2 = JointTrajectoryPoint()
    #     point2.positions = joint_motion
    #     point2.velocities = [0.0] * 7
    #     point2.accelerations = [0.0] * 7
    #     point2.time_from_start = rospy.Duration(4.0)  # Give time to decelerate 

    #     traj.points.extend([point1, point2])

    #     # Wait for publisher to connect
    #     rospy.sleep(1.0)
    #     pub.publish(traj)
    #     rospy.loginfo("Published joint start motion")

    #     rospy.sleep(2.0)  # Wait to finish motion before switching controllers

    #     self.grasp_client.wait_for_server()
    #     self.move_client.wait_for_server()

    #     self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)

    #     self.homing_client.wait_for_server()
    #     self.homing_client.send_goal(HomingGoal())
    #     self.homing_client.wait_for_result()

    def move_to_joint_start_position(self):
        pub = rospy.Publisher(
            '/position_joint_trajectory_controller/command',
            JointTrajectory,
            queue_size=10
        )

        joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        joint_motion = [0.0392, 0.1765, 0.3052, -2.5813, -0.1484, 2.8729, 1.9007]

        traj = JointTrajectory()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_motion
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration(3.0)  # Adjust time for speed

        traj.points.append(point)

        rospy.sleep(1.0)  # Wait for publisher to connect
        pub.publish(traj)
        rospy.loginfo("Published joint start motion")

        rospy.sleep(3.0)  # Wait for motion to finish

        self.grasp_client.wait_for_server()
        self.move_client.wait_for_server()

        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        move_goal = MoveGoal()
        move_goal.width = 0.08
        move_goal.speed = 0.05
        self.move_client.send_goal(move_goal)
        self.move_client.wait_for_result()
        # self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        # self.homing_client.wait_for_server()
        # self.homing_client.send_goal(HomingGoal())
        # self.homing_client.wait_for_result()

    def done_callback(self, msg):
        if msg.data == 1.0 and not self.in_joint_mode:
            rospy.loginfo("[DONE=1] Waiting for impedance motion to finish...")

            self.stop_pose_thread = True
            self.target_position = None
            self.target_quaternion = None
            rospy.sleep(0.5)

            self.wait_for_motion_finished(timeout=5.0)

            rospy.loginfo("[DONE=1] Switching to joint control and moving to home position...")
            self.switch_to_joint_controller()

            # self.publish_current_joint_hold_position()
            # rospy.sleep(2.0)

            self.move_to_joint_start_position()
            self.in_joint_mode = True
            self.in_impedance_mode = False
            self.reset_done_pub.publish(Float32(1))

        elif msg.data == 0.0 and not self.in_impedance_mode:
            rospy.loginfo("[DONE=0] Received — will wait until motion is finished before switching to impedance control...")
            Thread(target=self.wait_and_switch_to_impedance).start()
            self.reset_done_pub.publish(Float32(0))

    def switch_to_joint_controller(self):
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            switch = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            resp = switch(
                stop_controllers=['cartesian_impedance_example_controller'],
                start_controllers=['position_joint_trajectory_controller'],
                strictness=2
            )
            if resp.ok:
                rospy.loginfo("Switched to joint position controller.")
            else:
                rospy.logwarn("Failed to switch to joint controller!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def franka_state_callback(self, msg):
        # Extract translation from column-major O_T_EE
        tf = np.array(msg.O_T_EE).reshape((4, 4), order='F')
        position = tf[:3, 3]
        rotation = tf[:3, :3]
        quaternion = R.from_matrix(rotation).as_quat()

        self.current_position = position
        self.current_quaternion = quaternion

        if self.prev_position is not None:
            delta = np.linalg.norm(position - self.prev_position)
            if delta < 0.0005:
                self.motion_static_counter += 1
            else:
                self.motion_static_counter = 0
        else:
            self.motion_static_counter = 0

        old_status = self.last_motion_finished 
        self.last_motion_finished = (self.motion_static_counter > 10)
        self.prev_position = position.copy()

        if self.last_motion_finished != old_status:
            rospy.loginfo(f"[MOTION] Motion static status changed: {self.last_motion_finished}")

        rot = R.from_quat(self.current_quaternion)
        euler = rot.as_euler('zyx', degrees=True)
        # rospy.loginfo_throttle(1.0, f"EEF Euler angles: yaw={euler[0]:.2f}, pitch={euler[1]:.2f}, roll={euler[2]:.2f}")

        self.last_joint_position = list(msg.q[:7])

        if not self.state_received:
            self.target_position = position.copy()
            self.target_quaternion = quaternion.copy()
            self.state_received = True

    def gripper_state_callback(self, msg):
        if len(msg.position) >= 2:
            # gripper width = left + right finger joint positions
            width = msg.position[0] + msg.position[1]
            #rospy.loginfo(f"Gripper width: {width:.3f}")
            self.current_gripper_width = width
            self.state_gripper_pub.publish(Float32(width))

    def create_pose_msg(self, position, quaternion):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "panda_link0"
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def displacement_callback(self, msg):
        if len(msg.data) != 6:
            rospy.logwarn("Invalid displacement data")
            return

        dx, dy, dz, roll, pitch, yaw = msg.data
        
        angle_deg = -60.0
        angle_rad = np.radians(angle_deg)

        dx_rotated = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        dy_rotated = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

        x = self.initial_position[0] + dx_rotated * 1.5 #0.9
        y = self.initial_position[1] + dy_rotated * 1.5 #0.9
        z = self.initial_position[2] + dz * 0.9 #0.7

        if not (0.25 <= x <= 0.66): return
        if not (-0.55 <= y <= 0.55): return
        if not (z <= 0.71): return

        #pitch_deg = pitch / 10.0 * 17.4
        pitch_deg = (pitch * 60.0)
        
        yaw_deg = (yaw * 30.0) 
        
        roll_deg = roll * 80

        # Build incremental rotation (relative rotation)
        increment_rot = R.from_euler('zyx', [yaw_deg, 0, 0], degrees=True)

        # Compose with the initial rotation
        initial_rot = R.from_quat(self.initial_quaternion)
        new_rot = increment_rot * initial_rot

        # Get the resulting quaternion
        quat = new_rot.as_quat()

        new_target_pos = np.array([x, y, z])
        new_target_quat = quat.tolist()

        if (
            self.target_position is None or
            np.linalg.norm(new_target_pos - self.target_position) > 1e-4 or
            np.linalg.norm(np.array(new_target_quat) - np.array(self.target_quaternion)) > 1e-4
        ):
            rospy.loginfo_throttle(2.0, f"Updated target pose to ({x:.3f}, {y:.3f}, {z:.3f})")
            self.target_position = new_target_pos
            self.target_quaternion = new_target_quat
        rospy.loginfo_throttle(2.0, f"Published pose to ({x:.3f}, {y:.3f}, {z:.3f})")
        
    def finger_callback(self, msg):
        if msg.data == 0:
            rospy.logwarn("Received finger data = 0; ignoring.")
            return

        new_qpos = msg.data
        if abs(new_qpos - self.left_qpos) > 0.055:  # significant change
            self.left_qpos = new_qpos
            scale = 0.6 #0.4
            threshold = 0.04
            width_scaled = self.left_qpos * scale
            should_open = (width_scaled > threshold)

            if should_open:
                if self.is_clamp:
                    rospy.loginfo("Opening gripper.")
                    Thread(target=self.open_gripper, args=(width_scaled,)).start()
                    self.is_clamp = False
            else:
                if not self.is_clamp:
                    rospy.loginfo("Clamping gripper.")
                    Thread(target=self.clamp_gripper).start()
                    self.is_clamp = True

    def open_gripper(self, width):
        # === Do not use MoveGoal ===#
        # Operation success if d is in this range->
        # width - epsilon.inner < d < width + epsilon.outer (d is the actual grasp width)

        goal = GraspGoal()
        goal.width = 0.079                   # Fully close 0.079
        goal.speed = 0.1                   # Moderate closing speed
        goal.force = 80.0                  # Strong grip
        goal.epsilon.inner = 0.0          # Accept small final gap
        goal.epsilon.outer = 0.08          # Accept small overshoot

        # goal = MoveGoal()
        # goal.width = 0.079
        # print("width",width)
        # goal.speed = 0.1
        # self.move_client.send_goal(goal)

        self.grasp_client.send_goal(goal)

        rospy.loginfo("[Gripper] Sending open goal...")
        #self.grasp_client.send_goal(goal)

        finished = self.grasp_client.wait_for_result(rospy.Duration(5.0))
        if not finished:
            rospy.logwarn("[Gripper] Grasp action timeout.")
            return False

        result = self.grasp_client.get_result()
        if result:
            rospy.loginfo(f"[Gripper] Open success: {result.success}")
            return result.success
        else:
            rospy.logwarn("[Gripper] No result returned.")
            return False

    def clamp_gripper(self):
        # === Do not use MoveGoal ===#
        # Operation success if d is in this range->
        # width - epsilon.inner < d < width + epsilon.outer (d is the actual grasp width)

        goal = GraspGoal()
        goal.width = 0.0                   # Fully close
        goal.speed = 0.1                   # Moderate closing speed
        goal.force = 80.0                  # Strong grip
        goal.epsilon.inner = 0.04          # Accept small final gap
        goal.epsilon.outer = 0.08          # Accept small overshoot

        # goal = MoveGoal()
        # goal.width = 0.0
        # goal.speed = 0.1
        # self.move_client.send_goal(goal)

        self.grasp_client.send_goal(goal)

        rospy.loginfo("[Gripper] Sending grasp goal...")
        #self.grasp_client.send_goal(goal)

        finished = self.grasp_client.wait_for_result(rospy.Duration(5.0))
        if not finished:
            rospy.logwarn("[Gripper] Grasp action timeout.")
            return False

        result = self.grasp_client.get_result()
        if result:
            rospy.loginfo(f"[Gripper] Clamp success: {result.success}")
            return result.success
        else:
            rospy.logwarn("[Gripper] No result returned.")
            return False
        
    def start_state_publisher_thread(self):
        def loop():
            rate = rospy.Rate(60)  
            while not rospy.is_shutdown():
                pose_msg = self.create_pose_msg(self.current_position, self.current_quaternion)
                self.state_pose_pub.publish(pose_msg)
                self.state_gripper_pub.publish(Float32(self.current_gripper_width))

                if self.target_position is not None and self.target_quaternion is not None:
                    target_msg = self.create_pose_msg(self.target_position, self.target_quaternion)
                    self.pose_pub.publish(target_msg)

                rate.sleep()

        thread = Thread(target=loop)
        thread.daemon = True
        thread.start()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = DisplacementToPoseController()
        controller.run()
    except rospy.ROSInterruptException:
        pass



