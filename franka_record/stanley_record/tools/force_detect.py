#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
import numpy as np
import signal
import sys

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("Ctrl+C detected. Shutting down gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class FrankaControlledDown:
    def __init__(self):
        rospy.init_node("franka_controlled_down", anonymous=False)

        # Parameters
        self.force_threshold_N = 10.0
        self.down_step_m = 0.0002  # how much to reduce Z each iteration
        self.publish_rate_hz = 50

        # State
        self.stop_due_to_force = False
        self.current_z = 0.10  # Start Z height

        # Publisher
        self.pose_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose",
            PoseStamped,
            queue_size=1
        )

        # Subscriber to force estimates
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self.force_callback
        )

        # Initial PoseStamped message
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "panda_link0"
        self.pose_msg.pose.position.x = 0.393
        self.pose_msg.pose.position.y = 0.103
        self.pose_msg.pose.position.z = self.current_z
        self.pose_msg.pose.orientation.x = 0.930
        self.pose_msg.pose.orientation.y = -0.365
        self.pose_msg.pose.orientation.z = 0.0
        self.pose_msg.pose.orientation.w = 0.002

        rospy.loginfo("Starting controlled downward movement...")
        self.main_loop()

    def force_callback(self, msg):
        # Extract force vector
        force_vec = np.array(msg.O_F_ext_hat_K[:3])
        norm = np.linalg.norm(force_vec)

        if norm > self.force_threshold_N:
            if not self.stop_due_to_force:
                rospy.logwarn("Force threshold exceeded (%.2f N). Stopping downward motion." % norm)
                self.stop_due_to_force = True
        else:
            if self.stop_due_to_force:
                rospy.loginfo("Force back to normal (%.2f N). Resuming motion." % norm)
                self.stop_due_to_force = False

    def main_loop(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and not shutdown_requested:
            if not self.stop_due_to_force:
                # Move down
                self.current_z -= self.down_step_m
                if self.current_z < 0.0:
                    rospy.logwarn("Reached Z=0.0. Stopping further motion.")
                    self.current_z = 0.0

            self.pose_msg.pose.position.z = self.current_z
            self.pose_msg.header.stamp = rospy.Time.now()
            self.pose_pub.publish(self.pose_msg)

            rate.sleep()

if __name__ == "__main__":
    try:
        FrankaControlledDown()
    except rospy.ROSInterruptException:
        pass
