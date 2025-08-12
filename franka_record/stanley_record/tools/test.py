#!/usr/bin/env python
import rospy
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal

# 初始寬度 (8 cm)
initial_width = 0.08  # meters
delta = 4.7935259e-07  # 單次減少量（比較合理）

client = None

def send_gripper_goal(width):
    """ 非阻塞式發送 gripper width """
    goal = MoveGoal()
    goal.width = width
    goal.speed = 0.1
    client.send_goal(goal)

if __name__ == '__main__':
    rospy.init_node('test_gripper_replay')

    client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    rospy.loginfo("Waiting for gripper action server...")
    client.wait_for_server()
    rospy.loginfo("Gripper action server connected.")

    current_width = initial_width
    rate = rospy.Rate(20)  # 20Hz

    for i in range(20000):
        rospy.loginfo(f"[{i+1}/200] Sending width: {current_width:.5f} m")
        send_gripper_goal(current_width)
        current_width -= delta
        rate.sleep()

    rospy.loginfo("Gripper replay complete.")
