#!/usr/bin/env python3
# coding: utf-8

import argparse
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from franka_dataset import FrankaDataset


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('franka_recorder')
        self.bridge = CvBridge()

        # 最新感測資料
        self.latest_pose = None
        self.latest_width = None
        self.latest_eye = None
        self.latest_agent = None

        # 訂閱真實機器人 topic
        self.create_subscription(Float64MultiArray,
                                 '/robot0_eef_pose',
                                 self.pose_cb, 10)
        self.create_subscription(Float32MultiArray,
                                 '/robot0_gripper_width',
                                 self.grip_cb, 10)
        self.create_subscription(Image,
                                 '/robot0_eye_in_hand_image',
                                 self.eye_cb, 10)
        self.create_subscription(Image,
                                 '/robot0_agentview_image',
                                 self.agent_cb, 10)

    def pose_cb(self, msg: Float64MultiArray):
        """末端執行器位姿 [x,y,z,roll,pitch,yaw]"""
        self.latest_pose = list(msg.data)

    def grip_cb(self, msg: Float32MultiArray):
        """夾爪寬度"""
        if msg.data:
            self.latest_width = float(msg.data[0])

    def eye_cb(self, msg: Image):
        """Eye‑in‑Hand 相機影像"""
        try:
            self.latest_eye = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Eye-in-hand 影像轉換錯誤: {e}")

    def agent_cb(self, msg: Image):
        """Agent‑view 相機影像"""
        try:
            self.latest_agent = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Agent-view 影像轉換錯誤: {e}")

    def record_episode(self, ep_idx, args):
        """
        錄製一個 episode。
        返回:
          status: 'ok'|'exit'|'redo'
          data: tuple(poses, widths, eyes, agents) or None
        """
        fps = args.fps
        duration = args.episode_time_s
        total_frames = int(fps * duration)
        dt_ms = int(1000 / fps)

        poses, widths = [], []
        eyes, agents = [], []

        self.get_logger().info(f"[Ep{ep_idx:03d}] 開始錄製 "
                               f"{duration}s ({total_frames} frames @ {fps}Hz)")

        # 開啟顯示窗口
        cv2.namedWindow("Agent View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Eye-in-Hand", cv2.WINDOW_NORMAL)

        for i in range(total_frames):
            # 讓 ROS 回呼函式更新最新資料
            rclpy.spin_once(self, timeout_sec=0.0)

            # 擷取並補齊資料
            pose = self.latest_pose or [0.0]*6
            width = self.latest_width or 0.0
            eye   = self.latest_eye.copy() if self.latest_eye is not None else \
                    np.full((480,640,3), 255, dtype=np.uint8)
            agent = self.latest_agent.copy() if self.latest_agent is not None else \
                    np.full((480,640,3), 255, dtype=np.uint8)

            poses.append(pose)
            widths.append(width)
            eyes.append(eye)
            agents.append(agent)

            # 顯示
            cv2.imshow("Agent View", agent)
            cv2.imshow("Eye-in-Hand", eye)

            key = cv2.waitKey(dt_ms) & 0xFF
            if key == 27:   # Esc
                self.get_logger().info("按下 ESC，退出整個錄製流程")
                cv2.destroyAllWindows()
                return 'exit', None
            elif key == 83: # → 右方向鍵
                self.get_logger().info("按下 →，提前結束本集並保存已錄影像")
                break
            elif key == 81: # ← 左方向鍵
                self.get_logger().info("按下 ←，重新錄製本集")
                cv2.destroyAllWindows()
                return 'redo', None

        cv2.destroyAllWindows()
        self.get_logger().info(f"[Ep{ep_idx:03d}] 錄製完成，共 {len(poses)} 幀")
        return 'ok', (poses, widths, eyes, agents)


def main():
    parser = argparse.ArgumentParser(
        description="Franka ROS2 實機資料錄製腳本"
    )
    parser.add_argument('--repo_id',       type=str,   required=True)
    parser.add_argument('--num_episodes',  type=int,   default=1)
    parser.add_argument('--episode_time_s',type=float, default=10.0)
    parser.add_argument('--fps',           type=float, default=30.0)
    parser.add_argument('--reset_time_s',  type=float, default=5.0)
    parser.add_argument('--chunk_size',    type=int,   default=100)
    parser.add_argument('--single_task',   dest='task_name',
                        type=str, required=True)
    args = parser.parse_args()

    rclpy.init()
    recorder = DataRecorderNode()

    dataset = FrankaDataset(
        repo_id=args.repo_id,
        fps=args.fps,
        chunk_size=args.chunk_size,
        task_name=args.task_name
    )

    ep = 0
    while ep < args.num_episodes:
        status, data = recorder.record_episode(ep, args)

        if status == 'exit':
            break
        if status == 'redo':
            # 不增加 ep，重新錄製
            continue

        # 正常結束或提前結束 (→)
        poses, widths, eyes, agents = data
        arr = np.array(poses)
        eef_pos = arr[:, :3]
        eef_ori = arr[:, 3:]
        gripper = np.array(widths)

        dataset.add_episode(
            episode_index    = ep,
            eef_positions    = eef_pos,
            eef_orientations = eef_ori,
            gripper_widths   = gripper,
            view1_frames     = np.stack(agents, axis=0),  # agentview 存為 view1
            view2_frames     = np.stack(eyes, axis=0)     # eye-in-hand 存為 view2
        )

        # Reset 環境階段
        start = time.time()
        self_logger = recorder.get_logger()
        self_logger.info(f"[Ep{ep:03d}] 重置環境，等待 {args.reset_time_s}s…"
                         "（按 → 可跳過）")

        while time.time() - start < args.reset_time_s:
            key = cv2.waitKey(100) & 0xFF
            if key == 83:  # → 跳過剩餘重置
                self_logger.info("按下 →，跳過重置，直接開始下一集")
                break

        ep += 1

    # finalize
    dataset.finalize()
    recorder.get_logger().info(f"資料集已建立於 {dataset.root}")

    recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



"""
import argparse
import os
import time
import json
from threading import Event

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from franka_dataset import FrankaDataset  

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('franka_recorder')
        self.bridge = CvBridge()

        # 緩存最新資料
        self.latest_pose = None
        self.latest_width = None
        self.latest_eye = None
        self.latest_agent = None
        self.stop_event = Event()

        # 訂閱 topics
        self.create_subscription(Float64MultiArray,
                                 '/robot0_eef_pose',
                                 self.pose_cb, 10)
        self.create_subscription(Float32MultiArray,
                                 '/robot0_gripper_width',
                                 self.grip_cb, 10)
        self.create_subscription(Image,
                                 '/robot0_eye_in_hand_image',
                                 self.eye_cb, 10)
        self.create_subscription(Image,
                                 '/robot0_agentview_image',
                                 self.agent_cb, 10)

    def pose_cb(self, msg: Float64MultiArray):
        """末端位姿 [x,y,z,roll,pitch,yaw]"""
        self.latest_pose = list(msg.data)

    def grip_cb(self, msg: Float32MultiArray):
        """夾爪寬度"""
        if msg.data:
            self.latest_width = float(msg.data[0])

    def eye_cb(self, msg: Image):
        """Eye-in-Hand 相機圖"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_eye = img
        except Exception as e:
            self.get_logger().error(f"eye convert error: {e}")

    def agent_cb(self, msg: Image):
        """Agent View 相機圖"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_agent = img
        except Exception as e:
            self.get_logger().error(f"agent convert error: {e}")
    
    def countdown(self, secs: int):
        cv2.namedWindow("Countdown", cv2.WINDOW_NORMAL)
        for t in range(secs, 0, -1):
            print(f"Starting in {t}...")
            overlay = np.zeros((100,400,3), dtype=np.uint8)
            cv2.displayOverlay("Countdown", f"{t}", 1000)
            cv2.imshow("Countdown", overlay)
            if cv2.waitKey(1000) & 0xFF == 27:
                self.stop_event.set()
                break
        cv2.destroyWindow("Countdown")

    def record_episode(self, ep_idx, args):
        fps = args.fps
        duration = args.episode_time_s
        total_frames = int(fps * duration)

        poses, widths = [], []
        eyes, agents = [], []

        self.get_logger().info(f"[Ep{ep_idx:03d}] 開始錄製 {duration}s（{total_frames} frames @ {fps}Hz）")

        # 逐幀錄製
        t0 = time.time()
        for i in range(total_frames):
            # 處理停止標誌
            if self.stop_event.is_set():
                return 'exit', None

            # 等待到下一幀
            target = t0 + i / fps
            now = time.time()
            if target > now:
                time.sleep(target - now)

            # 接收 ROS callback
            rclpy.spin_once(self, timeout_sec=0.0)

            # 取最新資料，若無則填零或白板
            pose = self.latest_pose or [0.0]*6
            width = self.latest_width or 0.0
            eye   = (self.latest_eye.copy() if self.latest_eye is not None
                     else np.full((480,640,3),255,dtype=np.uint8))
            agent = (self.latest_agent.copy() if self.latest_agent is not None
                     else np.full((480,640,3),255,dtype=np.uint8))

            poses.append(pose)
            widths.append(width)
            eyes.append(eye)
            agents.append(agent)

            # 顯示並偵測按鍵
            cv2.imshow("Agent View", agent)
            cv2.imshow("Eye-in-Hand", eye)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                self.get_logger().info("收到 Esc - 退出錄製")
                self.stop_event.set()
                return 'exit', None
            elif key == ord('s'):
                self.get_logger().info("收到 's' - 停止並保存本集")
                break
            elif key == ord('r'):
                self.get_logger().info("收到 'r' - 重新錄製本集")
                return 'redo', None

        cv2.destroyAllWindows()
        self.get_logger().info(f"[Ep{ep_idx:03d}] 錄製完成，共 {len(poses)} 幅")
        return 'ok', (poses, widths, eyes, agents)

def main():
    parser = argparse.ArgumentParser(description="Franka ROS2 錄製節點")
    parser.add_argument('--repo_id',      type=str,   required=True)
    parser.add_argument('--num_episodes', type=int,   default=1)
    parser.add_argument('--episode_time_s', type=float, default=10.0)
    parser.add_argument('--fps',          type=float, default=30.0)
    parser.add_argument('--reset_time_s', type=float, default=5.0)
    parser.add_argument('--chunk_size',   type=int,   default=100)
    parser.add_argument('--single_task',  dest='task_name', type=str, required=True)
    args = parser.parse_args()

    # 初始化 ROS2
    rclpy.init()
    recorder = DataRecorderNode()

    # 建立資料集
    dataset = FrankaDataset(
        repo_id=args.repo_id,
        fps=args.fps,
        chunk_size=args.chunk_size,
        task_name=args.task_name
    )

    ep = 0
    while ep < args.num_episodes:
        status, data = recorder.record_episode(ep, args)
        if status == 'exit':
            break
        if status == 'redo':
            continue

        poses, widths, eyes, agents = data
        
        arr = np.array(poses)                 # shape (N,6)
        eef_positions    = arr[:, :3]         # shape (N,3)
        eef_orientations = arr[:, 3:]         # shape (N,3)
        gripper_widths   = np.array(widths)   # shape (N,)

        dataset.add_episode(
            episode_index    = ep,
            eef_positions    = eef_positions,
            eef_orientations = eef_orientations,
            gripper_widths   = gripper_widths,
            view1_frames     = np.stack(agents),  
            view2_frames     = np.stack(eyes)    
        )

        # reset 休息
        if ep < args.num_episodes-1 and args.reset_time_s > 0:
            recorder.get_logger().info(f"[Ep{ep:03d}] Reset 等待 {args.reset_time_s}s …")
            time.sleep(args.reset_time_s)

        ep += 1

    # 完成並輸出 meta
    dataset.finalize()
    recorder.get_logger().info(f"資料集已建立於 {dataset.root}")

    # 清理
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

"""

