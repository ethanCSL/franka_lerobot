import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from argparse import ArgumentParser
import numpy as np
import time
import cv2
import pyrealsense2 as rs
from frankx import Robot, Gripper
from scipy.spatial.transform import Rotation as R

def matrix_to_quaternion(matrix):
    """將轉換矩陣轉換為四元數"""
    rot = R.from_matrix(np.array(matrix).reshape(4, 4)[:3, :3])
    return rot.as_quat().tolist()

def extract_translation(matrix):
    """從轉換矩陣中提取平移向量"""
    mat = np.array(matrix).reshape((4, 4), order='F')
    return mat[0:3, 3].tolist()

class FrankaStatePublisher(Node):
    def __init__(self, robot_ip, usb_camera_index=0):
        """
        初始化Franka機器人狀態發布節點
        
        Args:
            robot_ip: Franka機器人的IP地址
            usb_camera_index: USB攝影機的索引
        """
        super().__init__('franka_state_publisher')
        
        # 初始化機器人
        try:
            self.robot = Robot(robot_ip)
            self.robot.set_default_behavior()
        except Exception as e:
            self.get_logger().error(f"無法連接到機器人: {e}")
            raise

        # 初始化夾爪
        try:
            self.gripper = Gripper(robot_ip)
            self.gripper_enabled = True
        except Exception as e:
            self.get_logger().warn(f"夾爪不可用: {e}")
            self.gripper_enabled = False

        # 初始化RealSense攝影機
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_pipeline.start(self.rs_config)
        except Exception as e:
            self.get_logger().error(f"無法初始化RealSense攝影機: {e}")
            self.rs_pipeline = None

        # 初始化USB攝影機
        try:
            self.cam_usb = cv2.VideoCapture(usb_camera_index)
            self.cam_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cam_usb.isOpened():
                self.get_logger().warn(f"無法打開USB攝影機(索引: {usb_camera_index})")
                self.cam_usb = None
        except Exception as e:
            self.get_logger().warn(f"USB攝影機錯誤: {e}")
            self.cam_usb = None

        # 只創建需要的四個發布者
        self.eef_pose_pub = self.create_publisher(Float64MultiArray, 'robot0_eef_pose', 10)
        self.gripper_width_pub = self.create_publisher(Float32MultiArray, 'robot0_gripper_width', 10)
        self.agentview_pub = self.create_publisher(Image, 'robot0_agentview_image', 10)
        self.eyeinhand_pub = self.create_publisher(Image, 'robot0_eye_in_hand_image', 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.05, self.publish_robot_state)
        
        self.get_logger().info("Franka狀態發布節點已初始化")

    def publish_robot_state(self):
        """發布機器人狀態和攝影機圖像"""
        # 發布末端執行器位姿
        try:
            state = self.robot.read_once()
            
            # 處理末端執行器數據
            eef_pos = extract_translation(state.O_T_EE)
            eef_quat = matrix_to_quaternion(state.O_T_EE)
            
            # 將四元數轉換為歐拉角 (roll, pitch, yaw)
            rot = R.from_quat(eef_quat)
            euler = rot.as_euler('xyz', degrees=False)  # 使用弧度制，順序為 roll, pitch, yaw
            
            # 創建 Float64MultiArray 消息，包含位置和歐拉角
            pose_array = Float64MultiArray()
            pose_array.data = [eef_pos[0], eef_pos[1], eef_pos[2], euler[0], euler[1], euler[2]]
            
            # 發布
            self.eef_pose_pub.publish(pose_array)
        except Exception as e:
            self.get_logger().error(f"讀取機器人狀態錯誤: {e}")

        # 處理夾爪數據
        try:
            if self.gripper_enabled:
                curr_gripper_width = self.gripper.width()
            else:
                curr_gripper_width = 0.0

            self.gripper_width_pub.publish(Float32MultiArray(data=[curr_gripper_width]))
        except Exception as e:
            self.get_logger().warn(f"讀取夾爪狀態錯誤: {e}")

        # 發布RealSense攝影機圖像
        if self.rs_pipeline:
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=200)
                color_frame = frames.get_color_frame()

                if color_frame:
                    rs_img = np.asanyarray(color_frame.get_data())
                    rs_msg = self.bridge.cv2_to_imgmsg(rs_img, encoding='bgr8')
                    self.eyeinhand_pub.publish(rs_msg)
            except Exception as e:
                self.get_logger().warn(f"RealSense攝影機錯誤: {e}")

        # 發布USB攝影機圖像
        if self.cam_usb and self.cam_usb.isOpened():
            try:
                ret, usb_img = self.cam_usb.read()
                if ret:
                    usb_msg = self.bridge.cv2_to_imgmsg(usb_img, encoding='bgr8')
                    self.agentview_pub.publish(usb_msg)
            except Exception as e:
                self.get_logger().warn(f"USB攝影機錯誤: {e}")

    def destroy_node(self):
        """清理資源"""
        self.get_logger().info("關閉Franka狀態發布節點")
        
        # 關閉攝影機
        if hasattr(self, 'cam_usb') and self.cam_usb:
            self.cam_usb.release()
            
        if hasattr(self, 'rs_pipeline') and self.rs_pipeline:
            self.rs_pipeline.stop()
            
        super().destroy_node()

def main(args=None):
    """主函數"""
    rclpy.init(args=args)
    
    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.0.2', help='機器人的FCI IP')
    parser.add_argument('--camera-index', type=int, default=0, help='USB攝影機索引')
    parsed_args = parser.parse_args()
    
    try:
        node = FrankaStatePublisher(parsed_args.host, parsed_args.camera_index)
        rclpy.spin(node)
    except Exception as e:
        print(f"錯誤: {e}")
    finally:
        # 確保無論如何都會清理
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


